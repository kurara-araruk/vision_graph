
"""
ViGのシンプルな実装
https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
"""

###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import torch.nn.init as init

from einops import repeat
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath

###############################################################################

# stem
# image -> patch

class Stem(nn.Module):
    """
    ViG公式実装は、Conv2d, BatchNorm2d, ReLU を 5層 積み上げて、パッチ分割とチャネル数の増加を行なっている。
    今回のシンプル実装は、ViTベースのパッチ分割を行う。なお、配列変換はViG風で実装する。
    もっとシンプルにするには、kernel_size, stride = patch_size にすればいい？
    """
    def __init__(self, img_size=224, patch_size=16, out_dim=768):
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size."

        in_dim = patch_size**2 * 3

        # ViG風 : [B, C, H, W] -> [B, 3*16*16, 14, 14]
        self.Patching = Rearrange("b c (h ph) (w pw) -> b (c ph pw) h w", ph = patch_size, pw = patch_size)
        # ViT風 : [B, C, H, W] -> [B, N(パッチ数), C*P^2]
        # self.Patching = Rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)", ph = patch_size, pw = patch_size)

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.Patching(x)
        x = self.conv(x)
        x = self.bn(x)

        return x


###############################################################################

# knn
# patch -> graph

class Knn(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.k = k

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()    # [B, C:768, H:14, W:14] -> [B, C:768, 196, 1]
        x = F.normalize(x, p=2.0, dim=1)    # [B, C, 196, 1] に対して C次元 方向に沿って正規化

        with torch.no_grad():
            x = x.transpose(2, 1).squeeze(-1)   # [B, C, 196, 1] -> [B, 196, C, 1] -> [B, 196, C]
            batch_size, n_points, n_dims = x.shape

            # すべての点同士のユークリッド距離の2乗
            # d(i, j) = ||x_i - x_j||^2 = x_i^2 - ( 2 * x_i * x_j ) + x_j^2
            x_inner = -2*torch.matmul(x, x.transpose(2, 1))     # - 2 * x_i * x_j  ->  [batch_size, n_points, n_points]
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)    # x_i^2  ->  [batch_size, n_points, 1]
            dist = x_square + x_inner + x_square.transpose(2, 1)     #  ->  [batch_size, n_points, n_points]

            _, nn_idx = torch.topk(-dist, k=self.k)    #  距離が近い点 k 個を取得するため、dist をマイナスにする。  ->  [batch_size, n_points, k]

            center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, self.k, 1).transpose(2, 1)   # ->  [batch_size, n_points, k]
            edge_index = torch.stack((nn_idx, center_idx), dim=0)   # ->  [2, batch_size, n_points*2, k]

        return edge_index


###############################################################################

# GNN (MRConv2d)
# graph -> graph'

def batched_index_select(x, idx):
    """
    
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()

    return feature

class GNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels*2),
            nn.GELU(),
            nn.Conv2d(in_channels*2, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, edge_idx):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()     # [B, C, H, W] -> [b, c, n, 1]

        x_i = batched_index_select(x, edge_idx[1])

        x_j = batched_index_select(x, edge_idx[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)

        b, c, n, _ = x.shape

        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)

        x = self.conv(x)

        return x.reshape(B, -1, H, W).contiguous()


###############################################################################

# Grapher
# patch -> [knn] -> graph -> [GNN] -> graph'

class Grapher(nn.Module):
    def __init__(self, in_channels, k=9, drop_path=0.0):
        super().__init__()

        # 全結合層
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        # グラフ化
        self.knn = Knn(k)

        # グラフ畳み込み
        self.graph_conv = GNN(in_channels, in_channels * 2)

        # 全結合層
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        edge_idx = self.knn(x)
        x = self.graph_conv(x, edge_idx)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x


###############################################################################

# FFN
# graph' -> [FFN] -> graph''

class FFN(nn.Module):
    """
    オーバースムージング現象を緩和するためのFFN層
    """
    def __init__(self, in_channels, hidden_channels, drop_path):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
        )
        self.gelu = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut

        return x


###############################################################################

# ViG
# image -> [stem] -> patch -> ( [Grapher] -> graph' -> [FFN] )*n_loop -> prediction

class ViG(nn.Module):
    def __init__(self,
                 n_classes,
                 img_size: int = 224,
                 patch_size: int = 16,
                 stem_out_dim: int = 768,
                 k: int = 9,
                 drop_path: float = 0.0,
                 n_blocks: int = 12,
                 ):
        super().__init__()

        self.n_blocks = n_blocks

        self.stem = Stem(img_size=img_size, patch_size=patch_size, out_dim=stem_out_dim)

        num_k = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]

        self.vig_blocks = nn.Sequential(*[nn.Sequential(Grapher(in_channels=stem_out_dim, k=num_k[i], drop_path=dpr[i]),
                                      FFN(in_channels=stem_out_dim, hidden_channels=stem_out_dim * 4, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = nn.Sequential(nn.Conv2d(stem_out_dim, 1024, 1, bias=True),
                                        nn.BatchNorm2d(1024),
                                        nn.GELU(),
                                        nn.Dropout(drop_path),
                                        nn.Conv2d(1024, n_classes, 1, bias=True))

    def forward(self, x):
        x = self.stem(x)

        for i in range(self.n_blocks):
            x = self.vig_blocks[i](x)

        x = F.adaptive_avg_pool2d(x, 1)

        return self.prediction(x).squeeze(-1).squeeze(-1)
