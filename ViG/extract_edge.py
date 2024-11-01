"""
nohup python vision_graph/ViG/extract_edge.py > vision_graph/ViG/log/20241026_140511/edge.txt 2>&1 &
"""
########################################################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from model.vig import ViG

########################################################################################################################

# デバイスの確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

########################################################################################################################

# # transformの定義
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# test_set = datasets.CIFAR10(
#     root = "/scr/data/CIFAR10",  train = False,
#     download = True, transform = transform)

# # テストデータから写真データを3枚取り出す
# images, labels = zip(*[test_set[i] for i in range(3)])

# # 3枚の画像を描画
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# for i, (img, ax) in enumerate(zip(images, axs)):
#     # TensorをNumpy配列に変換して描画
#     img = img.permute(1, 2, 0).numpy()
#     ax.imshow(img)

#     # グリッド線を引く設定
#     ax.grid(True, which='both', color='black', linestyle='--', linewidth=0.5)
#     ax.set_xticks(np.arange(0, 224, 16))  # 16ピクセルごとのX軸のグリッド線
#     ax.set_yticks(np.arange(0, 224, 16))

#     ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#     ax.set_title(f'pic: {i} \nLabel: {labels[i]}')
#     ax.axis('on')
# os.makedirs(os.path.dirname("/scr/vision_graph/ViG/log/edge_picture/1_edge_picture_1.png"), exist_ok=True)
# plt.savefig("/scr/vision_graph/ViG/log/edge_picture/1_edge_picture_1.png")

########################################################################################################################

# torch.set_printoptions(threshold=torch.inf)

# transformの定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_set = datasets.CIFAR10(
    root = "/scr/data/CIFAR10",  train = False,
    download = True, transform = transform)

batch_size = 1
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

net = ViG(10).to(device)
net.load_state_dict(torch.load('/scr/vision_graph/ViG/log/20241026_140511/ViG.pth'))

def extract_edge_index(input_data):
    net.eval()
    with torch.no_grad():
        edge_index_output = net(input_data, True)
    return edge_index_output

# テストデータから画像データを取得して推論
for j, (batch_data, labels) in enumerate(test_loader):
    if j == 0:
        batch_data = batch_data.to(device)
        edge_index = extract_edge_index(batch_data)  # edge_indexを取得

        for i, edge_idx in enumerate(edge_index):
            print(f"Edge Index {i} shape: {edge_idx.shape}")
            print(f"Edge Index {i}: {edge_idx}")

        break