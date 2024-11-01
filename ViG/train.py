"""
timestamp=$(date +%Y%m%d_%H%M%S) && mkdir -p scr/vision_graph/ViG/log/$timestamp && nohup python scr/vision_graph/ViG/train.py --logdir=scr/vision_graph/ViG/log/$timestamp > scr/vision_graph/ViG/log/$timestamp/tqdm.log 2>&1 &
"""
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchinfo import summary

from model.vig import ViG

########################################################################################################################

# 引数パーサー
parser = argparse.ArgumentParser(description='ログディレクトリを指定できるトレーニングスクリプト')
parser.add_argument('--logdir', type=str, default=None, help='ログや出力を保存するディレクトリ')
args = parser.parse_args()

########################################################################################################################

# fit関数の定義
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history, batch_size, scheduler=None, save_dir='log', save_name='output.log'):

    print("batch_size: ", batch_size)
    print("num_epochs: ", num_epochs)
    print(net)
    print(criterion)
    print(optimizer)
    print(scheduler)
    summary(net, input_size=(batch_size, 3, net.img_size, net.img_size), col_names=["input_size", "output_size", "num_params"])

    # tqdmのインポート
    from tqdm import tqdm

    # ログディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ログファイルのパスを設定
    filepath = os.path.join(save_dir, save_name)

    base_epochs = len(history)

    for epoch in range(base_epochs, base_epochs+num_epochs):
        # 時間計測スタート
        start = time.perf_counter()
        # 記録用
        n_train_acc, n_val_acc = 0,0
        train_loss, val_loss = 0,0
        n_train, n_test = 0,0

        # 訓練フェーズ
        net.train()
        for inputs, labels in tqdm(train_loader):

            train_batch_size = len(labels)
            n_train += train_batch_size

            # GPUへ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 学習
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            predicted = torch.max(outputs, 1)[1]

            # 記録用
            train_loss += loss.item() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()

        # 推論フェーズ
        net.eval()
        with torch.no_grad():
            for inputs_test, labels_test in tqdm(test_loader):

                test_batch_size = len(labels_test)
                n_test += test_batch_size

                # GPUヘ転送
                inputs_test = inputs_test.to(device)
                labels_test = labels_test.to(device)

                # 推論
                outputs_test = net(inputs_test)
                loss_test = criterion(outputs_test, labels_test)
                predicted_test = torch.max(outputs_test, 1)[1]

                # 記録用
                val_loss +=  loss_test.item() * test_batch_size
                n_val_acc +=  (predicted_test == labels_test).sum().item()

        # 時間計測終了
        end = time.perf_counter()
        epoch_time = end - start
        hours, rem = divmod(epoch_time, 3600)
        minutes, seconds = divmod(rem, 60)
        # tqdm形式に合わせてフォーマット
        if hours > 0:
            # 1時間以上の場合は "時間:分:秒" 形式
            formatted_time = "{:d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))
        else:
            # 1時間未満の場合は "分:秒" 形式
            formatted_time = "{:d}:{:02d}".format(int(minutes), int(seconds))
        # 精度計算
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        # 結果表示
        log_message = f'Epoch [{epoch +1}/{num_epochs + base_epochs}], time: {formatted_time}, loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}\n'

        # ログファイルを開く（追記モード）
        with open(filepath, 'a') as f:
            f.write(log_message)

        print(log_message)
        # 記録
        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])
        history = np.vstack((history, item))

        if scheduler is not None:
            scheduler.step()


    return history

########################################################################################################################

# 学習ログ出力関数の定義
def evaluate_history(history, save_dir, save_name="loss_acc.png"):
    # 保存先フォルダが存在しない場合、作成する
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, save_name)

    num_epochs = len(history)
    unit = num_epochs / 10

    # 一つの図にサブプロットを作成
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # 2行1列のサブプロット

    # 損失曲線の表示
    axes[0].plot(history[:,0], history[:,1], 'b', label='train loss')
    axes[0].plot(history[:,0], history[:,3], 'k', label='val loss')
    axes[0].set_xticks(np.arange(0, num_epochs+1, unit))
    axes[0].set_xlabel('num epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title('loss')
    axes[0].legend()
    axes[0].grid(True)

    # 精度曲線の表示
    axes[1].plot(history[:,0], history[:,2], 'b', label='train acc')
    axes[1].plot(history[:,0], history[:,4], 'k', label='val acc')
    axes[1].set_xticks(np.arange(0, num_epochs+1, unit))
    axes[1].set_xlabel('num epoch')
    axes[1].set_ylabel('acc')
    axes[1].set_title('acc')
    axes[1].legend()
    axes[1].grid(True)

    # レイアウトの調整
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

########################################################################################################################

# 乱数固定関数の定義
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

########################################################################################################################

# デバイスの確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# transformの定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# データの読み込み
data_path = "/container/data/imagenet"
train_set = ImageFolder(f"{data_path}/train", transform = transform)
val_set = ImageFolder(f"{data_path}/val", transform = transform)

batch_size = 100
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ログディレクトリのロジック
if args.logdir is not None:
    log_dir = args.logdir
else:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"/container/scr/vision_graph/ViG/log/{timestamp}"

# 学習
torch_seed()
net = ViG(1000, n_blocks=7).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.001
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
# scheduler = None
history = np.zeros((0, 5))
num_epochs = 30
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history, batch_size, scheduler, log_dir)
evaluate_history(history, log_dir)

# モデルの保存
model_save_path = f'{log_dir}/ViG.pth'  # 保存先のパスを指定
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(net.state_dict(), model_save_path)
