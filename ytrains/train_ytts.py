import datetime
import json
import torch
import torch.optim as optim
from torch import nn
import os
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from train_config import TrainConfig
from dataset import TextToSpeechDataset
from ytts_model import YTTS

class TextToSpeechTrainer:
    def __init__(self, model, train_loader, val_loader, train_config: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = train_config.learning_rate
        self.num_epochs = train_config.num_epochs

        # 定义优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
        # self.criterion = nn.MSELoss()  # 可根据任务选择合适的损失函数
        print(f"初始化完成 TextToSpeechTrainer model {self.model} train_loader {self.train_loader} val_loader {self.val_loader} learning_rate {self.learning_rate} num_epochs {self.num_epochs}")

    def train(self):
        print("开始训练")
        for epoch in range(self.num_epochs):
            print(f"开始训练 epoch={epoch}")
            self.model.train()
            total_loss = 0.0

            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.model.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / len(self.train_loader)}")

    def evaluate(self):
        if self.val_loader is None:
            print("No validation set provided.")
            return

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        print(f"Validation Loss: {total_loss / len(self.val_loader)}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

def main(args, local_rank):
    # 设置GPU或CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本 {torch.version.cuda}")
    print(f"use {device}")
    train_ytts_config = args.train_ytts_config

    # 从命令行传入 train_ytts_config 训练配置文件
    with open(train_ytts_config, 'r') as f:
        config_data = json.load(f)

    # 从JSON文件加载数据集配置和训练配置
    data_config = config_data['data_config']
    train_config = TrainConfig(**config_data['train_config'])

    print(f"加载完成配置文件 {train_config} {datetime.datetime.now()}")

    # 初始化进程组
    print(f"nccl version {torch.cuda.nccl.version()}")

    # 加载数据集
    dataset = TextToSpeechDataset(data_config['metadata'], data_config['wavs'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=train_config.batch_size, sampler=train_sampler)

    print(f"加载完成数据集,总计{len(dataset)}条数据待训练 {datetime.datetime.now()}")

    # 初始化模型并放置到GPU上
    model = YTTS()  # 替换成你的模型
    model.to(device)

    print(f"加载完成模型{model}")

    # 使用 DistributedDataParallel 包装模型

    print(f"开始加载分布式模型{local_rank}")

    model = DDP(model, device_ids=[local_rank])

    print(f"加载完成分布式模型{model}  {local_rank}")

    # 创建训练器
    trainer = TextToSpeechTrainer(model, train_loader, None, train_config)
    print(f"创建完成训练器 {trainer}")

    # 执行训练
    print(f"开始训练 {datetime.datetime.now()}")
    trainer.train()
    print(f"结束训练 {datetime.datetime.now()}")

    # 保存模型
    model_path = os.path.join(train_config.outputs, 'model.pth')
    trainer.save_model(model_path)
    print(f"保存模型 {model_path} {datetime.datetime.now()}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Text-to-Speech Model')
    parser.add_argument('--train_ytts_config', type=str, default="train_ytts_config.json", help='Path to train YTTS configuration JSON file')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    print(f"当前GPU {local_rank}")
    main(args, local_rank)
