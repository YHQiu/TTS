import json
import torch
import torch.optim as optim
from torch import nn
import os
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from ytrains.train_config import TrainConfig
from ytrains.dataset import TextToSpeechDataset
from ytrains.ytts_model import YTTS

class TextToSpeechTrainer:
    def __init__(self, model: YTTS, train_loader, val_loader, train_config: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = train_config.learning_rate
        self.num_epochs = train_config.num_epochs

        # 定义优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
        self.criterion = nn.MSELoss()  # 可根据任务选择合适的损失函数

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0

            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                # 输出为多个结果的张量数组
                outputs = self.model(inputs)
                # outputs和target都为结果的张量数组
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

def main(train_ytts_config):
    # 从命令行传入 train_ytts_config 训练配置文件
    with open(train_ytts_config, 'r') as f:
        config_data = json.load(f)

    # 从JSON文件加载数据集配置和训练配置
    data_config = config_data['data_config']
    train_config : TrainConfig = TrainConfig(**config_data['train_config'])

    # 加载数据集
    dataset = TextToSpeechDataset(data_config['metadata'], data_config['wavs'])
    train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

    # 设置GPU或CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型并放置到GPU上
    model = YTTS()  # 替换成你的模型
    model.to(device)

    # 初始化DistributedDataParallel，如果启用了分布式训练
    if torch.cuda.device_count() > 1:
        model = DDP(model)

    # 创建训练器
    trainer = TextToSpeechTrainer(model, train_loader, None, train_config)

    # 执行训练
    trainer.train()

    # 保存模型
    model_path = os.path.join(train_config.outputs, 'model.pth')
    trainer.save_model(model_path)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Train Text-to-Speech Model')
    parser.add_argument('--train_ytts_config', type=str, default="train_ytts_config.json", help='Path to train YTTS configuration JSON file')
    args = parser.parse_args()

    main(args.train_ytts_config)