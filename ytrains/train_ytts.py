import datetime
import json
import torch
import torch.optim as optim
from torch import nn
import os
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from train_config import TrainConfig
from dataset import TextToSpeechDataset
from ytts_model import YTTS
import torch.distributed as dist

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
        print("开始训练")
        for epoch in range(self.num_epochs):

            print(f"开始训练epoch={epoch}")

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

    print(f"加载完成配置文件{train_config} {datetime.datetime.now()}")

    # 设置分布式训练参数
    # 这些参数可以根据你的具体需求进行修改
    rank = args.local_rank  # 当前进程的排名
    world_size = torch.cuda.device_count()  # 获取可用的 GPU 数量
    backend = 'nccl'  # 使用的后端，可以是 gloo、nccl 等

    # 初始化进程组
    # print(f"正在开始初始化分布式训练进程组 {backend} {rank} {world_size} {datetime.datetime.now()}")
    # dist.init_process_group(
    #     backend=backend,
    #     init_method='tcp://localhost:54321',  # 这里的地址和端口需要根据实际情况进行设置
    #     rank=rank,
    #     world_size=world_size
    # )
    #
    # print(f"初始化完成分布式训练进程组tcp://localhost:54321：{backend} {rank} {world_size} {datetime.datetime.now()}")

    # 加载数据集
    dataset = TextToSpeechDataset(data_config['metadata'], data_config['wavs'])
    train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

    print(f"加载完成数据集,总计{len(dataset)}条数据待训练 {datetime.datetime.now()}")

    # 设置GPU或CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        print("use cuda")
    else:
        print("use cpu")

    # 初始化模型并放置到GPU上
    model = YTTS()  # 替换成你的模型
    model.to(device)

    # 初始化DistributedDataParallel，如果启用了分布式训练
    if torch.cuda.device_count() > 1:
        # 使用 DistributedDataParallel 包装模型
        model = DDP(model)
        print("使用分布式训练 {datetime.datetime}")

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
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--nproc_per_node', type=int, default=torch.cuda.device_count())
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    main(args.train_ytts_config)