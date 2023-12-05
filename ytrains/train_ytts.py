import datetime
import json
import torch
import os
from torch.utils.data import DataLoader, SequentialSampler
from train_config import TrainConfig
from dataset import TextToSpeechDataset
from ytrainer import TextToSpeechTrainer
from ytts_model import YTTS

def main(args):
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
    train_sampler = SequentialSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=train_config.batch_size, sampler=train_sampler)

    print(f"加载完成数据集,总计{len(dataset)}条数据待训练 {datetime.datetime.now()}")

    # 初始化模型并放置到对应设备上
    model = YTTS(device=device)

    print(f"加载完成模型{model}")

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
    args = parser.parse_args()
    main(args)
