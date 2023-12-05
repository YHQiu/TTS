import datetime
import json
import torch
import os
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from train_config import TrainConfig
from dataset import TextToSpeechDataset
from ytrainer import TextToSpeechTrainer
from ytts_model import YTTS

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

    # 初始化模型并放置到对应设备上
    model = YTTS(device=local_rank)

    print(f"加载完成模型{model}")

    # 使用 DistributedDataParallel 包装模型
    if args.nproc_per_node > 1:
        print(f"开始加载分布式模型 设备ID {local_rank}")
        # model = DDP(model, device_ids=[local_rank], output_device=[local_rank])
        model = DDP(model, device_ids=[local_rank])
        print(f"加载完成分布式模型{model} 设备ID {local_rank}")

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
    parser.add_argument('--world-size', type=int, default=1)  # 将world-size设置为1
    parser.add_argument('--nproc_per_node', type=int, help='必须指定--nproc_per_node 使用多少个GPU进行训练')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    dist.init_process_group(backend='nccl', world_size=world_size)  # 在这里指定world_size
    local_rank = torch.distributed.get_rank()
    print(f"当前GPU {local_rank} {world_size}")
    main(args, local_rank)
