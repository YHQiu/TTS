import os
from typing import Dict, Callable, Tuple
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch

from ytrains.train_config import TrainConfig


class TextToSpeechDataset(Dataset):
    def __init__(self, metadata_path, wavs_dir):
        self.metadata = pd.read_csv(metadata_path, sep='|', header=None, names=['ID', 'Text', 'Text2'])
        self.wavs_dir = wavs_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        wav_file = os.path.join(self.wavs_dir, self.metadata.iloc[idx, 0] + '.wav')

        waveform, sample_rate = torchaudio.load(wav_file)

        # 使用torchaudio库的函数来提取Mel频谱特征
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)

        # 将Mel频谱特征转为对数刻度
        log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

        # 读取文本并进行编码（示例中使用简单的单词索引作为编码）
        text = self.metadata.iloc[idx, 1]

        # 计算最大长度 max_len
        hop_length = 512  # 假设每个帧的时间间隔是 512（需要根据你的实际数据进行设定）

        max_audio_length = 11.6  # 最大音频长度限制为 11.6 秒
        max_len = int((max_audio_length * sample_rate) / hop_length)

        # 如果特征长度小于最大长度，则进行填充
        if log_mel_spectrogram.shape[2] < max_len:
            padded_spectrogram = F.pad(log_mel_spectrogram, (0, max_len - log_mel_spectrogram.shape[2]))
        # 如果特征长度大于最大长度，则进行截断
        elif log_mel_spectrogram.shape[2] > max_len:
            padded_spectrogram = log_mel_spectrogram[:, :, :max_len]
        # 如果特征长度等于最大长度，则保持不变
        else:
            padded_spectrogram = log_mel_spectrogram

        return text, padded_spectrogram

def load_tts_samples(
        datasets: Dict,
        eval_split=True,
        formatter: Callable = None,
        eval_split_max_size=None,
        eval_split_size=0.01,
) -> Tuple[DataLoader, DataLoader]:
    dataset = TextToSpeechDataset(datasets['metadata'], datasets['wavs'])

    if eval_split:
        train_size = int(len(dataset) * (1 - eval_split_size))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=TrainConfig().batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainConfig().batch_size, shuffle=False) if val_dataset else None

    return train_loader, val_loader