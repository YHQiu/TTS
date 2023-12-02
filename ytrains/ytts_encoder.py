import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_text):
        embedded = self.embedding(input_text)
        return embedded

import torchaudio.transforms as transforms

class AudioEncoder(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128):
        super(AudioEncoder, self).__init__()
        self.spectrogram = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)

    def forward(self, audio_signal):
        mel_spec = self.spectrogram(audio_signal)
        return mel_spec
