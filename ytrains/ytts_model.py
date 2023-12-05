import torch
import torch.nn as nn
from transformers import BertTokenizer

# 初始化一个预训练的 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.max_len = max_len
        self.d_model = d_model
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', self.encoding)  # 移至最后，避免重复注册

    def forward(self, x):
        seq_len = x.size(0)

        if seq_len >= self.max_len:
            # If the sequence length is greater than or equal to max_len, slice encoding to match the input sequence
            encoding = self.encoding[:seq_len, :]
        else:
            # Calculate the amount of padding needed
            pad_len = self.max_len - seq_len
            # Pad x with zeros to match the sequence length of encoding
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))  # Pad along the first dimension (time steps)

            # Update the sequence length after padding
            seq_len = x.size(0)

            # Truncate encoding to match the sequence length of x
            encoding = self.encoding[:seq_len, :]

        return x + encoding.detach().to(self.device)

class YTTS(nn.Module):
    def __init__(self, device, vocab_size=30522, d_model=768, num_layers=6, num_heads=8, d_ff=2048, max_len=512, mel_max_len=512):
        """
        Args:

            device: 模型训练设备

            vocab_size: 这是用于模型的输入的词汇表大小。它表示模型能够处理的不同token的数量。在文本到语音的任务中，这个参数决定了模型能够处理的文本的多样性程度。

            d_model: 这个参数表示Transformer模型的embedding维度或者隐藏单元的数量。在模型中，它决定了token的表示维度。较大的d_model能够捕捉更多的复杂特征，但也会增加模型的计算成本。

            num_layers: 这表示Transformer编码器中的编码器层数。它影响了模型的深度，层数越多，模型能够学习的特征层次也就越多。更深的模型通常可以提取更复杂的特征，但也可能增加过拟合的风险。

            num_heads: 这个参数决定了自注意力机制中头的数量。多头自注意力允许模型在不同的表示空间下学习并关注不同的token之间的关系。更多的头意味着模型可以并行关注更多的信息。

            d_ff: 这是Transformer中全连接层的隐藏单元数量或称为前馈神经网络的维度。它决定了Transformer中前馈神经网络的宽度，即每个位置的表示在全连接层中的映射维度。

            max_len: 这是位置编码器中序列最大长度。位置编码用于向模型提供序列中token的位置信息。设置一个合适的max_len能够确保模型在处理长序列时不会丢失位置信息。

            mel_output_size: 这是用于生成mel频谱的线性层的输出大小。它决定了模型生成的mel频谱的特征维度。在语音合成任务中，这个值影响着生成的语音质量和细节。
        """
        super(YTTS, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mum_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.mel_max_len = mel_max_len

        self.embedding = nn.Embedding(vocab_size, d_model).to(device)
        self.positional_encoding = PositionalEncoding(d_model, device=device, max_len=max_len).to(device)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff).to(device)
            for _ in range(num_layers)
        ])
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff),
            num_layers=num_layers
        ).to(device)

        self.mel_generation = nn.Linear(d_model, mel_max_len).to(device)
        self.hidden_layer = nn.Linear(d_model, d_model).to(device)

        self.to(device)

    def forward(self, input_sequence):

        texts = input_sequence

        # 初始化模型输出为全零张量，使用实际生成的最大长度
        mel_outputs = []

        for text in texts:
            tokens = tokenizer.tokenize(text)
            text_indices = tokenizer.convert_tokens_to_ids(tokens)
            text_indices = torch.tensor(text_indices).to(self.device)
            embedded_text = self.embedding(text_indices)
            embedded_text = embedded_text * torch.sqrt(
                torch.tensor(self.d_model, dtype=torch.float, device=self.device))
            embedded_text = self.positional_encoding(embedded_text)

            transformer_output = embedded_text
            for layer in self.transformer_layers:
                transformer_output = layer(transformer_output)

            encoded = self.transformer_encoder(transformer_output)
            hidden_output = self.hidden_layer(encoded)

            # 频谱输出
            mel_output = self.mel_generation(hidden_output)

            # 调整输出形状以匹配预期形状 (通道数, 频谱特征维度, 时间步数)
            mel_output = mel_output.permute(0, 2, 1).contiguous()  # 调整维度顺序

            mel_outputs.append(mel_output)

        # 将 mel_outputs 合并成一个张量
        mel_outputs = torch.stack(mel_outputs, dim=0)

        return mel_outputs

    def criterion(self, outputs, targets):
        loss_function = nn.MSELoss()
        loss = loss_function(outputs, targets)
        return loss