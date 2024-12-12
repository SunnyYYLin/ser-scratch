import torch
import torch.nn as nn
import torchaudio as ta
from config import *
from collections import OrderedDict

def get_backbone(config: BackboneConfig):
    return SpeechCNN(config)

class ResBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ResBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        return x + res
    
class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        return x

class SpeechCNN(nn.Module):
    def __init__(self, config: CNNConfig):
        super(SpeechCNN, self).__init__()
        self.spec = ta.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
        )
        conv_blocks = OrderedDict(
            [
                (
                    f'conv_{i}', 
                    ConvBlock1d(
                        1 if i==0 else config.num_filters[i-1], 
                        config.num_filters[i], 
                        config.kernel_sizes[i]
                    )
                )
                for i in range(len(config.kernel_sizes))
            ]
        )
        self.convs = nn.Sequential(conv_blocks)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor=None):
        """_summary_

        Args:
            audio (torch.Tensor): (batch_size, seq_len)
            attention_mask (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        # features = self.spec(audio).squeeze(1) # (batch_size, n_mel, seq_len)
        features = audio.unsqueeze(1)
        features = self.convs(features) # (batch_size, num_filters[-1], seq_len)
        features = self.pool(features).squeeze(-1) # (batch_size, num_filters[-1], dim_feats)
        features = self.dropout(features)
        return features