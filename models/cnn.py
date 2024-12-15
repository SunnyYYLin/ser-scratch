import torch
import torch.nn as nn
import torchaudio as ta
from config import *
from collections import OrderedDict
    
class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int=2):
        super(ConvBlock1d, self).__init__()
        self.pool_size = pool_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        return x

class SpeechCNN(nn.Module):
    def __init__(self, config: CNNConfig):
        super(SpeechCNN, self).__init__()
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
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor=None):
        """_summary_

        Args:
            audio (torch.Tensor): (batch_size, seq_len)
            attention_mask (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        features = self.convs(audio.unsqueeze(1)) # (batch_size, num_filters[-1], seq_len)
        features = self.pool(features).squeeze(-1) # (batch_size, num_filters[-1], dim_feats)
        features = self.dropout(features)
        return features