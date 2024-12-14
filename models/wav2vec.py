import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from .cnn import ConvBlock1d
from config import Wav2VecConfig

def get_pool(config: Wav2VecConfig):
    match config.pooling:
        case 'mean':
            return nn.AdaptiveAvgPool1d(1)
        case 'max':
            return nn.AdaptiveMaxPool1d(1)
        case 'attn':
            return AttentionPool(config)
        case 'conv':
            return CNNPool(config)
        case _:
            raise ValueError("Invalid pooling method")

class CNNPool(nn.Module):
    def __init__(self, config: Wav2VecConfig):
        super(CNNPool, self).__init__()
        self.conv = ConvBlock1d(
            in_channels=config.hidden_size,
            out_channels=config.dim_feats,
            kernel_size=config.kernel_size,
            pool_size=1
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, features: torch.Tensor):
        features = self.conv(features) # (batch_size, dim_feats, seq_len)
        features = self.pool(features).squeeze(-1) # (batch_size, dim_feats)
        return features
    
class AttentionPool(nn.Module):
    def __init__(self, config):
        super(AttentionPool, self).__init__()
        self.attention = nn.Linear(config.hidden_size, 1)  # 用于生成注意力权重

    def forward(self, features: torch.Tensor):
        """
        features: (batch_size, hidden_size, seq_len)
        """
        features = features.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        attention_scores = self.attention(features)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        pooled_features = torch.sum(features * attention_weights, dim=1)
        return pooled_features

class Wav2Vec2_CNN(nn.Module):
    def __init__(self, config: Wav2VecConfig):
        super(Wav2Vec2_CNN, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        self.pool = get_pool(config)
    
    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor=None):
        features = self.wav2vec(audio, attention_mask=attention_mask).last_hidden_state # (batch_size, seq_len, hidden_size)
        features = features.permute(0, 2, 1) # (batch_size, hidden_size, seq_len)
        features = self.pool(features).squeeze(-1) # (batch_size, dim_feats)
        return features
    
if __name__ == '__main__':
    model = Wav2Vec2_CNN(Wav2VecConfig())
    audio = torch.randn(2, 32_000)
    out = model(audio)
    print(out.shape)