from dataclasses import dataclass, field

@dataclass
class CNNConfig:
    sample_rate: int = 16_000
    
    kernel_sizes: list[int] = field(default_factory=lambda: [21, 19, 17, 15, 13, 11, 9])
    num_filters: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024, 1024])
    
    num_classes: int = 4
    dropout: float = 0.5
    
    @property
    def abbrev(self):
        return f"CNN-{self.kernel_sizes}-{self.num_filters}"
    
@dataclass
class Wav2VecConfig:
    sample_rate: int = 16_000
    num_classes: int = 4
    hidden_size: int = 768
    dim_feats: int = 256
    pooling: str = 'attn'
    
    kernel_size: int = 9
    dropout: float = 0.5
    
    @property
    def abbrev(self):
        return f"Wav2Vec-{self.pooling}"
    
BackboneConfig = CNNConfig|Wav2VecConfig