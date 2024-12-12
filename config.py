from dataclasses import dataclass, field

@dataclass
class CNNConfig:
    n_mels: int = 128
    sample_rate: int = 16_000
    
    kernel_sizes: list[int] = field(default_factory=lambda: [19, 17, 15, 13, 11, 9])
    num_filters: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024])
    dim_feats: int = 128
    num_classes: int = 4
    
@dataclass
class RNNConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    
BackboneConfig = CNNConfig|RNNConfig