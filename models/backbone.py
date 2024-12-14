from config import *
from .cnn import SpeechCNN
from .wav2vec import Wav2Vec2_CNN

def get_backbone(config: BackboneConfig):
    if isinstance(config, CNNConfig):
        return SpeechCNN(config)
    elif isinstance(config, Wav2VecConfig):
        return Wav2Vec2_CNN(config)
    else:
        raise ValueError("Invalid BackboneConfig")