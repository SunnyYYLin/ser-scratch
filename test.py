from models.wav2vec import Wav2Vec2_CNN
import torch
from config import *

model = Wav2Vec2_CNN(Wav2VecConfig())
audio = torch.randn(2, 32_000)
out = model(audio)
print(out.shape)