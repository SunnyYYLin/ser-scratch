from pathlib import Path
import numpy as np
import torch
import torchaudio as ta
from safetensors.torch import load_file
from models import EmotionClassifier
from config import *

STANDARD_SAMPLE_RATE = 16000
MODELS = ['CNN-large', 'CNN-small', 'Wav2Vec-maxpool', 'Wav2Vec-attentivepool', 'Wav2Vec-conv-maxpool']
ID2EMOTION_TXT = {
    0: 'Angry 🤬',
    1: 'Happy 😊',
    2: 'Neutral 😐',
    3: 'Sad 😟',
}

def load_model(model_name: str):
    print(f'Loading model: {model_name}')
    match model_name:
        case 'CNN-large':
            config = CNNConfig(
                kernel_sizes=[21, 19, 17, 15, 13, 11, 9],
                num_filters=[32, 64, 128, 256, 512, 1024, 1024],
            )
            model = EmotionClassifier(config)
        case 'CNN-small':
            config = CNNConfig(
                kernel_sizes=[19, 17, 15, 13],
                num_filters=[32, 64, 128, 256],
            )
            model = EmotionClassifier(config)
        case 'Wav2Vec-maxpool':
            config = Wav2VecConfig(
                pooling = 'max',
            )
            model = EmotionClassifier(config)
        case 'Wav2Vec-attentivepool':
            config = Wav2VecConfig(
                pooling = 'attentive',
            )
            model = EmotionClassifier(config)
        case 'Wav2Vec-conv-maxpool':
            config = Wav2VecConfig(
                pooling = 'conv',
            )
            model = EmotionClassifier(config)
        case _:
            return None
    
    save_dir = Path(__file__).parent.parent / 'checkpoints' / config.abbrev
    model_folder_list = save_dir.glob(f'checkpoint-*')
    best_iteration = max([int(folder.name.split('-')[-1]) for folder in model_folder_list])
    model_path = save_dir / f'checkpoint-{best_iteration}' / 'model.safetensors'
    model.load_state_dict(load_file(model_path))
    print(f'Model loaded.')
    return model

def predict(model: EmotionClassifier, audio: tuple[np.ndarray, int]):
    '''
    Args:
        audio (tuple[ndarray(seq_len, channels), int]): (waveform, sample_rate)
        model (EmotionClassifier): The model to use for prediction
    Returns:
        emo (str): The predicted emotion
    '''
    print('Predicting...')
    waveform = preprocess(audio)
    waveform = waveform.unsqueeze(0)
    logits: torch.Tensor = model(waveform)['logits']
    label = logits.argmax(dim=-1)
    print(f'Prediction: {ID2EMOTION_TXT[label.item()]}')
    return ID2EMOTION_TXT[label.item()]

def preprocess(audio: tuple[np.ndarray, int]):
    '''
    Args:
        audio (tuple[ndarray(seq_len, channels), int]): (waveform, sample_rate)
            or audio (str): The path to the audio file
    '''
    if isinstance(audio, str):
        waveform, sample_rate = ta.load(audio, channels_first=False)
    else:
        sample_rate, waveform = audio
    waveform = torch.tensor(waveform)
    waveform = waveform.to(torch.float32) / 2**15 # int16 to float32
    if sample_rate != STANDARD_SAMPLE_RATE:
        waveform = ta.transforms.Resample(sample_rate, STANDARD_SAMPLE_RATE)(waveform)
    if waveform.ndim != 1:
        waveform = waveform.mean(dim=-1)
    return waveform