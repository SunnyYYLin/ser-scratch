from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from data_process import AudioCollator
from config import *
from models.emotion_classifier import EmotionClassifier
from metrics import SERMetircs
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Load the dataset
print("Loading the dataset...")
dataset = load_dataset("Zahra99/IEMOCAP_Audio")
print(dataset)
dataset: Dataset = concatenate_datasets([dataset[f'session{i}'] for i in range(1, 6)])
dataset = dataset.train_test_split(test_size=0.1, seed=241218)
train_dataset, eval_dataset = dataset['train'], dataset['test']
print(train_dataset[2])
print(f"Train Size: {len(train_dataset)}\nEval Size: {len(eval_dataset)}")
print("Dataset loaded.")

# Load the model
config = Wav2VecConfig(
    pooling='attn',
)
model = EmotionClassifier(config)
print(model)

# Load the trainer
output_dir = Path('checkpoints') / config.abbrev
log_dir = Path('logs') / config.abbrev
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=128,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=4,
    logging_strategy='steps',
    eval_strategy='epoch',
    logging_dir=log_dir,
    logging_steps=20,
    save_strategy='epoch',
    save_steps=1,
    save_total_limit=2,
    load_best_model_at_end=True,
    use_cpu=False,
    fp16=True,
)
call_backs = [
    # EarlyStoppingCallback(early_stopping_patience=10)
    
]
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=AudioCollator(),
    callbacks=call_backs,
    compute_metrics=SERMetircs(config)
)
# trainer.train()