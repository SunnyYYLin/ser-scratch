from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from data_process import AudioCollator
from config import *
from models.emotion_classifier import EmotionClassifier
from metrics import SERMetircs

# Load the dataset
print("Loading the dataset...")
dataset = load_dataset("Zahra99/IEMOCAP_Audio")
dataset: Dataset = concatenate_datasets([dataset[f'session{i}'] for i in range(1, 6)])
dataset = dataset.train_test_split(test_size=0.1, seed=241218)
train_dataset, val_dataset = dataset['train'], dataset['test']
print("Dataset loaded.")

# Load the model
config = Wav2VecConfig()
model = EmotionClassifier(config)

# Load the trainer
args = TrainingArguments(
    output_dir='checkpoints',
    num_train_epochs=64,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_strategy='steps',
    eval_strategy='epoch',
    logging_dir='logs',
    logging_steps=20,
    save_strategy='epoch',
    save_steps=1,
    save_total_limit=2,
    load_best_model_at_end=True,
    use_cpu=False,
    fp16=True
)
call_backs = [
    # EarlyStoppingCallback(early_stopping_patience=10)
]
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=AudioCollator(),
    callbacks=call_backs,
    compute_metrics=SERMetircs(config)
)
trainer.train()