import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
from config import *

class SERMetircs:
    def __init__(self, config: BackboneConfig):
        self.accuracy_metric = Accuracy(task="multiclass", 
            num_classes=config.num_classes)
        self.precision_metric = Precision(task="multiclass", 
            num_classes=config.num_classes)
        self.recall_metric = Recall(task="multiclass",
            num_classes=config.num_classes)
        self.f1_metric = F1Score(task="multiclass", 
            num_classes=config.num_classes)
    
    def __call__(self, pred):
        labels = torch.tensor(pred.label_ids)
        preds = torch.tensor(pred.predictions).argmax(dim=-1)

        accuracy = self.accuracy_metric(preds, labels)
        precision = self.precision_metric(preds, labels)
        recall = self.recall_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)
        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
        }
        