import torch
import torch.nn as nn
from .backbone import get_backbone

class EmotionClassifier(nn.Module):
    def __init__(self, config):
        super(EmotionClassifier, self).__init__()
        self.backbone = get_backbone(config)
        self.classifier = nn.LazyLinear(config.num_classes)
        self._lazy_init()
        
    def _lazy_init(self):
        dummy_input = torch.randn(1, 16000)
        self.forward(dummy_input)
        
    def forward(self, 
                audio: torch.Tensor, 
                attention_mask: torch.Tensor=None,
                label: torch.Tensor=None):
        """_summary_

        Args:
            audio (torch.Tensor): (batch_size, seq_len)
            attention_mask (torch.Tensor, optional): 
            label (torch.Tensor, optional): 

        Returns:
            _type_: _description_
        """
        features = self.backbone(audio, attention_mask) # (batch_size, hidden_size)
        logits = self.classifier(features) # (batch_size, num_classes)
        results = {'logits': logits}
        if label is not None:
            loss = nn.CrossEntropyLoss()(logits, label)
            results['loss'] = loss
        return results