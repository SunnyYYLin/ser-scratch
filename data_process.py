from transformers.data.data_collator import DataCollatorMixin
from torch.nn.utils.rnn import pad_sequence
import torch

SAMPLING_RATE = 16_000

class AudioCollator(DataCollatorMixin):
    def __call__(self, batch: list[dict[str, ]]):
        audios: list[torch.Tensor] = [torch.tensor(x['audio']['array'], dtype=torch.float) 
                                     for x in batch]
        attn_masks = torch.tensor([audio.shape[0] for audio in audios], dtype=torch.long)
        audios = pad_sequence(audios, batch_first=True, padding_value=0)
        labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)
        
        return {
            'audio': audios,
            'label': labels,
            'attention_mask': attn_masks
        }