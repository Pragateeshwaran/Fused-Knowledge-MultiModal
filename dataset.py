import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json

class FineTuningDataset(Dataset):
    def __init__(self, data, transform=None, vocab_size=4635):
        self.data = data
        self.transform = transform
        self.vocab = self._build_vocab()
        self.vocab_size = vocab_size
        self.max_length = 50  # Adjust based on your caption length

    def _build_vocab(self):
        # Build vocabulary from data (simplified example)
        word_freq = {}
        for item in self.data:
            caption = item.get('caption', '').lower().split()
            for word in caption:
                word_freq[word] = word_freq.get(word, 0) + 1
        vocab = ['<pad>', '<start>', '<end>', '<unk>', 'yes', 'no']
        vocab.extend(sorted(word_freq, key=word_freq.get, reverse=True)[:self.vocab_size - 6])
        return {word: idx for idx, word in enumerate(vocab)}

    def _tokenize_caption(self, caption):
        tokens = ['<start>'] + caption.lower().split() + ['<end>']
        tokenized = [self.vocab.get(word, self.vocab['<unk>']) for word in tokens]
        # Pad or truncate to max_length
        if len(tokenized) < self.max_length:
            tokenized.extend([self.vocab['<pad>']] * (self.max_length - len(tokenized)))
        else:
            tokenized = tokenized[:self.max_length]
        return torch.tensor(tokenized)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item.get('image_path', '')
        caption = item.get('caption', '')

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        caption_tensor = self._tokenize_caption(caption)

        return image, caption_tensor, len(caption.split())  # Return length for dynamic padding in model