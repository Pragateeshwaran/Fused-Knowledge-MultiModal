import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from datasets import load_dataset
from ModelFlow import MultimodalFusionModel
from dataset import FineTuningDataset

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size = 4635  # Match checkpoint vocabulary size
embed_dim = 512
hidden_dim = 1024
num_layers = 2
dropout = 0.1
batch_size = 32
num_epochs = 10
learning_rate = 0.0001

# Data transforms (placeholder for images, if used)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load boolq dataset
dataset = load_dataset("google/boolq", split="train")
boolq_data = [{"question": item["question"], "answer": str(item["answer"])} for item in dataset]

# Custom dataset for boolq (text-only, with optional dummy images)
class BoolQDataset(FineTuningDataset):
    def __init__(self, data, transform=None, vocab_size=4635):
        super().__init__(data, transform, vocab_size)
        self.data = [{"image_path": "dummy.jpg", "caption": f"{item['question']} {item['answer']}"}
                     for item in data]  # Combine question and answer as caption

    def __getitem__(self, idx):
        item = self.data[idx]
        # Load dummy image (or skip if text-only)
        image = Image.new('RGB', (224, 224), color='black') if self.transform else None
        if self.transform:
            image = self.transform(image)
        caption_tensor = self._tokenize_caption(item["caption"])
        return image, caption_tensor, len(item["caption"].split())

# Initialize dataset and loader
boolq_dataset = BoolQDataset(boolq_data, transform=transform, vocab_size=vocab_size)
train_loader = DataLoader(boolq_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = MultimodalFusionModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                             num_layers=num_layers, dropout=dropout).to(device)

# Load checkpoint with strict=True
checkpoint_path = r'checkpoints\model_epoch_2_100.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    # Freeze image encoder parameters
    for param in model.image_encoder.parameters():
        param.requires_grad = False
else:
    print("No checkpoint found. Training from scratch.")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is <pad> token
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Training loop
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (images, captions, lengths) in enumerate(train_loader):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(images, captions, lengths)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss / len(train_loader)

# Train
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch + 1}')
    loss = train(model, train_loader, criterion, optimizer, epoch + 1)
    print(f'Epoch {epoch + 1} Loss: {loss:.4f}')
    torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch + 1}.pth')

print("Training completed.")