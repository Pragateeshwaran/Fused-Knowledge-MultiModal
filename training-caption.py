import json
from collections import Counter
import os
import random
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import torchvision.transforms as transforms

from ModelFlow import MultimodalFusionModel

#############################################
# Dataset for Image Captioning
#############################################
class CaptioningDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        all_captions = [caption.lower().split() for caption in self.data.values()]
        word_freq = Counter(word for caption in all_captions for word in caption)
        # Reserve special tokens: <pad>, <start>, <end>, <unk>
        self.vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.vocab.update({word: idx for idx, (word, _) in enumerate(word_freq.most_common(29996), start=4)})
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        self.image_paths = list(self.data.keys())
        self.captions = []
        for caption in self.data.values():
            words = caption.lower().split()
            indices = [self.vocab.get(word, self.vocab['<unk>']) for word in words]
            self.captions.append([self.vocab['<start>']] + indices + [self.vocab['<end>']])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption = self.captions[idx]
        return image, caption

#############################################
# Collate function to batch data
#############################################
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    max_len = max(len(c) for c in captions)
    padded_captions = [c + [0] * (max_len - len(c)) for c in captions]
    return images, torch.tensor(padded_captions)

#############################################
# Utility: Denormalize image for visualization
#############################################
def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image

#############################################
# Training Loop with Visualization of Random Samples
#############################################
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transformation for images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and compute vocabulary size
    dataset = CaptioningDataset('assets/processed_captions.json', transform=transform)
    vocab_size = len(dataset.vocab)

    # Instantiate the multimodal model.
    # Note: MultimodalFusionModel requires three inputs: image, tgt, and text_input.
    model = MultimodalFusionModel(vocab_size=vocab_size,
                                  embed_dim=512,
                                  num_layers_enc=2,  # Text encoder layers
                                  num_layers_dec=4,  # Decoder layers
                                  num_heads=8,
                                  ff_dim=2048,
                                  max_seq_len=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # Split dataset into training and validation sets.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=4,
                            pin_memory=True)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    num_epochs = 1000
    prompt_length = 10  # Fixed text prompt length for the text encoder

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, captions in train_loader:
            images, captions = images.to(device), captions.to(device)
            # Teacher forcing: use caption tokens except the last one as input (tgt)
            tgt = captions[:, :-1]
            target = captions[:, 1:]

            # Create a fixed text prompt filled with <start> token (id=1)
            batch_size = images.size(0)
            text_prompt = torch.ones(batch_size, prompt_length, device=device, dtype=torch.long) * 1

            optimizer.zero_grad()
            with autocast():
                # Forward pass (three inputs: image, tgt, and text_input)
                logits = model(image=images, tgt=tgt, text_input=text_prompt)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        # Save checkpoint and visualize predictions every 10 epochs or at the final epoch
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = f'checkpoints/model_epoch_2_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model saved to {checkpoint_path}')

            model.eval()
            with torch.no_grad():
                # Randomly sample 3 distinct indices from the validation dataset
                sample_indices = random.sample(range(len(val_dataset)), 3)
                samples = []
                sample_images = []
                sample_gt = []
                for idx in sample_indices:
                    img, gt_caption = val_dataset[idx]
                    sample_images.append(img)
                    sample_gt.append(torch.tensor(gt_caption))
                # Stack images and move to device
                images_batch = torch.stack(sample_images).to(device)
                # Create fixed prompt for each sample
                batch_size = images_batch.size(0)
                text_prompt = torch.ones(batch_size, prompt_length, device=device, dtype=torch.long) * 1

                # Generate captions auto-regressively
                generated = model.generate(images_batch, text_input=text_prompt, max_len=50)

                # Format the sample outputs
                for i in range(len(generated)):
                    gen_indices = generated[i].tolist()
                    gen_words = [dataset.idx_to_word.get(idx, '<unk>') for idx in gen_indices if idx not in [0, 1, 2]]
                    gt_indices = sample_gt[i].tolist()
                    gt_words = [dataset.idx_to_word.get(idx, '<unk>') for idx in gt_indices if idx not in [0, 1, 2]]
                    samples.append({
                        'image': images_batch[i],
                        'gen_caption': ' '.join(gen_words),
                        'gt_caption': ' '.join(gt_words)
                    })

                # Plot and save the visualization
                fig, axes = plt.subplots(3, 1, figsize=(8, 24))
                for i, sample in enumerate(samples):
                    img = denormalize(sample['image'])
                    axes[i].imshow(img)
                    axes[i].set_title(
                        f"Generated: {sample['gen_caption']}\nGround Truth: {sample['gt_caption']}",
                        fontsize=10)
                    axes[i].axis('off')
                plt.tight_layout()
                vis_path = f'visualizations/epoch_{epoch+1}.png'
                plt.savefig(vis_path)
                plt.close()
                print(f"Visualization saved to {vis_path}")


    print("Training completed. Model is ready for finetuning on logical reasoning.")
