import json
import random
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from PIL import Image
import torchvision.transforms as transforms

# ----------------------------------------
# 1️⃣ Load Pretrained GIT Model
# ----------------------------------------
MODEL_NAME = "microsoft/git-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ FIX: Use CausalLM instead of Seq2SeqLM
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
model.to("cuda")

# ----------------------------------------
# 2️⃣ Define Reward Model (Placeholder)
# ----------------------------------------
class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, image, caption):
        reward = torch.tensor([len(caption.split()) / 10.0], device="cuda")
        return reward

reward_model = RewardModel().to("cuda")

# ----------------------------------------
# 3️⃣ Define PPO Trainer for RLHF
# ----------------------------------------
ppo_config = PPOConfig(  # ✅ FIXED: Removed `model_name`
    batch_size=16,
    learning_rate=1e-6,
    mini_batch_size=4,
    optimize_cuda_cache=True,
)

ppo_trainer = PPOTrainer(
    model, model, tokenizer, config=ppo_config
)

# ----------------------------------------
# 4️⃣ Dataset & DataLoader
# ----------------------------------------
class CaptioningDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_paths = list(self.data.keys())
        self.captions = list(self.data.values())
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

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CaptioningDataset('assets/processed_captions.json', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# ----------------------------------------
# 5️⃣ RLHF-Modified Training Loop with Visualization
# ----------------------------------------
num_epochs = 1000
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, captions in train_loader:
        images = images.to("cuda")

        # 🔹 Step 1: Generate captions
        input_ids = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        generated_ids = model.generate(images, max_length=50)

        # 🔹 Step 2: Decode captions
        generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 🔹 Step 3: Compute rewards
        rewards = torch.tensor([reward_model(images[i], generated_captions[i]) for i in range(len(images))], device="cuda")

        # 🔹 Step 4: PPO Optimization
        queries = input_ids
        responses = generated_ids
        stats = ppo_trainer.step(queries, responses, rewards)

        total_loss += stats["ppo/loss/total"]

    print(f"Epoch {epoch+1}/{num_epochs}, RLHF Loss: {total_loss:.4f}")

    # 🔹 Step 5: Save Model & Visualization Every 10 Epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"checkpoints/rlhf_git_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            sample_indices = random.sample(range(len(val_dataset)), 3)
            samples = []
            sample_images = []
            sample_gt = []

            for idx in sample_indices:
                img, gt_caption = val_dataset[idx]
                sample_images.append(img)
                sample_gt.append(torch.tensor(gt_caption))

            images_batch = torch.stack(sample_images).to("cuda")

            generated = model.generate(images_batch, max_length=50)
            for i in range(len(generated)):
                gen_indices = generated[i].tolist()
                gen_words = [dataset.captions[i] for idx in gen_indices if idx not in [0, 1, 2]]
                gt_indices = sample_gt[i].tolist()
                gt_words = [dataset.captions[i] for idx in gt_indices if idx not in [0, 1, 2]]
                samples.append({
                    'image': images_batch[i],
                    'gen_caption': ' '.join(gen_words),
                    'gt_caption': ' '.join(gt_words)
                })

            # 🔹 Step 6: Save Visualization
            fig, axes = plt.subplots(3, 1, figsize=(8, 24))
            for i, sample in enumerate(samples):
                img = sample['image'].cpu().numpy().transpose(1, 2, 0)  # Convert tensor to image
                axes[i].imshow(img)
                axes[i].set_title(f"Generated: {sample['gen_caption']}\nGround Truth: {sample['gt_caption']}", fontsize=10)
                axes[i].axis('off')
            plt.tight_layout()
            vis_path = f'visualizations/epoch_{epoch+1}.png'
            plt.savefig(vis_path)
            plt.close()
            print(f"Visualization saved to {vis_path}")



print("Training completed. Model is ready with RLHF fine-tuning.")
