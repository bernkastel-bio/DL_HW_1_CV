import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from wavemix.classification import WaveMix

# Configuration
NUM_EPOCHS = 80
BATCH_SIZE = 64
ACCUM_STEPS = 2
NUM_CLASSES = 10
IMG_SIZE = 256
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4


class GalaxyDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

class GalaxyTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        galaxy_dataset = load_dataset("matthieulel/galaxy10_decals")
        train_val_split = galaxy_dataset['train'].train_test_split(test_size=0.2, seed=42)
        
        # Changed transforms
        self.train_transform = transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        self.trainset = GalaxyDataset(train_val_split['train'], self.train_transform)
        self.valset = GalaxyDataset(train_val_split['test'], self.val_transform)
        self.testset = GalaxyDataset(galaxy_dataset['test'], self.val_transform)

        # Class weights
        labels = np.array(train_val_split['train']['label'])
        class_counts = np.bincount(labels)
        self.class_weights = torch.tensor(
            len(labels) / (len(class_counts) * (class_counts + 1e-6)),
            dtype=torch.float32
        ).to(self.device)

        self.model = WaveMix(
            num_classes=NUM_CLASSES,
            depth=16,
            mult=2,
            ff_channel=192,
            final_dim=192,
            dropout=0.3,  # Dropout lower
            level=3,
            initial_conv='pachify',
            patch_size=4
        ).to(self.device).to(torch.float32)

        # Data loaders
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.valset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.testset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Training setup
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=BASE_LR,
            weight_decay=WEIGHT_DECAY
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=BASE_LR,
            steps_per_epoch=len(self.train_loader),
            epochs=NUM_EPOCHS
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device)

            with autocast(device_type='cuda', enabled=False):
                outputs = self.model(data)
                loss = F.cross_entropy(outputs, targets, weight=self.class_weights) / ACCUM_STEPS

            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        total_correct = 0
        
        with torch.no_grad():
            for data, targets in loader:
                data = data.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device)
                
                outputs = self.model(data)
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()

        return total_correct / len(loader.dataset)

if __name__ == "__main__":
    trainer = GalaxyTrainer()
    best_val_acc = 0.0

    print(f"Training Galaxy10 on {trainer.device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Initial LR: {BASE_LR:.2e}")

    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Training Progress", unit='epoch')
    for epoch in epoch_bar:
        train_loss = trainer.train_epoch(epoch)
        val_acc = trainer.evaluate(trainer.val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(trainer.model.state_dict(), "wavemix_try3.pth")
            print(f"New best model saved: {val_acc:.2%}")
            
            # Test evaluation only on best model
            test_acc = trainer.evaluate(trainer.test_loader)
            print(f"Test Acc: {test_acc:.2%}")

    print("Training complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2%}")