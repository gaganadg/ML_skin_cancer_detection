import argparse
import csv
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


HAM_MALIGNANT = { 'mel', 'bcc', 'akiec' }
HAM_BENIGN = { 'nv', 'bkl', 'df', 'vasc' }


def map_ham_label(dx: str) -> int:
    dx = (dx or '').strip().lower()
    if dx in HAM_MALIGNANT:
        return 1
    if dx in HAM_BENIGN:
        return 0
    raise ValueError(f"Unknown HAM10000 label: {dx}")


def load_ham_metadata(csv_path: Path) -> List[Tuple[str, int]]:
    rows = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            image_id = r.get('image_id') or r.get('imageid') or r.get('image')
            dx = r.get('dx') or r.get('diagnosis') or r.get('label')
            if not image_id or not dx:
                continue
            label = map_ham_label(dx)
            rows.append((image_id, label))
    return rows


def stratified_split(pairs: List[Tuple[str, int]], val_ratio: float, seed: int) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    random.seed(seed)
    by_class = { 0: [], 1: [] }
    for pid, y in pairs:
        by_class[y].append((pid, y))
    train, val = [], []
    for y, items in by_class.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


class HamDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,int]], images_root: Path, image_size: int = 224, transform=None):
        self.samples = samples
        self.images_root = images_root
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, label = self.samples[idx]
        # Try common extensions
        path = None
        for ext in ('.jpg', '.jpeg', '.png'): 
            p = self.images_root / f"{image_id}{ext}"
            if p.exists():
                path = p
                break
        if path is None:
            raise FileNotFoundError(f"Image file not found for {image_id} under {self.images_root}")
        img = Image.open(path).convert('RGB')
        if self.transform is None:
            tfm = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            tfm = self.transform
        x = tfm(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


def train(args):
    csv_path = Path(args.metadata_csv)
    images_root = Path(args.images_dir)
    pairs = load_ham_metadata(csv_path)
    # keep only those that exist on disk
    pairs = [(pid, y) for (pid, y) in pairs if any((images_root / f"{pid}{ext}").exists() for ext in ('.jpg', '.jpeg', '.png'))]
    if not pairs:
        raise RuntimeError(f"No images found next to metadata under {images_root}")

    train_pairs, val_pairs = stratified_split(pairs, val_ratio=args.val_split, seed=args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = HamDataset(train_pairs, images_root=images_root, image_size=args.image_size, transform=train_transform)
    val_ds = HamDataset(val_pairs, images_root=images_root, image_size=args.image_size, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN(num_classes=2).to(device)
    # Class weights (address imbalance)
    counts = {0: 0, 1: 0}
    for _, y in train_pairs:
        counts[y] += 1
    total = max(1, counts[0] + counts[1])
    # inverse frequency
    w0 = total / max(1, counts[0])
    w1 = total / max(1, counts[1])
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    best_val_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patience = max(2, args.patience)
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(1, total)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        scheduler.step(val_acc)

        if val_acc >= best_val_acc - 1e-6:
            best_val_acc = val_acc
            torch.save({'state_dict': model.state_dict(), 'val_acc': best_val_acc}, out_path)
            print(f"Saved best model to {out_path} (val_acc={best_val_acc:.3f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping due to no improvement.")
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata-csv', type=str, required=True)
    ap.add_argument('--images-dir', type=str, required=True)
    ap.add_argument('--out', type=str, default='ml_service/model_weights.pth')
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--patience', type=int, default=3)
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()





