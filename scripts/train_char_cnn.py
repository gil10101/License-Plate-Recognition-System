#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

from models.ocr.char_cnn import CharCNN


class CharFolderDataset(Dataset):
    def __init__(self, root: str, classes: List[str], img_size: int = 32, augment: bool = False):
        self.samples = []
        self.labels = []
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.img_size = img_size
        for cls in classes:
            d = Path(root) / cls
            if not d.exists():
                continue
            for p in d.glob('*.png'):
                self.samples.append(str(p))
                self.labels.append(self.class_to_idx[cls])
        base = [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
        ]
        aug = []
        if augment:
            aug = [
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        tail = [
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
        self.transform = transforms.Compose(base + aug + tail)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        x = self.transform(img)
        y = self.labels[idx]
        return x, y


def main():
    parser = argparse.ArgumentParser(description='Train CharCNN on char_dataset')
    parser.add_argument('--data', type=str, default='data/char_dataset')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', type=str, default='models/ocr_charcnn')
    args = parser.parse_args()

    classes = [str(c) for c in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'classes.json'), 'w') as f:
        json.dump(classes, f)

    dataset = CharFolderDataset(args.data, classes, augment=True)
    n = len(dataset)
    n_train = int(0.9 * n)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharCNN(num_classes=len(classes)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', patience=3, factor=0.5)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    hist = {'epoch': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), torch.tensor(y).to(device)
            optim.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        train_acc = correct / max(1, total)

        model.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), torch.tensor(y).to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                vcorrect += (pred == y).sum().item()
                vtotal += x.size(0)
        val_acc = vcorrect / max(1, vtotal)
        print(f"Epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        hist['epoch'].append(epoch)
        hist['train_acc'].append(train_acc)
        hist['val_acc'].append(val_acc)
        hist['lr'].append(optim.param_groups[0]['lr'])
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'state_dict': model.state_dict()}, os.path.join(args.out, 'charcnn_best.pt'))
        scheduler.step(val_acc)

    print(f"Best val acc: {best_acc:.3f}")

    # Save metrics
    import pandas as pd
    df = pd.DataFrame(hist)
    df.to_csv(os.path.join(args.out, 'charcnn_metrics.csv'), index=False)

    # Plots
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_acc'], label='train_acc')
    plt.plot(df['epoch'], df['val_acc'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('CharCNN Accuracy'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, 'charcnn_accuracy.png'), dpi=200)

    # Confusion matrix on validation set with best checkpoint
    try:
        state = torch.load(os.path.join(args.out, 'charcnn_best.pt'), map_location=device)
        model.load_state_dict(state.get('state_dict', state))
        model.eval()
        import numpy as np
        from sklearn.metrics import confusion_matrix
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1).cpu().numpy().tolist()
                y_pred.extend(pred)
                y_true.extend(y)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        plt.figure(figsize=(12,10))
        sns.heatmap(cm / cm.sum(axis=1, keepdims=True), cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('CharCNN Confusion (normalized)')
        plt.tight_layout(); plt.savefig(os.path.join(args.out, 'charcnn_confusion.png'), dpi=200)
    except Exception as e:
        print('Confusion matrix generation failed:', e)


if __name__ == '__main__':
    main()


