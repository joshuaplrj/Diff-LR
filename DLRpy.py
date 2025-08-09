# Below is a minimal, self-contained PyTorch implementation of the two-phase “Dynamic Learning Rate” algorithm (Thanks to kimi k2 for writing this)
"""
Dynamic Learning Rate (DLR) – two-phase implementation
Phase 1: binary-search to find a good initial LR
Phase 2: monitor train/val loss and adapt LR on the fly
"""

import math, time, copy, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple

# ------------------------------------------------------------------
# 1.  A tiny CNN for CIFAR-10
# ------------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ------------------------------------------------------------------
# 2.  Utilities
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loaders(root: str = "./data", batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010))
    ])
    train_set = datasets.CIFAR10(root, train=True,  download=True, transform=transform)
    val_set   = datasets.CIFAR10(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def loss_on_loader(model, loader, criterion):
    model.eval()
    total, loss_sum = 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            total += x.size(0)
    return loss_sum / total

# ------------------------------------------------------------------
# 3.  Phase-1: binary search for initial LR
# ------------------------------------------------------------------
def binary_search_lr(
        model_fn,
        train_loader,
        val_loader,
        low: float = 1e-6,
        high: float = 1e-1,
        max_iters: int = 10,
        epochs_per_trial: int = 3) -> float:
    """
    Returns a reasonable initial learning rate.
    """
    criterion = nn.CrossEntropyLoss()
    best_lr = None
    for _ in range(max_iters):
        mid = 10 ** ((math.log10(low) + math.log10(high)) / 2)
        print(f"Phase-1: trying lr={mid:.2e}  (low={low:.2e}, high={high:.2e})")
        model = model_fn().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=mid)
        for epoch in range(epochs_per_trial):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
        val_loss = loss_on_loader(model, val_loader, criterion)
        train_loss = loss_on_loader(model, train_loader, criterion)
        print(f"    -> train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        # simple heuristic: if val_loss explodes, it's overshooting
        if val_loss > train_loss * 1.5 or math.isnan(val_loss):
            high = mid
            print("    overshoot detected, lowering high")
        else:
            low = mid
            best_lr = mid
            print("    undershoot or ok, raising low")
        if high / low < 1.5:
            break
    return best_lr or 1e-3

# ------------------------------------------------------------------
# 4.  Phase-2: full training with dynamic LR
# ------------------------------------------------------------------
def dynamic_train(
        model,
        train_loader,
        val_loader,
        init_lr: float,
        max_epochs: int = 50,
        patience: int = 3,
        factor: float = 0.5,
        grow_factor: float = 1.3):
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=factor, patience=patience, verbose=True)

    best_val = float("inf")
    best_model = None
    history = []

    for epoch in range(max_epochs):
        # --- train one epoch ---
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # --- validate ---
        val_loss = loss_on_loader(model, val_loader, criterion)
        history.append((train_loss, val_loss))

        print(f"Epoch {epoch:02d}:  train={train_loss:.4f}  val={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}")

        # dynamic adjustment based on simple rules
        if len(history) > 1:
            prev_train, prev_val = history[-2]
            # overshoot: val increases while train decreases
            if val_loss > prev_val and train_loss < prev_train:
                new_lr = max(opt.param_groups[0]['lr'] * 0.3, 1e-7)
                print("  -> overshoot detected, reducing lr")
                for g in opt.param_groups:
                    g['lr'] = new_lr
            # undershoot: both losses stuck high
            elif abs(train_loss - prev_train) < 1e-3 and train_loss > 1.0:
                new_lr = min(opt.param_groups[0]['lr'] * grow_factor, 1e-1)
                print("  -> undershoot detected, increasing lr")
                for g in opt.param_groups:
                    g['lr'] = new_lr

        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_model = copy.deepcopy(model.state_dict())
        elif epoch - patience > 0 and val_loss > best_val:
            print("Early stopping triggered")
            break

    if best_model:
        model.load_state_dict(best_model)
    return model, history

# ------------------------------------------------------------------
# 5.  Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader = get_loaders()
    print("Phase-1: Binary search for initial learning rate ...")
    init_lr = binary_search_lr(TinyCNN, train_loader, val_loader)
    print(f"Chosen initial LR = {init_lr:.2e}\n")

    print("Phase-2: Full training with dynamic LR ...")
    model = TinyCNN().to(DEVICE)
    model, hist = dynamic_train(model, train_loader, val_loader, init_lr)

    # quick test accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)