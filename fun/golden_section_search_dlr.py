# Golden-section Search Dynamic Learning Rate (DLR)
"""
This script implements a learning rate finder based on Golden-section Search.
It is an optimization over Ternary Search, reducing the number of function
evaluations per iteration from two to one (after the first iteration).

Golden-section search is a technique for finding the extremum of a unimodal
function by successively narrowing the range of values inside which the
extremum is known to exist.

The algorithm works as follows:
1.  Define a search range [low, high] for the learning rate.
2.  Choose two points m1 and m2 within the range according to the golden
    ratio.
3.  In each iteration, one of the points m1 or m2 can be reused for the
    next iteration's search, which makes it efficient.
4.  The search range is narrowed based on the comparison of function values
    at m1 and m2.
5.  Repeat until the search range is small enough.

This file is self-contained and demonstrates the method on the MNIST dataset.
"""

import math, time, copy, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple

# ------------------------------------------------------------------
# 1.  A simple CNN for MNIST
# ------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ------------------------------------------------------------------
# 2.  Utilities
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loaders(root: str = "./data", batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Returns MNIST train and validation data loaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root, train=True,  download=True, transform=transform)
    val_set   = datasets.MNIST(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def loss_on_loader(model, loader, criterion):
    """
    Computes the loss of a model on a given data loader.
    """
    model.eval()
    total, loss_sum = 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            total += x.size(0)
    return loss_sum / total

def train_for_search(model_fn, train_loader, val_loader, lr, epochs_per_trial, criterion):
    """
    Helper function to train a model for a few epochs and return the validation loss.
    """
    model = model_fn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs_per_trial):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
    return loss_on_loader(model, val_loader, criterion)

# ------------------------------------------------------------------
# 3.  Phase-1: Golden-section search for initial LR
# ------------------------------------------------------------------
def golden_section_search_lr(
        model_fn,
        train_loader,
        val_loader,
        low: float = 1e-6,
        high: float = 1e-1,
        max_iters: int = 10,
        epochs_per_trial: int = 2) -> float:
    """
    Returns a reasonable initial learning rate using Golden-section search.
    """
    criterion = nn.CrossEntropyLoss()
    log_low, log_high = math.log10(low), math.log10(high)

    golden_ratio = (math.sqrt(5) + 1) / 2

    # Initial points
    m2_log = log_high - (log_high - log_low) / golden_ratio
    m1_log = log_low + (log_high - log_low) / golden_ratio

    f_m2 = train_for_search(model_fn, train_loader, val_loader, 10**m2_log, epochs_per_trial, criterion)
    f_m1 = train_for_search(model_fn, train_loader, val_loader, 10**m1_log, epochs_per_trial, criterion)

    for i in range(max_iters):
        print(f"Iteration {i+1}/{max_iters}: Searching in [{10**log_low:.2e}, {10**log_high:.2e}]")
        print(f"  lr1={10**m1_log:.2e} -> val_loss={f_m1:.4f}")
        print(f"  lr2={10**m2_log:.2e} -> val_loss={f_m2:.4f}")

        if f_m1 < f_m2:
            log_high = m2_log
            m2_log = m1_log
            f_m2 = f_m1
            m1_log = log_low + (log_high - log_low) / golden_ratio
            f_m1 = train_for_search(model_fn, train_loader, val_loader, 10**m1_log, epochs_per_trial, criterion)
            print("  f(m1) < f(m2), narrowing to the left.")
        else:
            log_low = m1_log
            m1_log = m2_log
            f_m1 = f_m2
            m2_log = log_high - (log_high - log_low) / golden_ratio
            f_m2 = train_for_search(model_fn, train_loader, val_loader, 10**m2_log, epochs_per_trial, criterion)
            print("  f(m1) >= f(m2), narrowing to the right.")

        if abs(log_high - log_low) < math.log10(1.5):
            print("  Search range is small enough, stopping.")
            break

    return 10**((log_low + log_high) / 2)

# ------------------------------------------------------------------
# 4.  Phase-2: full training with dynamic LR
# ------------------------------------------------------------------
def dynamic_train(
        model,
        train_loader,
        val_loader,
        init_lr: float,
        max_epochs: int = 20,
        patience: int = 3,
        factor: float = 0.5,
        grow_factor: float = 1.3):
    """
    This is the same dynamic training phase from DLR.py.
    """
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=factor, patience=patience, verbose=True)

    best_val = float("inf")
    best_model = None
    history = []

    for epoch in range(max_epochs):
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

        val_loss = loss_on_loader(model, val_loader, criterion)
        history.append((train_loss, val_loss))

        print(f"Epoch {epoch:02d}:  train={train_loss:.4f}  val={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}")

        if len(history) > 1:
            prev_train, prev_val = history[-2]
            if val_loss > prev_val and train_loss < prev_train:
                new_lr = max(opt.param_groups[0]['lr'] * 0.3, 1e-7)
                print("  -> overshoot detected, reducing lr")
                for g in opt.param_groups:
                    g['lr'] = new_lr
            elif abs(train_loss - prev_train) < 1e-3 and train_loss > 1.0:
                new_lr = min(opt.param_groups[0]['lr'] * grow_factor, 1e-1)
                print("  -> undershoot detected, increasing lr")
                for g in opt.param_groups:
                    g['lr'] = new_lr

        scheduler.step(val_loss)

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
    print("Phase-1: Golden-section search for initial learning rate ...")
    init_lr = golden_section_search_lr(SimpleCNN, train_loader, val_loader)
    print(f"\nChosen initial LR = {init_lr:.2e}\n")

    print("Phase-2: Full training with dynamic LR ...")
    model = SimpleCNN().to(DEVICE)
    model, hist = dynamic_train(model, train_loader, val_loader, init_lr)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"\nFinal validation accuracy: {correct/total:.4f}")
