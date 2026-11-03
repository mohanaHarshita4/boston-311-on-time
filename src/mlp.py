"""
Early Flag: MLP skeleton for Boston 311, no training yet.

Run: python src/mlp.py
Input: data/raw/boston311_2025.csv, through src/features.py
Output: prints the model shape, the parameter count, and a
32 row forward pass shape check. No training happens here.
Training comes next, on Nov 4.

The model uses the same intake only features and the same
time based split as baselines.py: train Jan to Sep 2025,
test Oct to Dec 2025. Only closed requests are used, because
open requests have no outcome yet.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    x_train, x_test, y_train, y_test = build_features()
    model = MLP(input_dim=x_train.shape[1])
    total = sum(p.numel() for p in model.parameters())
    print(f"Input features: {x_train.shape[1]}")
    print(f"Model parameters: {total:,}")
    dummy = torch.zeros(32, x_train.shape[1])
    with torch.no_grad():
        out = model(dummy)
    print(f"Forward pass output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
