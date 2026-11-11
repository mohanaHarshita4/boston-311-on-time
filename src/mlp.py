"""Early Flag: tune the PyTorch MLP for Boston 311.

Run: python src/mlp.py
Input: data/raw/boston311_2025.csv, through src/features.py
Output: prints a small table, one row per setting, and writes
        reports/mlp_tuning.md

Same intake-only features and same time split as baselines.py:
train Jan-Sep 2025, test Oct-Dec 2025. Only closed requests.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features

ROOT = os.path.join(os.path.dirname(__file__), "..")
REPORTS = os.path.join(ROOT, "reports")
NUM = ["sla_days", "open_hour", "open_weekday", "open_month"]

CONFIGS = [
    {"name": "64-32, lr 0.001, no dropout", "hidden1": 64, "hidden2": 32,
     "lr": 0.001, "dropout": 0.0},
    {"name": "128-64, lr 0.001, dropout 0.2", "hidden1": 128, "hidden2": 64,
     "lr": 0.001, "dropout": 0.2},
    {"name": "64-32, lr 0.0005, dropout 0.2", "hidden1": 64, "hidden2": 32,
     "lr": 0.0005, "dropout": 0.2},
]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


def run_config(config, x_train_t, x_test_t, y_train_t, y_test_np, pos_weight):
    model = MLP(input_dim=x_train_t.shape[1],
                hidden1=config["hidden1"], hidden2=config["hidden2"],
                dropout=config["dropout"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    epochs = 5
    batch_size = 512
    n = len(x_train_t)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(x_train_t[idx])
            loss = criterion(out, y_train_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        print(f"[{config['name']}] Epoch {epoch + 1}/{epochs}, "
              f"loss: {total_loss / batches:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(x_test_t)
        pred = (torch.sigmoid(logits) >= 0.5).int().numpy().flatten()

    return {
        "name": config["name"],
        "accuracy": accuracy_score(y_test_np, pred),
        "f1": f1_score(y_test_np, pred, zero_division=0),
        "recall": recall_score(y_test_np, pred, zero_division=0),
    }


def main():
    x_train, x_test, y_train, y_test = build_features()

    x_train = x_train.copy()
    x_test = x_test.copy()
    for col in NUM:
        mean = x_train[col].mean()
        std = x_train[col].std()
        if std > 0:
            x_train[col] = (x_train[col] - mean) / std
            x_test[col] = (x_test[col] - mean) / std

    # Bool columns to float, torch needs numbers.
    x_train_t = torch.tensor(x_train.values.astype(np.float32))
    x_test_t = torch.tensor(x_test.values.astype(np.float32))
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_np = y_test.values

    train_rate = float(y_train.mean())
    pos_weight = torch.tensor((1 - train_rate) / train_rate, dtype=torch.float32)
    print(f"Train overdue rate: {train_rate:.4f}, pos_weight: {pos_weight.item():.2f}")

    results = []
    for config in CONFIGS:
        result = run_config(config, x_train_t, x_test_t,
                            y_train_t, y_test_np, pos_weight)
        results.append(result)
        print(f"{result['name']}: accuracy {result['accuracy']:.4f}, "
              f"F1 {result['f1']:.4f}, overdue recall {result['recall']:.4f}")

    best = max(results, key=lambda r: r["f1"])

    table = (
        "| Setting | Accuracy | F1 (overdue) | Overdue recall |\n"
        "|---|---|---|---|\n"
    )
    for r in results:
        table += (f"| {r['name']} | {r['accuracy']:.4f} "
                  f"| {r['f1']:.4f} | {r['recall']:.4f} |\n")
    summary = (
        "# MLP tuning — Boston 311, 2025\n\n"
        "Same intake-only features and same time-based split as baselines: "
        f"train Jan-Sep 2025 ({len(x_train):,} rows, "
        f"overdue {y_train.mean():.4f}), test Oct-Dec 2025 "
        f"({len(x_test):,} rows, overdue {y_test.mean():.4f}). "
        f"Features: {x_train.shape[1]}. Each setting: Adam, batch 512, "
        f"5 epochs, BCE loss with pos_weight {pos_weight.item():.2f}, "
        "threshold 0.5. Best setting by F1 for the overdue class: "
        f"{best['name']}.\n\n"
        + table
    )
    print(summary)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "mlp_tuning.md"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
