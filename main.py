import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ограничение потоков для CPU
torch.set_num_threads(1)

# Фиксация рабочей директории
OS_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
RESULTS_DIR = os.path.join(OS_DIR, "results")
FIGURES_DIR = os.path.join(OS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Подготовка данных
# ---------------------------------------------------------
def prepare_data():
    X, y = make_moons(n_samples=2400, noise=0.25, random_state=2026)

    # 20% test, оставшиеся 80% разбить в 3:1 (train: 60%, val: 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2026, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=2026, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler

# ---------------------------------------------------------
# 2. Модель MLP
# ---------------------------------------------------------
class FlexibleMLP(nn.Module):
    def __init__(self, L, activation_name: str, init_type):
        super(FlexibleMLP, self).__init__()
        self.L = L
        self.layers = nn.ModuleList()

        # Выбор функции активации
        if activation_name == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_name == 'Tanh':
            self.activation = nn.Tanh()
        elif activation_name == 'ReLU':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation_name}")

        # Входной и скрытые слои
        in_dim = 2
        for i in range(L):
            layer = nn.Linear(in_dim, 32)
            self._init_hidden(layer, init_type)
            self.layers.append(layer)
            in_dim = 32

        # Выходной слой
        self.out_layer = nn.Linear(32, 2)
        nn.init.xavier_normal_(self.out_layer.weight, gain=1.0)
        nn.init.zeros_(self.out_layer.bias)
      
    def _init_hidden(self, layer, init_type):
        if init_type == 'Xavier normal':
            nn.init.xavier_normal_(layer.weight, gain=1.0)
        elif init_type == 'He normal':
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        nn.init.zeros_(layer.bias)


    def forward(self, x):
        h = x
        for layer in self.layers:
            z = layer(h)
            h = self.activation(z)
        out = self.out_layer(h)
        return out

# ---------------------------------------------------------
# 3. Функции оценки метрик и измеритель градиентов
# ---------------------------------------------------------
def evaluate(model, X_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor).item()
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_tensor).float().mean().item()
    return loss, acc

def compute_probe_gradients(model, X_probe, y_probe, criterion):
    model.eval()
    model.zero_grad()
    outputs = model(X_probe)
    loss = criterion(outputs, y_probe)
    loss.backward()

    grad_norms = []
    for l_idx, layer in enumerate(model.layers):
        if layer.weight.grad is not None:
            norm = torch.norm(layer.weight.grad, p='fro').item()
            grad_norms.append((l_idx + 1, norm))
        else:
            grad_norms.append((l_idx + 1, 0.0))
    model.zero_grad()
    return grad_norms

# ---------------------------------------------------------
# 4. Основной эксперимент
# ---------------------------------------------------------
def run_experiments():
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), scaler = prepare_data()

    # Визуализация 1: Диаграмма рассеяния train
    plt.figure(figsize=(6, 5))
    plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap='coolwarm', alpha=0.7, edgecolors='k', s=20)
    plt.title("Train set (StandardScaler)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "dataset_scatter.png"))
    plt.close()

    # Подготовка тензоров
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    X_probe_t = X_tr_t[:128]
    y_probe_t = y_tr_t[:128]

    configs = {
        'A': {'L': 6, 'act': 'Sigmoid', 'init': 'Xavier normal', 'opt': 'Adam', 'lr': 0.003},
        'B': {'L': 6, 'act': 'Tanh',    'init': 'Xavier normal', 'opt': 'Adam', 'lr': 0.003},
        'C': {'L': 6, 'act': 'ReLU',    'init': 'Xavier normal', 'opt': 'Adam', 'lr': 0.003},
        'D': {'L': 6, 'act': 'ReLU',    'init': 'He normal',     'opt': 'Adam', 'lr': 0.003},
        'E': {'L': 6, 'act': 'ReLU',    'init': 'He normal',     'opt': 'SGD',  'lr': 0.05},
        'F': {'L': 1, 'act': 'ReLU',    'init': 'He normal',     'opt': 'Adam', 'lr': 0.003},
    }

    seeds = [11, 29]
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metrics_records = []
    gradient_records = []
    trained_models = {}

    for cfg_code, cfg in configs.items():
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            model = FlexibleMLP(cfg['L'], cfg['act'], cfg['init'])

            if cfg['opt'] == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
            elif cfg['opt'] == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0)

            generator = torch.Generator().manual_seed(10000 + seed)
            dataset = TensorDataset(X_tr_t, y_tr_t)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True, generator=generator)

            # Эпоха 0
            tr_loss, tr_acc = evaluate(model, X_tr_t, y_tr_t, criterion)
            va_loss, va_acc = evaluate(model, X_va_t, y_va_t, criterion)
            metrics_records.append({
                'config': cfg_code, 'seed': seed, 'epoch': 0,
                'train_loss': tr_loss, 'train_accuracy': tr_acc,
                'val_loss': va_loss, 'val_accuracy': va_acc
            })

            # Градиенты на эпохе 0 для A, C, D
            if cfg_code in ['A', 'C', 'D']:
                g_norms = compute_probe_gradients(model, X_probe_t, y_probe_t, criterion)
                for layer_idx, g_norm in g_norms:
                    gradient_records.append({
                        'config': cfg_code, 'seed': seed, 'epoch': 0,
                        'layer': layer_idx, 'gradient_norm': g_norm
                    })

            # Обучение
            for epoch in range(1, 101):
                model.train()
                for bx, by in dataloader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

                tr_loss, tr_acc = evaluate(model, X_tr_t, y_tr_t, criterion)
                va_loss, va_acc = evaluate(model, X_va_t, y_va_t, criterion)
                metrics_records.append({
                    'config': cfg_code, 'seed': seed, 'epoch': epoch,
                    'train_loss': tr_loss, 'train_accuracy': tr_acc,
                    'val_loss': va_loss, 'val_accuracy': va_acc
                })

                # Градиенты на эпохах 20 и 100
                if cfg_code in ['A', 'C', 'D'] and epoch in [20, 100]:
                    g_norms = compute_probe_gradients(model, X_probe_t, y_probe_t, criterion)
                    for layer_idx, g_norm in g_norms:
                        gradient_records.append({
                            'config': cfg_code, 'seed': seed, 'epoch': epoch,
                            'layer': layer_idx, 'gradient_norm': g_norm
                        })

            trained_models[(cfg_code, seed)] = model

    # Сохранение результатов
    df_metrics = pd.DataFrame(metrics_records)
    df_grads = pd.DataFrame(gradient_records)

    df_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_history.csv"), index=False)
    df_grads.to_csv(os.path.join(RESULTS_DIR, "gradient_norms.csv"), index=False)

    # ---------------------------------------------------------
    # 5. Визуализация графиков
    # ---------------------------------------------------------
    # График (а): Validation loss при seed 11
    plt.figure(figsize=(8, 5))
    for cfg_code in ['A', 'B', 'C', 'D', 'E', 'F']:
        sub = df_metrics[(df_metrics['config'] == cfg_code) & (df_metrics['seed'] == 11)]
        plt.plot(sub['epoch'], sub['val_loss'], label=f"Config {cfg_code}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch (Seed 11)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "exp_val_loss_seed11.png"))
    plt.close()

    # График (б): Validation accuracy при seed 11
    plt.figure(figsize=(8, 5))
    for cfg_code in ['A', 'B', 'C', 'D', 'E', 'F']:
        sub = df_metrics[(df_metrics['config'] == cfg_code) & (df_metrics['seed'] == 11)]
        plt.plot(sub['epoch'], sub['val_accuracy'], label=f"Config {cfg_code}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Epoch (Seed 11)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "exp_val_acc_seed11.png"))
    plt.close()

    # График градиентов A, C, D (seed 11)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for idx, cfg_code in enumerate(['A', 'C', 'D']):
        ax = axes[idx]
        for ep in [0, 20, 100]:
            sub = df_grads[(df_grads['config'] == cfg_code) & (df_grads['seed'] == 11) & (df_grads['epoch'] == ep)]
            ax.plot(sub['layer'], sub['gradient_norm'], marker='o', label=f"Epoch {ep}")
        ax.set_yscale('log')
        ax.set_xlabel("Hidden Layer Index l")
        ax.set_ylabel("Frobenius Norm g_l")
        ax.set_title(f"Config {cfg_code} Gradients")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "gradient_norms_seed11.png"))
    plt.close()

    # ---------------------------------------------------------
    # 6. Выбор модели и оценка на Test
    # ---------------------------------------------------------
    ep100 = df_metrics[df_metrics['epoch'] == 100]
    avg_val_loss = ep100.groupby('config')['val_loss'].mean().to_dict()

    best_config = min(sorted(avg_val_loss.keys()), key=lambda c: avg_val_loss[c])

    cfg_ep100 = ep100[ep100['config'] == best_config]
    best_seed = cfg_ep100.sort_values(by='val_loss')['seed'].iloc[0]

    best_model = trained_models[(best_config, best_seed)]
    test_loss, test_acc = evaluate(best_model, X_te_t, y_te_t, criterion)

    # График для выбранной модели
    best_sub = df_metrics[(df_metrics['config'] == best_config) & (df_metrics['seed'] == best_seed)]
    plt.figure(figsize=(7, 4.5))
    plt.plot(best_sub['epoch'], best_sub['train_loss'], label="Train Loss")
    plt.plot(best_sub['epoch'], best_sub['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Best Model (Config {best_config}, Seed {best_seed}) Loss Curves")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "best_model_curves.png"))
    plt.close()

    print("=== SUMMARY RESULTS AT EPOCH 100 ===")
    summary_table = ep100.pivot(index='config', columns='seed', values=['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
    print(summary_table)
    print("\nMean Val Loss across seeds:", avg_val_loss)
    print(f"Selected Configuration: {best_config}")
    print(f"Selected Run Seed: {best_seed}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    run_experiments()
