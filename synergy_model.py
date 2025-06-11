import json
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from sklearn.metrics import precision_score, recall_score, f1_score

# ----------------------
# Model Architectures
# ----------------------
class ModelSimple(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, embed1, embed2):
        x = torch.cat([embed1, embed2], dim=-1)
        return self.net(x)


class ModelComplex(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        input_dim = 4 * embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, embed1, embed2):
        x = torch.cat([embed1, embed2, embed1 * embed2, torch.abs(embed1 - embed2)], dim=-1)
        return self.net(x)


# ----------------------
# Model Factory
# ----------------------
def build_model(arch_name, embedding_dim):
    if arch_name == "modelSimple":
        return ModelSimple(embedding_dim)
    elif arch_name == "modelComplex":
        return ModelComplex(embedding_dim)
    else:
        raise ValueError(f"Unknown model architecture: {arch_name}")


# ----------------------
# Optimizer Factory
# ----------------------
def build_optimizer(optimizer_name, model_params, lr, optimizer_config):
    if optimizer_name == "Adam":
        return optim.Adam(model_params, lr=lr, **optimizer_config)
    elif optimizer_name == "SGD":
        return optim.SGD(model_params, lr=lr, **optimizer_config)
    elif optimizer_name == "AdamW":
        return optim.AdamW(model_params, lr=lr, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class SynergyDataset(Dataset):
    def __init__(self, synergy_file, bulk_embedding_file):
        with open(bulk_embedding_file, 'r') as f:
            bulk_data = json.load(f)
        self.embedding_dict = {
            card['name']: torch.tensor(card['emb'][0], dtype=torch.float)
            for card in bulk_data if card.get('emb') and len(card['emb']) > 0
        }

        self.real_pairs = []
        self.fake_pairs = []

        with open(synergy_file, 'r') as f:
            synergy_data = json.load(f)

        for entry in synergy_data:
            card1 = entry['card1']['name']
            card2 = entry['card2']['name']
            synergy = entry.get('synergy', 0)

            if card1 not in self.embedding_dict or card2 not in self.embedding_dict:
                continue

            emb1 = self.embedding_dict[card1]
            emb2 = self.embedding_dict[card2]
            pair = (emb1, emb2, torch.tensor([synergy], dtype=torch.float))

            if "synergy_edhrec" in entry:
                self.real_pairs.append(pair)
            elif synergy == 0:
                self.fake_pairs.append(pair)

        # Placeholder; data will be set externally
        self.data = []

    def set_data(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

@torch.no_grad()
def eval_loop(model, dataloader, device, epoch, writer, prefix="Val"):
    model.eval()
    all_preds = []
    all_labels = []

    loop = tqdm(dataloader, desc=f"{prefix} Evaluation")
    for emb1, emb2, label in loop:
        emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device)

        logits = model(emb1, emb2).view(-1)
        preds = (logits > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    writer.add_scalar(f"{prefix}/Precision", precision, epoch)
    writer.add_scalar(f"{prefix}/Recall", recall, epoch)
    writer.add_scalar(f"{prefix}/F1", f1, epoch)

    print(f"{prefix} Epoch {epoch+1} — Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


# ----------------------
# Training Loop
# ----------------------
def train_one_config(config):
    run_name = config["run_name"]
    log_dir = f"runs/{run_name}"
    save_dir = f"checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = config["embedding_dim"]
    model = build_model(config["modelArchitecture"], embedding_dim).to(device)
    optimizer = build_optimizer(config["optimizer"], model.parameters(), config["learning_rate"], config["optimizer_config"])
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    # ----------------------
    # Dataset and DataLoader
    # ----------------------
    dataset = SynergyDataset(config["synergy_file"], config["embedding_file"])

    # Shuffle and split real pairs (80/20)
    real_pairs = dataset.real_pairs
    random.shuffle(real_pairs)
    split_idx = int(0.8 * len(real_pairs))
    real_train = real_pairs[:split_idx]
    real_val = real_pairs[split_idx:]

    # 100% of fake pairs go to training
    fake_train = dataset.fake_pairs

    # Create datasets with set_data
    train_dataset = SynergyDataset(config["synergy_file"], config["embedding_file"])
    train_dataset.set_data(fake_train + real_train)
    val_dataset = SynergyDataset(config["synergy_file"], config["embedding_file"])
    val_dataset.set_data(real_val)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    print(f"Training with {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")
    print(f"Real pairs: {len(real_pairs)}, Fake pairs: {len(dataset.fake_pairs)}")
    print(f"Training on {len(real_train)} real pairs and {len(fake_train)} fake pairs.")
    print(f"Model architecture: {config['modelArchitecture']}, Optimizer: {config['optimizer']}, Learning rate: {config['learning_rate']}")
    print(f"\n----- Run name: {run_name} -----\n")


    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_weighted_loss = 0

        all_preds, all_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        for emb1, emb2, label in loop:
            emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(emb1, emb2).view(-1)
            label = label.view(-1)

            loss = loss_fn(logits, label)
            preds = (logits > 0.5).float()

            weights = torch.ones_like(label)
            false_positive_mask = (label == 0) & (preds == 1)
            weights[false_positive_mask] = config.get("false_positive_penalty", 1.0)
            weighted_loss = (loss * weights).mean()

            weighted_loss.backward()
            optimizer.step()

            total_loss += loss.mean().item()
            total_weighted_loss += weighted_loss.item()

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())

            loop.set_postfix(weighted_loss=weighted_loss.item())

        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)


        writer.add_scalar("Train/Loss", total_loss / len(train_loader), epoch)
        writer.add_scalar("Train/WeightedLoss", total_weighted_loss / len(train_loader), epoch)
        writer.add_scalar("Train/Precision", precision, epoch)
        writer.add_scalar("Train/Recall", recall, epoch)
        writer.add_scalar("Train/F1", f1, epoch)

        if (epoch + 1) % config.get("eval_every", 5) == 0:
            print(f"Evaluating on validation set at epoch {epoch + 1}...")
            eval_loop(model, val_loader, device, epoch, writer, prefix="Val")

        # Optional: Save model checkpoint
        if (epoch + 1) % config.get("save_every", 50) == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    writer.close()


# ----------------------
# Multi-Config Runner
# ----------------------
def run_all_configs(config_path):
    with open(config_path) as f:
        config_list = json.load(f)

    for config in config_list:
        print(f"Starting run: {config['run_name']}")
        train_one_config(config)
        print(f"Finished run: {config['run_name']}\n\n")


# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python synergy_model.py <config_file.json>")
        exit(1)

    config_file = sys.argv[1]
    run_all_configs(config_file)