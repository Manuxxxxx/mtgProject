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
            nn.Linear(64, 1)
            # Removed Sigmoid: use BCEWithLogitsLoss + torch.sigmoid in forward or after logits
        )

    def forward(self, embed1, embed2):
        x = torch.cat([embed1, embed2], dim=-1)
        return self.net(x)


class ModelComplex(nn.Module):
    def __init__(self, embedding_dim, tag_projector_dim):
        super().__init__()
        input_dim = 4 * embedding_dim + 4 * tag_projector_dim  # embed1, embed2, embed1*embed2, abs(embed1-embed2), tag_projector1, tag_projector2, tag_projector1*tag_projector2, abs(tag_projector1-tag_projector2)
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
            nn.Linear(64, 1)
            # Removed Sigmoid here too
        )

    def forward(self, embed1, embed2, tag_projector1, tag_projector2):
        x = torch.cat([embed1, embed2, embed1 * embed2, torch.abs(embed1 - embed2), tag_projector1, tag_projector2, tag_projector1 * tag_projector2, torch.abs(tag_projector1 - tag_projector2)], dim=-1)
        return self.net(x)


# ----------------------
# Weight Initialization
# ----------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ----------------------
# Model Factory
# ----------------------
def build_synergy_model(arch_name, embedding_dim):
    if arch_name == "modelSimple":
        model = ModelSimple(embedding_dim)
    elif arch_name == "modelComplex":
        model = ModelComplex(embedding_dim)
    else:
        raise ValueError(f"Unknown model architecture: {arch_name}")
    model.apply(init_weights)
    return model


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
                # print(f"Skipping pair ({card1}, {card2}) due to missing embeddings.")
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



# ----------------------
# Weighted Loss Calculation Helper
# ----------------------
def calculate_weighted_loss(logits, labels, loss_fn, false_positive_penalty=1.0):
    """
    logits: raw output from model (no sigmoid applied yet)
    labels: ground truth labels (0 or 1)
    loss_fn: loss function (expects raw logits for BCEWithLogitsLoss)
    false_positive_penalty: multiplier applied to false positives in the batch
    """
    loss = loss_fn(logits, labels)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    #print all the predictions and labels
    # print(f"Predictions: {preds.cpu().numpy()}, Labels: {labels.cpu().numpy()}")
    # print("Logits:", logits[:10].detach().cpu().numpy())

    weights = torch.where(labels == 0, false_positive_penalty, 1.0)    
    weighted_loss = (loss * weights).mean()

    # #calculate TP, TN, FP, FN as a dictionary
    confusion_matrix = {}

    
    return weighted_loss, preds, confusion_matrix


@torch.no_grad()
def eval_loop(model, dataloader, device, epoch, writer, prefix="Val", false_positive_penalty=1.0, BCEweight = 1.0):
    model.eval()
    all_preds = []
    all_labels = []
    full_confusion_matrix = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }
    total_weighted_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([BCEweight]).to(device))


    loop = tqdm(dataloader, desc=f"{prefix} Evaluation")
    for emb1, emb2, label in loop:
        emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device).view(-1)
        logits = model(emb1, emb2).view(-1)

        weighted_loss, preds, confusion_matrix = calculate_weighted_loss(logits, label, loss_fn, false_positive_penalty)
        total_weighted_loss += weighted_loss.item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        for key in full_confusion_matrix:
            full_confusion_matrix[key] += confusion_matrix[key]

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    avg_weighted_loss = total_weighted_loss / len(dataloader)

    writer.add_scalar(f"{prefix}/WeightedLoss", avg_weighted_loss, epoch)
    writer.add_scalar(f"{prefix}/Precision", precision, epoch)
    writer.add_scalar(f"{prefix}/Recall", recall, epoch)
    writer.add_scalar(f"{prefix}/F1", f1, epoch)

    writer.add_scalar(f"{prefix}/confusion_matrix/TP", full_confusion_matrix['TP'], epoch)
    writer.add_scalar(f"{prefix}/confusion_matrix/TN", full_confusion_matrix['TN'], epoch)
    writer.add_scalar(f"{prefix}/confusion_matrix/FP", full_confusion_matrix['FP'], epoch)
    writer.add_scalar(f"{prefix}/confusion_matrix/FN", full_confusion_matrix['FN'], epoch)


    print(f"| {prefix} | Epoch {epoch+1} — Weighted Loss: {avg_weighted_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"| {prefix} | Confusion Matrix: TP={full_confusion_matrix['TP']}, TN={full_confusion_matrix['TN']}, FP={full_confusion_matrix['FP']}, FN={full_confusion_matrix['FN']}")


def initialize_run_dirs(config):
    run_name = config["run_name"] + "_" + time.strftime("_%Y%m%d_%H%M%S")
    log_dir = f"runs/{run_name}"
    save_dir = f"checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    return run_name, log_dir, save_dir


def build_training_components(config, device):
    model = build_synergy_model(config["modelArchitecture"], config["embedding_dim"]).to(device)
    optimizer = build_optimizer(config["optimizer"], model.parameters(), config["learning_rate"], config["optimizer_config"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.get("BCEweight", 1.0)]).to(device))
    return model, optimizer, loss_fn


def prepare_dataloaders(config):
    dataset = SynergyDataset(config["synergy_file"], config["embedding_file"])

    real_pairs = dataset.real_pairs
    random.shuffle(real_pairs)
    split_real = int(0.8 * len(real_pairs))
    real_train, real_val = real_pairs[:split_real], real_pairs[split_real:]

    split_fake = int(0.9 * len(dataset.fake_pairs))
    fake_train, fake_val = dataset.fake_pairs[:split_fake], dataset.fake_pairs[split_fake:]

    train_dataset = SynergyDataset(config["synergy_file"], config["embedding_file"])
    train_dataset.set_data(fake_train + real_train)

    val_dataset = SynergyDataset(config["synergy_file"], config["embedding_file"])
    val_dataset.set_data(real_val[:int(len(real_val) * 0.5)])

    val_fake_dataset = SynergyDataset(config["synergy_file"], config["embedding_file"])
    val_fake_dataset.set_data(fake_val + real_val[int(len(real_val) * 0.5):])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    val_fake_loader = DataLoader(val_fake_dataset, batch_size=config["batch_size"])

    return train_loader, val_loader, val_fake_loader


def print_training_summary(train_loader, val_loader, val_fake_loader, config, run_name):
    print(f"Training with {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} real validation samples and {len(val_fake_loader.dataset)} fake&real validation samples.")
    print(f"Model architecture: {config['modelArchitecture']}, Optimizer: {config['optimizer']}, Learning rate: {config['learning_rate']}")
    print(f"\n----- Run name: {run_name} -----\n")


def train_one_epoch(model, train_loader, optimizer, loss_fn, writer, device, epoch, config):
    model.train()
    total_weighted_loss = 0
    all_preds, all_labels = [], []
    confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1} Train")
    for emb1, emb2, label in loop:
        emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device).view(-1)

        optimizer.zero_grad()
        logits = model(emb1, emb2).view(-1)

        weighted_loss, preds, cm = calculate_weighted_loss(
            logits, label, loss_fn, false_positive_penalty=config.get("false_positive_penalty", 1.0)
        )

        weighted_loss.backward()
        optimizer.step()

        total_weighted_loss += weighted_loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(label.detach().cpu().numpy())

        for key in confusion_matrix:
            confusion_matrix[key] += cm[key]

        loop.set_postfix(weighted_loss=weighted_loss.item())

    log_training_metrics(writer, epoch, total_weighted_loss, train_loader, all_labels, all_preds, confusion_matrix)


def log_training_metrics(writer, epoch, total_loss, loader, labels, preds, confusion_matrix):
    writer.add_scalar("Train/WeightedLoss", total_loss / len(loader), epoch)
    writer.add_scalar("Train/Precision", precision_score(labels, preds, zero_division=0), epoch)
    writer.add_scalar("Train/Recall", recall_score(labels, preds, zero_division=0), epoch)
    writer.add_scalar("Train/F1", f1_score(labels, preds, zero_division=0), epoch)

    for k, v in confusion_matrix.items():
        writer.add_scalar(f"Train/confusion_matrix/{k}", v, epoch)

    print(f"Training | Confusion Matrix: TP={confusion_matrix['TP']}, TN={confusion_matrix['TN']}, FP={confusion_matrix['FP']}, FN={confusion_matrix['FN']}")

def save_model(model, save_dir, epoch):
    save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

# ----------------------
# Training Loop
# ----------------------
def train_one_config(config):
    run_name, log_dir, save_dir = initialize_run_dirs(config)
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, loss_fn = build_training_components(config, device)

    train_loader, val_loader, val_fake_loader = prepare_dataloaders(config)
    
    print_training_summary(train_loader, val_loader, val_fake_loader, config, run_name)

    for epoch in range(config["epochs"]):
        train_one_epoch(
            model, train_loader, optimizer, loss_fn, writer, device, epoch, config
        )

        if (epoch + 1) % config.get("eval_every", 5) == 0:
            eval_loop(model, val_loader, device, epoch, writer, "Val", config.get("false_positive_penalty", 1.0), config.get("BCEweight", 1.0))
            eval_loop(model, val_fake_loader, device, epoch, writer, "ValFake", config.get("false_positive_penalty", 1.0), config.get("BCEweight", 1.0))

        if (epoch + 1) % config.get("save_every", 50) == 0:
            save_model(model, save_dir, epoch)

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
