import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Subset
import time
import sys
from torch.amp import autocast, GradScaler

from synergy_model import (
    build_model,
    calculate_weighted_loss,
    eval_loop,
    build_optimizer,
)
from bert_emb_tags import BertEmbedRegressor

import bert_parsing


def set_seed(seed: int = 42):
    random.seed(seed)  # Python random module
    # np.random.seed(seed)                   # NumPy
    # torch.manual_seed(seed)                # PyTorch CPU
    # torch.cuda.manual_seed(seed)           # PyTorch GPU
    # torch.cuda.manual_seed_all(seed)       # All GPUs (if using DataParallel or DDP)

    # # Ensures deterministic behavior (at the expense of performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ------------------------
# Combined Dataset
# ------------------------
class JointCardDataset(Dataset):
    def __init__(self, synergy_data, card_data, tokenizer, max_length=320):
        synergy_data = json.load(open(synergy_data, "r"))
        card_data = json.load(open(card_data, "r"))
        self.synergy_data = synergy_data
        self.tokenizer = tokenizer
        self.card_lookup = build_card_lookup(card_data)
        self.max_length = max_length

    def __len__(self):
        return len(self.synergy_data)

    def __getitem__(self, idx):
        entry = self.synergy_data[idx]
        card1 = self.find_card_by_name(entry["card1"]["name"])
        card2 = self.find_card_by_name(entry["card2"]["name"])

        if card1 is None or card2 is None:
            raise ValueError(
                f"Missing cards: {entry['card1']['name']} or {entry['card2']['name']}"
            )

        if not card1 or not card2:
            raise ValueError(
                f"Card not found for entry {idx}: {entry['card1']['name']} or {entry['card2']['name']}"
            )

        inputs1 = self.tokenizer(
            bert_parsing.format_card_for_bert(card1),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs2 = self.tokenizer(
            bert_parsing.format_card_for_bert(card2),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        label = torch.tensor([entry.get("synergy", 0)], dtype=torch.float)
        return {
            "input_ids1": inputs1["input_ids"].squeeze(0),
            "attention_mask1": inputs1["attention_mask"].squeeze(0),
            "input_ids2": inputs2["input_ids"].squeeze(0),
            "attention_mask2": inputs2["attention_mask"].squeeze(0),
            "label": label,
        }

    def find_card_by_name(self, name):
        # Try exact match first
        card = self.card_lookup.get(name)
        if card is not None:
            return card
        # If no exact match, try substring match in keys
        for full_name, card_data in self.card_lookup.items():
            if name in full_name:
                return card_data
        # Not found
        return None


def build_card_lookup(card_data):
    """
    Build a dictionary mapping card names to their full data.

    Args:
        card_data (list): List of card dictionaries, each with a 'name' key.

    Returns:
        dict: A lookup dict of {card_name: card_dict}
    """
    if not isinstance(card_data, list):
        raise ValueError("Expected card_data to be a list of dicts.")

    card_lookup = {}
    for i, card in enumerate(card_data):
        if not isinstance(card, dict):
            raise TypeError(f"Item at index {i} is not a dict: {card!r}")
        if "name" not in card:
            raise KeyError(f"Missing 'name' key in card at index {i}: {card!r}")
        card_lookup[card["name"]] = card

    return card_lookup


def get_real_fake_indices(synergy_file):
    """
    Load the synergy file and return the indices for real and fake entries.
    Real = contains "synergy_edhrec"
    Fake = does not contain it and synergy == 0
    """
    with open(synergy_file, "r") as f:
        data = json.load(f)

    real_indices = []
    fake_indices = []

    for i, entry in enumerate(data):
        if "synergy_edhrec" in entry:
            real_indices.append(i)
        elif entry.get("synergy", 0) == 0:
            fake_indices.append(i)

    return real_indices, fake_indices


def build_training_components(config, bert_model, synergy_model, device):
    optimizer = build_joint_optimizer(
        config["optimizer"],
        bert_model,
        synergy_model,
        config["bert_learning_rate"],
        config["synergy_learning_rate"],
        config.get("optimizer_config", {}),
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.get("BCEweight", 1.0)]).to(device)
    )

    return optimizer, loss_fn


def build_joint_optimizer(
    optimizer_name, bert_model, synergy_model, bert_lr, synergy_lr, optimizer_config
):
    param_groups = [
        {"params": bert_model.parameters(), "lr": bert_lr},
        {"params": synergy_model.parameters(), "lr": synergy_lr},
    ]

    if optimizer_name == "Adam":
        return optim.Adam(param_groups, **optimizer_config)
    elif optimizer_name == "AdamW":
        return optim.AdamW(param_groups, **optimizer_config)
    elif optimizer_name == "SGD":
        return optim.SGD(param_groups, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_loop(
    bert_model,
    synergy_model,
    dataloader,
    optimizer,
    loss_fn,
    epoch,
    writer,
    device,
    false_positive_penalty=1.0,
    accumulation_steps=1,
    use_empty_cache=False,
):

    bert_model.train()
    synergy_model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    scaler = GradScaler()

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Train")):
        input_ids1 = batch["input_ids1"].to(device)
        attention_mask1 = batch["attention_mask1"].to(device)
        input_ids2 = batch["input_ids2"].to(device)
        attention_mask2 = batch["attention_mask2"].to(device)
        labels = batch["label"].to(device)

        with autocast(device_type="cuda"):
            # Get BERT embeddings for both cards
            embed1 = bert_model(input_ids1, attention_mask1)
            embed2 = bert_model(input_ids2, attention_mask2)

            # Forward pass through synergy model
            logits = synergy_model(embed1, embed2)

            weighted_loss, preds, _ = calculate_weighted_loss(
                logits, labels, loss_fn, false_positive_penalty=false_positive_penalty
            )

            # Normalize loss for gradient accumulation
            loss_scaled = weighted_loss / accumulation_steps

        # AMP backward pass
        scaler.scale(loss_scaled).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if use_empty_cache:
                torch.cuda.empty_cache()

        total_loss += weighted_loss.item()  # already scaled back for logging
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy().astype(int))

    # Final step if dataset isn't divisible by accumulation_steps
    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(
        f"Train | Loss: {avg_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )
    print(f"Train | Confusion Matrix:\n")
    print(f"Total | TP: {cm[1,1]}, TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}")

    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Precision", precision, epoch)
    writer.add_scalar("Train/Recall", recall, epoch)
    writer.add_scalar("Train/F1", f1, epoch)
    writer.add_scalar("Train/TP", cm[1, 1], epoch)
    writer.add_scalar("Train/TN", cm[0, 0], epoch)
    writer.add_scalar("Train/FP", cm[0, 1], epoch)
    writer.add_scalar("Train/FN", cm[1, 0], epoch)

    return avg_loss, precision, recall, f1, cm


def eval_loop(
    bert_model,
    synergy_model,
    dataloader,
    loss_fn,
    epoch,
    writer,
    device,
    label="Val",
    false_positive_penalty=1.0,
):
    bert_model.eval()
    synergy_model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{label} Eval"):
            input_ids1 = batch["input_ids1"].to(device)
            attention_mask1 = batch["attention_mask1"].to(device)
            input_ids2 = batch["input_ids2"].to(device)
            attention_mask2 = batch["attention_mask2"].to(device)
            labels = batch["label"].to(device)

            # Get BERT embeddings for both cards
            embed1 = bert_model(input_ids1, attention_mask1)
            embed2 = bert_model(input_ids2, attention_mask2)

            # Forward pass through synergy model
            logits = synergy_model(embed1, embed2)

            weighted_loss, preds, _ = calculate_weighted_loss(
                logits, labels, loss_fn, false_positive_penalty=false_positive_penalty
            )

            total_loss += weighted_loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))

    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(
        f"{label} | Loss: {avg_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )
    print(f"{label} | Confusion Matrix (scikit-learn):\n")
    print(
        f"{label} | Total TP: {np.sum(cm[1, 1])}, TN: {np.sum(cm[0, 0])}, "
        f"FP: {np.sum(cm[0, 1])}, FN: {np.sum(cm[1, 0])}"
    )

    writer.add_scalar(f"{label}/Loss", avg_loss, epoch)
    writer.add_scalar(f"{label}/Precision", precision, epoch)
    writer.add_scalar(f"{label}/Recall", recall, epoch)
    writer.add_scalar(f"{label}/F1", f1, epoch)
    writer.add_scalar(f"{label}/conf_matrix/TP", np.sum(cm[1, 1]), epoch)
    writer.add_scalar(f"{label}/conf_matrix/TN", np.sum(cm[0, 0]), epoch)
    writer.add_scalar(f"{label}/conf_matrix/FP", np.sum(cm[0, 1]), epoch)
    writer.add_scalar(f"{label}/conf_matrix/FN", np.sum(cm[1, 0]), epoch)

    return avg_loss, precision, recall, f1, cm


def train_joint_model(config):
    log_full_dir = os.path.join(config["log_dir"], config["run_name"])
    save_full_dir = os.path.join(config["save_dir"], config["run_name"])

    os.makedirs(log_full_dir, exist_ok=True)
    os.makedirs(save_full_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_full_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Run: {config["run_name"]}")

    # write var config with writer
    writer.add_text("Config", json.dumps(config, indent=4))

    # Step 1: Load and split indices
    real_indices, fake_indices = get_real_fake_indices(config["synergy_file"])
    random.shuffle(real_indices)
    random.shuffle(fake_indices)

    # Split into sets
    # 80% train, 10% validation for real, 10% validation for fake

    # Define split proportions
    splits = {
        "train": {"real": 0.8, "fake": 0.1},
        "val_real": {"real": 0.1, "fake": 0.0},
        "val_real_fake": {"real": 0.1, "fake": 0.03},
    }

    random.shuffle(real_indices)
    random.shuffle(fake_indices)

    # Compute counts
    num_real = len(real_indices)
    num_fake = len(fake_indices)

    real_train_end = int(splits["train"]["real"] * num_real)
    real_val_end = real_train_end + int(splits["val_real"]["real"] * num_real)
    real_val2_end = real_val_end + int(splits["val_real_fake"]["real"] * num_real)

    fake_train_end = int(splits["train"]["fake"] * num_fake)
    fake_val_end = fake_train_end + int(splits["val_real"]["fake"] * num_fake)
    fake_val2_end = fake_val_end + int(splits["val_real_fake"]["fake"] * num_fake)

    # Real subsets
    real_train = real_indices[:real_train_end]
    real_val = real_indices[real_train_end:real_val_end]
    real_val_2 = real_indices[real_val_end:real_val2_end]

    # Fake subsets
    fake_train = fake_indices[:fake_train_end]
    fake_val = fake_indices[fake_train_end:fake_val_end]
    fake_val_2 = fake_indices[fake_val_end:fake_val2_end]

    # Final indices
    train_indices = real_train + fake_train
    val_real_indices = real_val + fake_val
    val_real_fake_indices = real_val_2 + fake_val_2

    model_name = config["bert_model_name"]
    tokenizer = None
    if model_name is None or model_name == "":
        print("Using default BertTokenizer")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_name == "bert-base-uncased":
        print("Using BertTokenizer")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_name == "distilbert-base-uncased":
        print("Using DistilBertTokenizer")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # Step 2: Build full dataset and wrap in Subsets
    full_dataset = JointCardDataset(
        config["synergy_file"],
        config["bulk_file"],
        tokenizer,
        config["max_length_bert_tokenizer"],
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_real_indices)
    val_fake_dataset = Subset(full_dataset, val_real_fake_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=2,  # 4 parallel loading processes
        pin_memory=True,  # enable faster transfer to GPU
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], drop_last=True
    )
    val_fake_loader = DataLoader(
        val_fake_dataset, batch_size=config["batch_size"], drop_last=True
    )

    print(
        f"Train size: {len(train_loader.dataset)}, real: {len(real_train)}, fake: {len(fake_train)}"
    )
    print(
        f"Val Real size: {len(val_loader.dataset)}, real: {len(real_val)}, fake: {len(fake_val)}"
    )
    print(
        f"Val Fake+Real size: {len(val_fake_loader.dataset)}, real: {len(real_val_2)}, fake: {len(fake_val_2)}"
    )

    # Models
    bert_model = BertEmbedRegressor(
        config["embedding_dim"], model_name=config["bert_model_name"]
    ).to(device)
    if config["bert_checkpoint"] and config["bert_checkpoint"] != "":
        bert_model.load_state_dict(torch.load(config["bert_checkpoint"]))
        print(f"Loaded BERT checkpoint: {config['bert_checkpoint']}")

    synergy_model = build_model(config["synergy_arch"], config["embedding_dim"]).to(
        device
    )

    optimizer, loss_fn = build_training_components(
        config, bert_model, synergy_model, device
    )

    for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
        train_loop(
            bert_model,
            synergy_model,
            train_loader,
            optimizer,
            loss_fn,
            epoch,
            writer,
            device,
            accumulation_steps=config["accumulation_steps"],
            use_empty_cache=config.get("use_empty_cache", False),
        )
        eval_loop(
            bert_model,
            synergy_model,
            val_loader,
            loss_fn,
            epoch,
            writer,
            device,
            label="Val Real",
        )
        eval_loop(
            bert_model,
            synergy_model,
            val_fake_loader,
            loss_fn,
            epoch,
            writer,
            device,
            label="Val Fake+Real",
        )

        if (epoch + 1) % config["save_every"] == 0:
            bert_model_path = os.path.join(
                save_full_dir, f"bert_model_epoch_{epoch + 1}.pth"
            )
            synergy_model_path = os.path.join(
                save_full_dir, f"synergy_model_epoch_{epoch + 1}.pth"
            )
            torch.save(bert_model.state_dict(), bert_model_path)
            torch.save(synergy_model.state_dict(), synergy_model_path)
            print(f"Saved models at epoch {epoch + 1}")

    # Final save
    bert_model_path = os.path.join(save_full_dir, "bert_model_final.pth")
    synergy_model_path = os.path.join(save_full_dir, "synergy_model_final.pth")
    torch.save(bert_model.state_dict(), bert_model_path)
    torch.save(synergy_model.state_dict(), synergy_model_path)
    print(f"Final models saved at {save_full_dir}")
    writer.close()


def run_all_configs(config_path):
    with open(config_path) as f:
        config_list = json.load(f)

    for config in config_list:
        config["run_name"] += time.strftime("_%Y%m%d_%H%M%S")  # Append timestamp here
        train_joint_model(config)
        print(f"Finished run\n\n")


if __name__ == "__main__":
    set_seed(1006)
    if len(sys.argv) != 2:
        print("Usage: python joint_train.py <config_file.json>")
        exit(1)

    config_file = sys.argv[1]
    run_all_configs(config_file)
