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
import shutil
from tag_model import TagModel

from synergy_model import (
    build_model,
    calculate_weighted_loss,
    eval_loop,
    build_optimizer,
)
from bert_emb_tags import BertEmbedRegressor, initialize_bert_model

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
    def __init__(self, synergy_data, card_data, tokenizer, max_length=320, tags_len=None, subset_indices=None, dataset_name="joint"):
        synergy_data = json.load(open(synergy_data, "r"))
        card_data = json.load(open(card_data, "r"))

        self.synergy_data = synergy_data
        if subset_indices is not None:
            self.synergy_data = [self.synergy_data[i] for i in subset_indices]

        self.tokenizer = tokenizer
        self.card_lookup = build_card_lookup(card_data)
        self.max_length = max_length
        self.tags_len = tags_len
        self.dataset_name = dataset_name
        
        if tags_len is not None:
            all_tags = set()
            for c in self.card_lookup.values():
                if "tags" in c and c["tags"]:
                    all_tags.update(c["tags"])

            self.tag_to_index = {tag: i for i, tag in enumerate(all_tags)}
            if len(self.tag_to_index) != tags_len:
                raise ValueError(
                    f"Expected {tags_len} tags, but found {len(self.tag_to_index)} unique tags in the dataset."
                )
            self.tag_counts = torch.zeros(self.tags_len, dtype=torch.float32)
            self.total_tag_samples = 0

            for synergy_pair in self.synergy_data:
                card1 = self.find_card_by_name(synergy_pair["card1"]["name"])
                card2 = self.find_card_by_name(synergy_pair["card2"]["name"])

                if card1:
                    vec1 = self.hot_encode_tags(card1)
                    if vec1.shape[0] > 0:
                        self.tag_counts += vec1
                        self.total_tag_samples += 1

                if card2:
                    vec2 = self.hot_encode_tags(card2)
                    if vec2.shape[0] > 0:
                        self.tag_counts += vec2
                        self.total_tag_samples += 1

        self.calculate_synergy_counts()
        self.print_synergy()



    def __len__(self):
        return len(self.synergy_data)
    
    def calculate_synergy_counts(self):
        self.counts = [0,0,0,0]
        for synergy in self.synergy_data:
            if "synergy_edhrec" in synergy:
                self.counts[0] += 1
                if synergy["synergy"] == 0:
                    self.counts[1] += 1
            elif synergy.get("synergy", 0) == 1:
                self.counts[2] += 1
            else:
                self.counts[3] += 1

    def print_synergy(self):
        print(f"Dataset {self.dataset_name} counts: "
                f"Real Synergy = 1: {self.counts[0]},"
                f"Real Synergy = 0: {self.counts[1]},"
                f"Fake Synergy = 1: {self.counts[2]},"  
                f"Fake Synergy = 0: {self.counts[3]}"
 )

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
        label = torch.tensor([entry.get("synergy", 0)], dtype=torch.float32)

        tag_hot1 = self.hot_encode_tags(card1)
        tag_hot2 = self.hot_encode_tags(card2)

        return {
            "input_ids1": inputs1["input_ids"].squeeze(0),
            "attention_mask1": inputs1["attention_mask"].squeeze(0),
            "input_ids2": inputs2["input_ids"].squeeze(0),
            "attention_mask2": inputs2["attention_mask"].squeeze(0),
            "label": label,
            "tag_hot1": tag_hot1,
            "tag_hot2": tag_hot2,
        }
    
    def hot_encode_tags(self, card):
        """
        Convert card tags to a one-hot encoded vector.
        """
        if self.tags_len is None:
            return torch.zeros(0, dtype=torch.float32)
        if "tags" not in card or not card["tags"]:
            return torch.zeros(self.tags_len, dtype=torch.float32)
        
        
        tag_vector = np.zeros(len(self.tag_to_index), dtype=np.float32)
        
        
        for tag in card["tags"]:
            if tag in self.tag_to_index:
                tag_vector[self.tag_to_index[tag]] = 1.0
            else:
                raise ValueError(
                    f"Tag '{tag}' not found in tag_to_index mapping. Available tags: {', '.join(self.tag_to_index.keys())}"
                )
        
        return torch.tensor(tag_vector, dtype=torch.float32)

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


def build_training_components(config, bert_model, synergy_model, device, tag_model=None, tag_model_pos_weight=None, synergy_model_pos_weight=1.0):
    optimizer = build_joint_optimizer(
        config["optimizer"],
        bert_model,
        synergy_model,
        config["bert_learning_rate"],
        config["synergy_learning_rate"],
        config.get("optimizer_config", {}),
        tag_model=tag_model,
        tag_lr=config.get("tag_learning_rate", None),

    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(synergy_model_pos_weight).to(device)
    )   

    if tag_model is not None and tag_model_pos_weight is not None:
        loss_tag_fn = nn.BCEWithLogitsLoss().to(device)

    return optimizer, loss_fn, loss_tag_fn


def build_joint_optimizer(
    optimizer_name, bert_model, synergy_model, bert_lr, synergy_lr, optimizer_config, tag_model=None, tag_lr=None
):
    param_groups = [
        {"params": bert_model.parameters(), "lr": bert_lr},
        {"params": synergy_model.parameters(), "lr": synergy_lr},
    ]
    if tag_model is not None and tag_lr is not None:
        param_groups.append({"params": tag_model.parameters(), "lr": tag_lr})

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
    tag_model,
    dataloader,
    optimizer,
    loss_synergy_model,
    loss_tag_model,
    epoch,
    writer,
    device,
    false_positive_penalty=1.0,
    tag_loss_weight=1,
    accumulation_steps=1,
    use_empty_cache=False,
):

    bert_model.train()
    synergy_model.train()
    if tag_model is not None:
        tag_model.train()
    total_synergy_loss = 0.0
    total_tag_loss = 0.0
    all_preds_synergy, all_labels_synergy = [], []
    if tag_model is not None:
        all_preds_tag, all_labels_tag = [], []

    scaler = GradScaler()

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Train")):
        input_ids1 = batch["input_ids1"].to(device)
        attention_mask1 = batch["attention_mask1"].to(device)
        input_ids2 = batch["input_ids2"].to(device)
        attention_mask2 = batch["attention_mask2"].to(device)
        labels_synergy = batch["label"].to(device)

        tag_hot1 = None
        tag_hot2 = None
        if tag_model is not None:
            tag_hot1 = batch["tag_hot1"].to(device)
            tag_hot2 = batch["tag_hot2"].to(device)

        with autocast(device_type="cuda"):
            # Get BERT embeddings for both cards
            embed1 = bert_model(input_ids1, attention_mask1)
            embed2 = bert_model(input_ids2, attention_mask2)

            tag_loss = 0.0
            if tag_model is not None:
                tags_pred1 = tag_model(embed1)
                tags_pred2 = tag_model(embed2)

            # If tag_model is not None, calculate tag loss
            
                tag_loss1 = loss_tag_model(tags_pred1, tag_hot1)
                tag_loss2 = loss_tag_model(tags_pred2, tag_hot2)
                tag_loss = (tag_loss1 + tag_loss2) / 2.0

                preds_tag1 = torch.sigmoid(tags_pred1)
                preds_tag2 = torch.sigmoid(tags_pred2)
                
            # Forward pass through synergy model
            logits_synergy = synergy_model(embed1, embed2)

            weighted_loss_synergy, preds_synergy, _ = calculate_weighted_loss(
                logits_synergy, labels_synergy, loss_synergy_model, false_positive_penalty=false_positive_penalty
            )

            # Combine losses
            full_loss = weighted_loss_synergy + tag_loss_weight * tag_loss
            # Normalize loss for gradient accumulation
            loss_scaled = full_loss / accumulation_steps

        # AMP backward pass
        scaler.scale(loss_scaled).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if use_empty_cache:
                torch.cuda.empty_cache()

        total_synergy_loss += weighted_loss_synergy.item()  # already scaled back for logging
        total_tag_loss += tag_loss.item() * tag_loss_weight
        all_preds_synergy.extend(preds_synergy.cpu().numpy())
        all_labels_synergy.extend(labels_synergy.cpu().numpy().astype(int))
        if tag_model is not None:
            all_preds_tag.extend(preds_tag1.detach().cpu().numpy())
            all_preds_tag.extend(preds_tag2.detach().cpu().numpy())
            all_labels_tag.extend(tag_hot1.cpu().numpy())
            all_labels_tag.extend(tag_hot2.cpu().numpy())

    # Final step if dataset isn't divisible by accumulation_steps
    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = (total_tag_loss+total_synergy_loss) / len(dataloader)
    avg_synergy_loss = total_synergy_loss / len(dataloader)
    avg_tag_loss = total_tag_loss / len(dataloader)
    precision_synergy = precision_score(all_labels_synergy, all_preds_synergy, zero_division=0)
    recall_synergy = recall_score(all_labels_synergy, all_preds_synergy, zero_division=0)
    f1_synergy = f1_score(all_labels_synergy, all_preds_synergy, zero_division=0)
    cm_synergy = confusion_matrix(all_labels_synergy, all_preds_synergy)

    print(
        f"Train | Loss: {avg_loss:.4f} | Precision: {precision_synergy:.4f} | Recall: {recall_synergy:.4f} | F1: {f1_synergy:.4f} | "
        f"Synergy Loss: {avg_synergy_loss:.4f} | Tag Loss: {avg_tag_loss:.4f}"
    )

    if tag_model is not None:
        precision_tag = precision_score(all_labels_tag, all_preds_tag, average='macro', zero_division=0)
        recall_tag = recall_score(all_labels_tag, all_preds_tag, average='macro', zero_division=0)
        f1_tag = f1_score(all_labels_tag, all_preds_tag, average='macro', zero_division=0)
        print(f"Train | Tag Loss: {avg_tag_loss:.4f}  "
              f"| Precision Tag: {precision_tag:.4f} | Recall Tag: {recall_tag:.4f} | F1 Tag: {f1_tag:.4f} |")
        
        writer.add_scalar("Train/ Tag Loss", avg_tag_loss, epoch)
        writer.add_scalar("Train_tag/ Precision", precision_tag, epoch)
        writer.add_scalar("Train_tag/ Recall", recall_tag, epoch)
        writer.add_scalar("Train_tag/ F1", f1_tag, epoch)

    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Synergy Loss", avg_synergy_loss, epoch)
    writer.add_scalar("Train/Precision", precision_synergy, epoch)
    writer.add_scalar("Train/Recall", recall_synergy, epoch)
    writer.add_scalar("Train/F1", f1_synergy, epoch)
    writer.add_scalar("Train_cmSin/TP", cm_synergy[1, 1], epoch)
    writer.add_scalar("Train_cmSin/TN", cm_synergy[0, 0], epoch)
    writer.add_scalar("Train_cmSin/FP", cm_synergy[0, 1], epoch)
    writer.add_scalar("Train_cmSin/FN", cm_synergy[1, 0], epoch)

    return avg_loss, precision_synergy, recall_synergy, f1_synergy, cm_synergy


def eval_loop(
    bert_model,
    synergy_model,
    dataloader,
    loss_synergy_model,
    epoch,
    writer,
    device,
    label="Val",
    false_positive_penalty=1.0,
    tag_model=None,
    loss_tag_model=None,
    tag_loss_weight=1.0,
):
    bert_model.eval()
    synergy_model.eval()
    if tag_model is not None:
        tag_model.eval()

    total_synergy_loss = 0.0
    total_tag_loss = 0.0
    all_preds_sinergy, all_labels_sinergy = [], []
    if tag_model is not None:
        all_preds_tag, all_labels_tag = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{label} Eval"):
            input_ids1 = batch["input_ids1"].to(device)
            attention_mask1 = batch["attention_mask1"].to(device)
            input_ids2 = batch["input_ids2"].to(device)
            attention_mask2 = batch["attention_mask2"].to(device)
            labels_synergy = batch["label"].to(device)

            tag_hot1 = batch.get("tag_hot1")
            tag_hot2 = batch.get("tag_hot2")
            if tag_hot1 is not None and tag_hot2 is not None:
                tag_hot1 = tag_hot1.to(device)
                tag_hot2 = tag_hot2.to(device)

            # Get embeddings
            embed1 = bert_model(input_ids1, attention_mask1)
            embed2 = bert_model(input_ids2, attention_mask2)

            tag_loss = 0.0
            if tag_model is not None and tag_hot1 is not None:
                tags_pred1 = tag_model(embed1)
                tags_pred2 = tag_model(embed2)

                tag_loss1 = loss_tag_model(tags_pred1, tag_hot1)
                tag_loss2 = loss_tag_model(tags_pred2, tag_hot2)
                tag_loss = (tag_loss1 + tag_loss2) / 2.0

                preds_tag1 = torch.sigmoid(tags_pred1)
                preds_tag2 = torch.sigmoid(tags_pred2)

                

            # Synergy model
            logits_synergy = synergy_model(embed1, embed2)
            weighted_loss_synergy, preds_synergy, _ = calculate_weighted_loss(
                logits_synergy, labels_synergy, loss_synergy_model, false_positive_penalty=false_positive_penalty
            )

            total_synergy_loss += weighted_loss_synergy.item()
            total_tag_loss += tag_loss.item() * tag_loss_weight
            all_preds_sinergy.extend(preds_synergy.cpu().numpy())
            all_labels_sinergy.extend(labels_synergy.cpu().numpy().astype(int))
            if tag_model is not None:
                all_preds_tag.extend(preds_tag1.cpu().numpy())
                all_preds_tag.extend(preds_tag2.cpu().numpy())
                all_labels_tag.extend(tag_hot1.cpu().numpy())
                all_labels_tag.extend(tag_hot2.cpu().numpy())

    avg_loss = (total_synergy_loss + total_tag_loss) / len(dataloader)
    avg_synergy_loss = total_synergy_loss / len(dataloader)
    avg_tag_loss = total_tag_loss / len(dataloader)

    precision_sinergy = precision_score(all_labels_sinergy, all_preds_sinergy, zero_division=0)
    recall_synergy = recall_score(all_labels_sinergy, all_preds_sinergy, zero_division=0)
    f1_synergy = f1_score(all_labels_sinergy, all_preds_sinergy, zero_division=0)
    cm_synergy = confusion_matrix(all_labels_sinergy, all_preds_sinergy)


    print(
        f"{label} | Loss: {avg_loss:.4f}  "
        f"| Synergy Loss: {avg_synergy_loss:.4f} | Precision Synergy: {precision_sinergy:.4f} | Recall Synergy: {recall_synergy:.4f} | F1 Synergy: {f1_synergy:.4f} |"
    )
    if tag_model is not None:
        precision_tag = precision_score(all_labels_tag, all_preds_tag, average='macro', zero_division=0)
        recall_tag = recall_score(all_labels_tag, all_preds_tag, average='macro', zero_division=0)
        f1_tag = f1_score(all_labels_tag, all_preds_tag, average='macro', zero_division=0)
        print(f"{label} | Tag Loss: {avg_tag_loss:.4f}  "
              f"| Precision Tag: {precision_tag:.4f} | Recall Tag: {recall_tag:.4f} | F1 Tag: {f1_tag:.4f} |")
        
        writer.add_scalar(f"{label}/Tag Loss", avg_tag_loss, epoch)
        writer.add_scalar(f"{label}_tag/ Precision", precision_tag, epoch)
        writer.add_scalar(f"{label}_tag/ Recall", recall_tag, epoch)
        writer.add_scalar(f"{label}_tag/ F1", f1_tag, epoch)


    writer.add_scalar(f"{label}/Loss", avg_loss, epoch)
    writer.add_scalar(f"{label}/Synergy Loss", avg_synergy_loss, epoch)
    writer.add_scalar(f"{label}_sin/Precision", precision_sinergy, epoch)
    writer.add_scalar(f"{label}_sin/Recall", recall_synergy, epoch)
    writer.add_scalar(f"{label}_sin/F1", f1_synergy, epoch)
    writer.add_scalar(f"{label}_cmSin/TP", cm_synergy[1, 1], epoch)
    writer.add_scalar(f"{label}_cmSin/TN", cm_synergy[0, 0], epoch)
    writer.add_scalar(f"{label}_cmSin/FP", cm_synergy[0, 1], epoch)
    writer.add_scalar(f"{label}_cmSin/FN", cm_synergy[1, 0], epoch)

    return avg_loss, precision_sinergy, recall_synergy, f1_synergy, cm_synergy


def split_indices(real_indices, fake_indices, splits, log_splits=False):
    random.shuffle(real_indices)
    random.shuffle(fake_indices)

    num_real = len(real_indices)
    num_fake = len(fake_indices)

    real_ptr = 0
    fake_ptr = 0

    real_allocations = {}
    fake_allocations = {}

    # Allocate real indices
    for split_name, ratios in splits.items():
        count = int(ratios.get("real", 0) * num_real)
        real_allocations[split_name] = real_indices[real_ptr : real_ptr + count]
        real_ptr += count

    # Allocate fake indices
    for split_name, ratios in splits.items():
        count = int(ratios.get("fake", 0) * num_fake)
        fake_allocations[split_name] = fake_indices[fake_ptr : fake_ptr + count]
        fake_ptr += count

    # Combine real and fake indices per split
    final_splits = {}
    for split_name in splits:
        final_splits[split_name] = real_allocations.get(
            split_name, []
        ) + fake_allocations.get(split_name, [])
        if log_splits:
            print(
                f"{split_name} - Real: {len(real_allocations[split_name])}, Fake: {len(fake_allocations[split_name])}, Total: {len(final_splits[split_name])}"
            )

    return final_splits

def create_dataloaders(config, tokenizer, index_splits):

    # Create DataLoaders
    dataloaders = {}
    for split_name, indices in index_splits.items():
        dataset = JointCardDataset(
            config["synergy_file"],
            config["bulk_file"],
            tokenizer,
            max_length=config["max_length_bert_tokenizer"],
            tags_len=config.get("tag_output_dim", None),
            subset_indices=indices,
            dataset_name=split_name,
        )

        is_train = split_name == "train"
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=is_train,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2 if is_train else None,
        )

    return dataloaders

def clone_content_of_dir(src_dir, dest_dir):
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def setup_dirs_writer(config):
    start_epoch = config.get("start_epoch", 0)
    if start_epoch is None:
        start_epoch = 0
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")
        # Load the last save model writer and logs
        previous_run_name = config["run_name_previous"]
        # clone the log directory
        prev_log_full_dir = os.path.join(config["log_dir"], previous_run_name)
        prev_save_full_dir = os.path.join(config["save_dir"], previous_run_name)
        if not os.path.exists(prev_save_full_dir) or not os.path.exists(prev_log_full_dir):
            raise FileNotFoundError(
                f"Previous run directories do not exist: {prev_save_full_dir} or {prev_log_full_dir}"
            )
        new_run_name = config["run_name"]
        log_full_dir = os.path.join(config["log_dir"], new_run_name)
        save_full_dir = os.path.join(config["save_dir"], new_run_name)
        os.makedirs(log_full_dir, exist_ok=True)
        os.makedirs(save_full_dir, exist_ok=True)
        clone_content_of_dir(prev_log_full_dir, log_full_dir)
        clone_content_of_dir(prev_save_full_dir, save_full_dir)
        writer = SummaryWriter(log_dir=log_full_dir)
        
        print(f"Loaded previous run logs from {prev_log_full_dir}")
    else:
        print("Starting from scratch, no previous epoch to resume.")

        log_full_dir = os.path.join(config["log_dir"], config["run_name"])
        save_full_dir = os.path.join(config["save_dir"], config["run_name"])

        os.makedirs(log_full_dir, exist_ok=True)
        os.makedirs(save_full_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=log_full_dir)

        

        # write var config with writer
        writer.add_text("Config", json.dumps(config, indent=4))
    
    return writer, save_full_dir, start_epoch

def train_joint_model(config):
    
    # Set up directories and writer
    writer, save_full_dir, start_epoch = setup_dirs_writer(config)

    print(f"Run: {config["run_name"]}")

    # Step 1: Load and split indices
    real_indices, fake_indices = get_real_fake_indices(config["synergy_file"])

    # Define split proportions
    splits = {
        "train": {"real": 0.8, "fake": 0.1},
        "val_real": {"real": 0.1, "fake": 0.0},
        "val_real_fake": {"real": 0.1, "fake": 0.03},
    }

    split_indices_result = split_indices(
        real_indices, fake_indices, splits, log_splits=True
    )

    # Step 2: Initialize BERT model and tokenizer
    model_name = config["bert_model_name"]
    embedding_dim = config.get("embedding_dim", 384)
    bert_model, tokenizer, device = initialize_bert_model(model_name, embedding_dim)

    data_loaders = create_dataloaders(config, tokenizer, split_indices_result)

    train_loader = data_loaders["train"]

    # Models loading

    if config["bert_checkpoint"] and config["bert_checkpoint"] != "":
        bert_model.load_state_dict(torch.load(config["bert_checkpoint"]))
        print(f"Loaded BERT checkpoint: {config['bert_checkpoint']}")

    synergy_model = build_model(config["synergy_arch"], config["embedding_dim"]).to(
        device
    )

    print(f"Using synergy model architecture: {config['synergy_arch']}")

    train_dataset = train_loader.dataset
    
    if config.get("synergy_pos_weight", None) is not None:
        synergy_model_pos_weight = torch.tensor(
            [config["synergy_pos_weight"]]
        ).to(device)
    else:
        synergy_1_counts = train_dataset.counts[0]+ train_dataset.counts[2]
        synergy_0_counts = train_dataset.counts[1] + train_dataset.counts[3]

        synergy_model_pos_weight = torch.tensor(
            [synergy_0_counts / synergy_1_counts]
        ).to(device)

    tag_model = None
    tag_model_pos_weight = None
    if config.get("use_tag_model", None):

        tag_model = TagModel(
            input_dim=config["embedding_dim"],
            hidden_dims=config.get("tag_hidden_dims", [512,256]),
            output_dim=config.get("tag_output_dim", 271),
            dropout=config.get("tag_dropout", 0.2),
            use_batchnorm=True,
            use_sigmoid_output=False
        ).to(device)

        
        tag_counts = train_dataset.tag_counts
        total = train_dataset.total_tag_samples

        # Compute pos_weight for each tag
        neg_counts = total - tag_counts
        tag_model_pos_weight = neg_counts / (tag_counts + 1e-6)


        print(f"Using tag model with output dimension: {config.get('tag_output_dim', 271)}")


    optimizer, loss_sin_fn, loss_tag_fn = build_training_components(
        config, bert_model, synergy_model, device, tag_model=tag_model, tag_model_pos_weight=tag_model_pos_weight
    )

    

    for epoch in tqdm(
        range(start_epoch, config["epochs"]),
        desc="Epochs",
        initial=start_epoch
    ):

        train_loop(
            bert_model,
            synergy_model,
            tag_model,
            train_loader,
            optimizer,
            loss_sin_fn,
            loss_tag_fn,
            epoch,
            writer,
            device,
            tag_loss_weight=config.get("tag_loss_weight", 1.0),
            accumulation_steps=config["accumulation_steps"],
            use_empty_cache=config.get("use_empty_cache", False),
        )
        if (epoch + 1) % config["eval_every"] == 0:
            for split_name, loader in data_loaders.items():
                if split_name.startswith("val"):
                    eval_loop(
                        bert_model,
                        synergy_model,
                        loader,
                        loss_sin_fn,
                        epoch,
                        writer,
                        device,
                        label=split_name,
                        false_positive_penalty=config.get("false_positive_penalty", 1.0),
                        tag_model=tag_model,
                        loss_tag_model=loss_tag_fn,
                        tag_loss_weight=config.get("tag_loss_weight", 1.0),
                    )

        if (epoch + 1) % config["save_every"] == 0:
            bert_model_path = os.path.join(
                save_full_dir, f"bert_model_epoch_{epoch + 1}.pth"
            )
            synergy_model_path = os.path.join(
                save_full_dir, f"synergy_model_epoch_{epoch + 1}.pth"
            )
            if tag_model is not None:
                tag_model_path = os.path.join(
                    save_full_dir, f"tag_model_epoch_{epoch + 1}.pth"
                )
                torch.save(tag_model.state_dict(), tag_model_path)
                print(f"Saved tag model at epoch {epoch + 1}")
            
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
