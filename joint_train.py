import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
import sys
from torch.amp import autocast, GradScaler
import shutil
from transformers import get_linear_schedule_with_warmup

from focal_loss import FocalLoss
from tag_model import  build_tag_model
from bert_model import build_bert_model
from synergy_model import build_synergy_model, calculate_weighted_loss
from tag_projector_model import build_tag_projector_model

import bert_parsing

ANSI_COLORS = {
    "reset": "\033[0m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bold": "\033[1m"
}

# Store original stdout
original_write = sys.stdout.write

# Current color state
current_color = ""

def set_color(color_name=None):
    """
    Set the global color for all print() calls.
    Call with None or 'reset' to reset to default.
    """
    global current_color
    if color_name is None or color_name == "reset":
        current_color = ""
        sys.stdout.write = original_write  # Restore default
    else:
        color_code = ANSI_COLORS.get(color_name.lower())
        if not color_code:
            raise ValueError(f"Unknown color: {color_name}")
        
        current_color = color_code

        def color_write(text):
            original_write(color_code + text + ANSI_COLORS["reset"] if text.strip() else text)

        sys.stdout.write = color_write

def calculate_stats_lenght_tokenizer(tokenizer, cards, max_length=450):
    """
    Calculate the average and max length of tokenized cards using the provided tokenizer.
    Plots all the lengths in a graph.
    
    Args:
        tokenizer: The tokenizer to use for tokenization.
        cards (list): List of card dictionaries to tokenize.
        max_length (int): Maximum length for tokenization.
        
    Returns:
        None: This function does not return anything, but it will plot a histogram of token lengths.
    """
    #plot histogram of lengths and save it
    import matplotlib.pyplot as plt

    tokenized_lengths = [len(tokenizer(bert_parsing.format_card_for_bert(card), truncation=True, max_length=max_length)["input_ids"]) for card in cards]
    avg_length = np.mean(tokenized_lengths)
    max_length = np.max(tokenized_lengths)
    min_length = np.min(tokenized_lengths)
    plt.hist(tokenized_lengths, bins=50, alpha=0.75)
    plt.title("Tokenized Lengths Histogram")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.axvline(avg_length, color='r', linestyle='dashed', linewidth=1, label=f'Avg Length: {avg_length:.2f}')
    plt.axvline(max_length, color='g', linestyle='dashed', linewidth=1, label=f'Max Length: {max_length}')
    plt.axvline(min_length, color='b', linestyle='dashed', linewidth=1, label=f'Min Length: {min_length}')
    plt.grid(True)
    plt.savefig("tokenized_lengths_histogram.png")
    plt.close()


def set_seed(seed: int = 42):
    random.seed(seed)  # Python random module
    # np.random.seed(seed)                   # NumPy
    # torch.manual_seed(seed)                # PyTorch CPU
    # torch.cuda.manual_seed(seed)           # PyTorch GPU
    # torch.cuda.manual_seed_all(seed)       # All GPUs (if using DataParallel or DDP)

    # # Ensures deterministic behavior (at the expense of performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class CardDataset(Dataset):
    def __init__(self, card_data, tokenizer, max_length=320, tags_len=None, dataset_name="tag", tag_to_index_file=None):
        
        self.tokenizer = tokenizer
        self.data = card_data
        self.max_length = max_length
        self.tags_len = tags_len
        self.dataset_name = dataset_name
        if tag_to_index_file is not None:
            with open(tag_to_index_file, "r") as f:
                self.tag_to_index = json.load(f)
    

        if tags_len is not None:
            if tag_to_index_file is None:
                all_tags = set()
                for card in self.data:
                    if "tags" in card and card["tags"]:
                        all_tags.update(card["tags"])

            
                self.tag_to_index = {tag: i for i, tag in enumerate(all_tags)}
                if len(self.tag_to_index) != tags_len:
                    raise ValueError(
                        f"Expected {tags_len} tags, but found {len(self.tag_to_index)} unique tags in the dataset."
                    )
            
            self.tag_counts = torch.zeros(self.tags_len, dtype=torch.float32)
            self.total_tag_samples = 0

            for card in self.data:
                tag_vec = self.hot_encode_tags(card)  # shape: (tags_len,)
                if tag_vec.shape[0] == self.tags_len:
                    self.tag_counts += tag_vec
                    self.total_tag_samples += 1
                
                card["tag_hot"] = tag_vec  # Add tag hot encoding to card data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        card = self.data[idx]
        inputs = self.tokenizer(
            bert_parsing.format_card_for_bert(card),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "tag_hot": card.get("tag_hot"),  # Add tag hot encoding to card data
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

# ------------------------
# Combined Dataset
# ------------------------
class JointCardDataset(Dataset):
    def __init__(self, synergy_data, card_data, tokenizer, max_length=320, tags_len=None, subset_indices=None, dataset_name="joint", debug_dataset=False, tag_to_index_file=None):
        synergy_data = json.load(open(synergy_data, "r"))
        card_data = json.load(open(card_data, "r"))

        self.synergy_data = synergy_data
        if subset_indices is not None:
            self.synergy_data = [self.synergy_data[i] for i in subset_indices]

        self.tokenizer = tokenizer
        self.build_card_lookup(card_data)
        self.max_length = max_length
        self.tags_len = tags_len
        self.dataset_name = dataset_name

        if tag_to_index_file is not None:
            with open(tag_to_index_file, "r") as f:
                self.tag_to_index = json.load(f)
        
        if tags_len is not None:
            if tag_to_index_file is None:
                # If no tag_to_index_file is provided, build the tag_to_index from the synergy data
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
                for card_key in ["card1", "card2"]:
                    card = self.find_card_by_name(synergy_pair[card_key]["name"])
                    if card:
                        tag_vec = self.hot_encode_tags(card)  # shape: (tags_len,)
                        if tag_vec.shape[0] == self.tags_len:
                            self.tag_counts += tag_vec
                            self.total_tag_samples += 1
                        
                        card["tag_hot"] = tag_vec  # Add tag hot encoding to card data


        self.calculate_synergy_counts()
        if debug_dataset:
            self.print_synergy()



    def __len__(self):
        return len(self.synergy_data)
    
    def calculate_synergy_counts(self):
        self.counts = [0,0,0,0]
        for synergy in self.synergy_data:
            if "synergy_edhrec" in synergy:
                if synergy.get("synergy", 0) == 1:
                    self.counts[0] += 1
                else:
                    self.counts[1] += 1
            elif synergy.get("synergy", 0) == 1:
                self.counts[2] += 1
            else:
                self.counts[3] += 1

    def print_synergy(self):
        print(f"Dataset {self.dataset_name} counts: "
                f"Real Syn=1: {self.counts[0]}, "
                f"Real Syn=0: {self.counts[1]}, "
                f"Fake Syn=1: {self.counts[2]}, "  
                f"Fake Syn=0: {self.counts[3]}"
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
        if label is None:
            raise ValueError(f"Missing synergy label for entry {idx}: {entry}")

        tag_hot1 = card1.get("tag_hot")
        tag_hot2 = card2.get("tag_hot")

        if tag_hot1 is None or tag_hot2 is None:
            raise ValueError(
                f"Missing tag hot encoding for cards: {card1['name']} or {card2['name']}"
            )
        

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

    def build_card_lookup(self, card_data):
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

        self.card_lookup = card_lookup

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

def get_loss_tag_fn(
    config, device, tag_model_pos_weight=None
):
    """
    Build the loss function for the tag model based on the configuration.
    If use_focal is True, use FocalLoss; otherwise, use weighted BCE.
    """
    if config.get("use_focal", False):
        # Use FocalLoss, normalize alpha if provided
        if tag_model_pos_weight is not None:
            alpha = tag_model_pos_weight / tag_model_pos_weight.max()  # Normalize to 0–1
        else:
            alpha = None

        loss_tag_fn = FocalLoss(
            alpha=alpha,
            gamma=config.get("focal_gamma", 2.0)
        ).to(device)

        print("Using Focal Loss for tag model with alpha:", alpha, "and gamma:", config.get("focal_gamma", 2.0))

    else:
        # Use weighted BCE
        loss_tag_fn = nn.BCEWithLogitsLoss(pos_weight=tag_model_pos_weight).to(device)
        print("Using weighted BCE Loss for tag model with pos_weight:", tag_model_pos_weight)

    return loss_tag_fn

def build_training_components_tag(config, bert_model, tag_model, device, tag_model_pos_weight=None):
    optimizer = build_tag_optimizer(
        optimizer_name=config["optimizer"],
        tag_model=tag_model,
        bert_model=bert_model,
        tag_lr=config["tag_learning_rate_tag"],
        bert_lr=config["bert_learning_rate_tag"],
        optimizer_config=config.get("optimizer_config", {}),
    )

    
    models_with_names = [
        ("bert_model", bert_model),
        ("tag_model", tag_model)
    ]

    print_models_param_summary(models_with_names, optimizer)

    print_separator()

    loss_tag_fn = get_loss_tag_fn(
        config=config,
        device=device,
        tag_model_pos_weight=tag_model_pos_weight
    )

    return optimizer, loss_tag_fn

def build_training_components_multitask(config, bert_model, synergy_model, device, tag_model, tag_projector_model, tag_model_pos_weight=1.0, synergy_model_pos_weight=1.0):
    optimizer = build_multitask_optimizer(
        bert_model=bert_model,
        synergy_model=synergy_model,
        tag_projector_model=tag_projector_model,
        tag_model=tag_model,
        bert_lr=config["bert_learning_rate_multi"],
        synergy_lr=config["synergy_learning_rate"],
        tag_projector_lr=config["tag_projector_learning_rate"],
        tag_lr= config["tag_learning_rate_multi"],
        optimizer_config=config.get("optimizer_config", {}),
        optimizer_name=config.get("optimizer")
    )

    models_with_names = [
        ("bert_model", bert_model),
        ("synergy_model", synergy_model),
        ("tag_projector_model", tag_projector_model),
        ("tag_model", tag_model)
    ]

    print_models_param_summary(models_with_names, optimizer)

    print_separator()

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=synergy_model_pos_weight
    ).to(device)

    print("Using BCEWithLogitsLoss for synergy model with pos_weight:", synergy_model_pos_weight)

    print_separator()

    loss_tag_fn = get_loss_tag_fn(
        config=config,
        device=device,
        tag_model_pos_weight=tag_model_pos_weight
    )

    return optimizer, loss_fn, loss_tag_fn

def build_multitask_optimizer(
    optimizer_name, bert_model, synergy_model, tag_projector_model, bert_lr, tag_projector_lr ,synergy_lr, optimizer_config, tag_model, tag_lr
):
    param_groups = [
        {"params": bert_model.parameters(), "lr": bert_lr, "name": "bert_model"},
        {"params": synergy_model.parameters(), "lr": synergy_lr, "name": "synergy_model"},
        {"params": tag_projector_model.parameters(), "lr": tag_projector_lr, "name": "tag_projector_model"},
        {"params": tag_model.parameters(), "lr": tag_lr, "name": "tag_model"}
    ]

    if optimizer_name == "Adam":
        return optim.Adam(param_groups, **optimizer_config)
    elif optimizer_name == "AdamW":
        return optim.AdamW(param_groups, **optimizer_config)
    elif optimizer_name == "SGD":
        return optim.SGD(param_groups, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
def build_tag_optimizer(
    optimizer_name, tag_model, bert_model, tag_lr, bert_lr, optimizer_config
):
    param_groups = [
        {"params": tag_model.parameters(), "lr": tag_lr, "name": "tag_model"},
        {"params": bert_model.parameters(), "lr": bert_lr, "name": "bert_model"},
    ]

    if optimizer_name == "Adam":
        return optim.Adam(param_groups, **optimizer_config)
    elif optimizer_name == "AdamW":
        return optim.AdamW(param_groups, **optimizer_config)
    elif optimizer_name == "SGD":
        return optim.SGD(param_groups, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def print_models_param_summary(models_with_names, optimizer):
    """
    Args:
        models_with_names: list of tuples (model_name, model_instance)
        optimizer: the optimizer containing param_groups
    """
    print("Optimizer name:", optimizer.__class__.__name__)
    print("Optimizer parameter groups:")
    for param_group in optimizer.param_groups:
        print(f"  - Learning rate: {param_group['lr']}, "
              f"Params: {len(param_group['params'])}, "
              f"Name: {param_group.get('name', 'N/A')}")

    print("\nModel parameter summary:")
    for name, model in models_with_names:
        if model is None:
            print(f"Model '{name}': None")
            continue

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Model '{name}':")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Frozen parameters: {frozen_params:,}")

def get_embeddings_and_tag_preds(
    bert_model, tag_model, tag_projector_model, batch, device
):
    input_ids1 = batch["input_ids1"].to(device)
    attention_mask1 = batch["attention_mask1"].to(device)
    input_ids2 = batch["input_ids2"].to(device)
    attention_mask2 = batch["attention_mask2"].to(device)

    embed1 = bert_model(input_ids1, attention_mask1)
    embed2 = bert_model(input_ids2, attention_mask2)

    tags_pred1 = tag_model(embed1)
    tags_pred2 = tag_model(embed2)

    preds_tag1 = torch.sigmoid(tags_pred1)
    preds_tag2 = torch.sigmoid(tags_pred2)

    projected_tag_embed1 = tag_projector_model(preds_tag1)
    projected_tag_embed2 = tag_projector_model(preds_tag2)

    return embed1, embed2, tags_pred1, tags_pred2, preds_tag1, preds_tag2, projected_tag_embed1, projected_tag_embed2

def compute_synergy_loss(
    synergy_model, embed1, embed2, projected_tag_embed1, projected_tag_embed2, labels_synergy, loss_synergy_model, false_positive_penalty
):
    logits_synergy = synergy_model(embed1, embed2, projected_tag_embed1, projected_tag_embed2)
    weighted_loss_synergy, preds_synergy, _ = calculate_weighted_loss(
        logits_synergy, labels_synergy, loss_synergy_model, false_positive_penalty=false_positive_penalty
    )
    return weighted_loss_synergy, preds_synergy

def update_metrics_multi(calc_metrics, all_preds_synergy, all_labels_synergy, all_preds_tag, all_labels_tag, preds_synergy, labels_synergy, preds_tag1, preds_tag2, tag_hot1, tag_hot2):
    if not calc_metrics:
        return
    all_preds_synergy.extend(preds_synergy.cpu().numpy())
    all_labels_synergy.extend(labels_synergy.cpu().numpy().astype(int))
    all_preds_tag.extend(preds_tag1.detach().cpu().numpy())
    all_preds_tag.extend(preds_tag2.detach().cpu().numpy())
    all_labels_tag.extend(tag_hot1.cpu().numpy())
    all_labels_tag.extend(tag_hot2.cpu().numpy())

def log_metrics_multitask(writer, epoch, avg_loss, avg_synergy_loss, avg_tag_loss, all_preds_synergy, all_labels_synergy, all_preds_tag, all_labels_tag, label_prefix="Train"):
    writer.add_scalar(f"{label_prefix}/Loss", avg_loss, epoch)
    writer.add_scalar(f"{label_prefix}/Synergy Loss", avg_synergy_loss, epoch)

    precision_synergy = precision_score(all_labels_synergy, all_preds_synergy, zero_division=0)
    recall_synergy = recall_score(all_labels_synergy, all_preds_synergy, zero_division=0)
    f1_synergy = f1_score(all_labels_synergy, all_preds_synergy, zero_division=0)
    cm_synergy = confusion_matrix(all_labels_synergy, all_preds_synergy)

    print(
        f"{label_prefix} | Loss: {avg_loss:.4f}  "
        f"| Synergy Loss: {avg_synergy_loss:.4f} | Precision Synergy: {precision_synergy:.4f} | Recall Synergy: {recall_synergy:.4f} | F1 Synergy: {f1_synergy:.4f} |"
    )

    writer.add_scalar(f"{label_prefix}/Precision", precision_synergy, epoch)
    writer.add_scalar(f"{label_prefix}/Recall", recall_synergy, epoch)
    writer.add_scalar(f"{label_prefix}/F1", f1_synergy, epoch)
    writer.add_scalar(f"{label_prefix}_cmSin/TP", cm_synergy[1, 1], epoch)
    writer.add_scalar(f"{label_prefix}_cmSin/TN", cm_synergy[0, 0], epoch)
    writer.add_scalar(f"{label_prefix}_cmSin/FP", cm_synergy[0, 1], epoch)
    writer.add_scalar(f"{label_prefix}_cmSin/FN", cm_synergy[1, 0], epoch)

    log_metrics_tag(writer, epoch, avg_tag_loss, all_preds_tag, all_labels_tag, label_prefix)

def log_metrics_tag(writer, epoch, avg_loss, all_preds_tag, all_labels_tag, label_prefix="Train"):
    writer.add_scalar(f"{label_prefix}/Tag Loss", avg_loss, epoch)
    binary_preds_tag = np.array(all_preds_tag) > 0.5

    # print(f"shape of all_preds_tag: {np.array(all_preds_tag).shape}")
    # print(f"shape of all_labels_tag: {np.array(all_labels_tag).shape}")
    # # Convert to integers (True → 1, False → 0)
    # binary_preds_tag_int = binary_preds_tag.astype(int)
    # # Total number of 1s
    # total_ones = np.sum(binary_preds_tag_int)
    # total_ones_labels = np.sum(np.array(all_labels_tag))
    # # Total number of predictions (575 * 103)
    # total_preds = binary_preds_tag_int.size
    # # Total number of 0s
    # total_zeros = total_preds - total_ones
    # total_zeros_labels = len(all_labels_tag) * binary_preds_tag_int.shape[1] - total_ones_labels
    # # Average number of 1s per prediction (per row)
    # avg_ones_per_prediction = np.mean(np.sum(binary_preds_tag_int, axis=1))
    # avg_ones_per_labels = np.mean(np.sum(np.array(all_labels_tag), axis=1))
    # # Output results
    # print(f"Total 1s: {total_ones}, Total 0s: {total_zeros}")
    # print(f"Total 1s in labels: {total_ones_labels}, Total 0s in labels: {total_zeros_labels}")
    # print(f"Average 1s per prediction: {avg_ones_per_prediction:.2f}")
    # print(f"Average 1s per labels: {avg_ones_per_labels:.2f}")
    
    precision_tag = precision_score(all_labels_tag, binary_preds_tag, average='macro', zero_division=0)
    recall_tag = recall_score(all_labels_tag, binary_preds_tag, average='macro', zero_division=0)
    f1_tag = f1_score(all_labels_tag, binary_preds_tag, average='macro', zero_division=0)
    cm_tag = confusion_matrix(np.array(all_labels_tag).flatten(), binary_preds_tag.flatten())

    print(f"{label_prefix} | Tag Loss: {avg_loss:.4f} | Precision Tag: {precision_tag:.4f} | Recall Tag: {recall_tag:.4f} | F1 Tag: {f1_tag:.4f} |")

    writer.add_scalar(f"{label_prefix}_tag/Precision", precision_tag, epoch)
    writer.add_scalar(f"{label_prefix}_tag/Recall", recall_tag, epoch)
    writer.add_scalar(f"{label_prefix}_tag/F1", f1_tag, epoch)
    writer.add_scalar(f"{label_prefix}_tag/cmTag/TP", cm_tag[1, 1], epoch)
    writer.add_scalar(f"{label_prefix}_tag/cmTag/TN", cm_tag[0, 0], epoch)
    writer.add_scalar(f"{label_prefix}_tag/cmTag/FP", cm_tag[0, 1], epoch)
    writer.add_scalar(f"{label_prefix}_tag/cmTag/FN", cm_tag[1, 0], epoch)

def train_tag_loop(
    bert_model,
    tag_model,
    dataloader,
    optimizer,
    loss_tag_model,
    epoch,
    writer,
    device,
    accumulation_steps=1,
    use_empty_cache=False,
    calc_metrics=False,
):
    tag_model.train()
    bert_model.train()
    total_tag_loss = 0.0
    if calc_metrics:
        all_preds_tag, all_labels_tag = [], []

    scaler = GradScaler()
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Train Tag")):
        tag_hot = batch["tag_hot"].to(device)

        with autocast(device_type="cuda"):
            embed = bert_model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            preds_tag = tag_model(embed)

            tag_loss = loss_tag_model(preds_tag, tag_hot)
            full_loss =  tag_loss
            loss_scaled = full_loss / accumulation_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if use_empty_cache:
                torch.cuda.empty_cache()

        total_tag_loss += tag_loss.item()

        if calc_metrics:
            all_preds_tag.extend(preds_tag.detach().cpu().numpy())
            all_labels_tag.extend(tag_hot.cpu().numpy())

    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_tag_loss / len(dataloader)

    log_metrics_tag(writer, epoch, avg_loss, all_preds_tag if calc_metrics else [], all_labels_tag if calc_metrics else [], "Train")

def eval_tag_loop(
    bert_model,
    tag_model,
    dataloader,
    loss_tag_model,
    epoch,
    writer,
    device
):
    tag_model.eval()
    bert_model.eval()

    total_tag_loss = 0.0
    all_preds_tag, all_labels_tag = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval Tag"):
            tag_hot = batch["tag_hot"].to(device)

            embed = bert_model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            preds_tag = tag_model(embed)

            tag_loss = loss_tag_model(preds_tag, tag_hot)
            total_tag_loss += tag_loss.item()

            all_preds_tag.extend(preds_tag.cpu().numpy())
            all_labels_tag.extend(tag_hot.cpu().numpy())

    avg_loss = total_tag_loss / len(dataloader)

    log_metrics_tag(writer, epoch, avg_loss, all_preds_tag, all_labels_tag, "Val")

def train_multitask_loop(
    bert_model,
    synergy_model,
    tag_model,
    tag_projector_model,
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
    calc_metrics=False,
    scheduler=None,
):
    bert_model.train()
    synergy_model.train()
    tag_model.train()
    total_synergy_loss = 0.0
    total_tag_loss = 0.0
    if calc_metrics:
        all_preds_synergy, all_labels_synergy = [], []
        all_preds_tag, all_labels_tag = [], []

    scaler = GradScaler()
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Train")):
        labels_synergy = batch["label"].to(device)
        tag_hot1 = batch["tag_hot1"].to(device)
        tag_hot2 = batch["tag_hot2"].to(device)

        with autocast(device_type="cuda"):
            embed1, embed2, tags_pred1, tags_pred2, preds_tag1, preds_tag2, projected_tag_embed1, projected_tag_embed2 = get_embeddings_and_tag_preds(
                bert_model, tag_model, tag_projector_model, batch, device
            )

            tag_loss1 = loss_tag_model(tags_pred1, tag_hot1)
            tag_loss2 = loss_tag_model(tags_pred2, tag_hot2)
            tag_loss =  (tag_loss1 + tag_loss2) / 2.0
            weighted_loss_synergy, preds_synergy = compute_synergy_loss(
                synergy_model, embed1, embed2, projected_tag_embed1, projected_tag_embed2, labels_synergy, loss_synergy_model, false_positive_penalty
            )

            full_loss = weighted_loss_synergy + tag_loss_weight * tag_loss
            loss_scaled = full_loss / accumulation_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if use_empty_cache:
                torch.cuda.empty_cache()

        total_synergy_loss += weighted_loss_synergy.item()
        total_tag_loss += tag_loss.item() * tag_loss_weight

        if calc_metrics:
            update_metrics_multi(calc_metrics, all_preds_synergy, all_labels_synergy, all_preds_tag, all_labels_tag, preds_synergy, labels_synergy, preds_tag1, preds_tag2, tag_hot1, tag_hot2)

    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    avg_loss = (total_tag_loss + total_synergy_loss) / len(dataloader)
    avg_synergy_loss = total_synergy_loss / len(dataloader)
    avg_tag_loss = total_tag_loss / len(dataloader)

    log_metrics_multitask(writer, epoch, avg_loss, avg_synergy_loss, avg_tag_loss,
                all_preds_synergy if calc_metrics else [], all_labels_synergy if calc_metrics else [],
                all_preds_tag if calc_metrics else [], all_labels_tag if calc_metrics else [], "Train")

def eval_multitask_loop(
    bert_model,
    synergy_model,
    tag_model,
    tag_projector_model,
    dataloader,
    loss_synergy_model,
    loss_tag_model,
    epoch,
    writer,
    device,
    label="Val",
    false_positive_penalty=1.0,
    tag_loss_weight=1.0,
):
    bert_model.eval()
    synergy_model.eval()
    tag_model.eval()

    total_synergy_loss = 0.0
    total_tag_loss = 0.0
    all_preds_synergy, all_labels_synergy = [], []
    all_preds_tag, all_labels_tag = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{label} Eval"):
            labels_synergy = batch["label"].to(device)
            tag_hot1 = batch["tag_hot1"].to(device)
            tag_hot2 = batch["tag_hot2"].to(device)

            embed1, embed2, tags_pred1, tags_pred2, preds_tag1, preds_tag2, projected_tag_embed1, projected_tag_embed2 = get_embeddings_and_tag_preds(
                bert_model, tag_model, tag_projector_model, batch, device
            )

            tag_loss1 = loss_tag_model(tags_pred1, tag_hot1)
            tag_loss2 = loss_tag_model(tags_pred2, tag_hot2)
            tag_loss =  (tag_loss1 + tag_loss2) / 2.0
            weighted_loss_synergy, preds_synergy = compute_synergy_loss(
                synergy_model, embed1, embed2, projected_tag_embed1, projected_tag_embed2, labels_synergy, loss_synergy_model, false_positive_penalty
            )

            total_synergy_loss += weighted_loss_synergy.item()
            total_tag_loss += tag_loss.item() * tag_loss_weight

            all_preds_synergy.extend(preds_synergy.cpu().numpy())
            all_labels_synergy.extend(labels_synergy.cpu().numpy().astype(int))

            all_preds_tag.extend(preds_tag1.cpu().numpy())
            all_preds_tag.extend(preds_tag2.cpu().numpy())
            all_labels_tag.extend(tag_hot1.cpu().numpy())
            all_labels_tag.extend(tag_hot2.cpu().numpy())

    avg_loss = (total_synergy_loss + total_tag_loss) / len(dataloader)
    avg_synergy_loss = total_synergy_loss / len(dataloader)
    avg_tag_loss = total_tag_loss / len(dataloader)

    log_metrics_multitask(writer, epoch, avg_loss, avg_synergy_loss, avg_tag_loss,
                all_preds_synergy, all_labels_synergy, all_preds_tag, all_labels_tag, label)

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

def create_dataloaders_multi(config, tokenizer, index_splits):

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
            debug_dataset=True,
            tag_to_index_file=config.get("tag_to_index_file", None),
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

def print_separator():
    print("=" * 30)

def train_tag_model(config, writer, save_full_dir, bert_model, tag_model, tokenizer, device, start_epoch):

    set_color("blue")

    bert_model.unfreeze_bert()

    # --- Freeze BERT based on config ---
    freeze_epochs = config.get("freeze_bert_epochs_tag", None)
    freeze_layers = config.get("freeze_bert_layers_tag", None)

    if freeze_layers == "all":
        print(f"Freezing all BERT layers for tag model training, for {freeze_epochs} epochs.")
        bert_model.freeze_bert()
    elif isinstance(freeze_layers, int):
        print(f"Freezing the first {freeze_layers} BERT layers for tag model training, for {freeze_epochs} epochs.")
        bert_model.freeze_bert_layers(freeze_layers)

    with open(config["bulk_file"], "r") as f:
        bulk_data = json.load(f)

    splits = config.get("splits_tag", {"train": 0.4, "val": 0.1})
    print("Using splits:", splits)

    random.shuffle(bulk_data)
    train_data = bulk_data[:int(len(bulk_data) * splits["train"])]
    val_data = bulk_data[int(len(bulk_data) * splits["train"]):int(len(bulk_data) * (splits["train"] + splits["val"]))]

    train_dataset = CardDataset(
        train_data,
        tokenizer,
        max_length=config["max_length_bert_tokenizer"],
        tags_len=config.get("tag_output_dim", None),
        dataset_name="train_tag",
        tag_to_index_file=config.get("tag_to_index_file", None),
    )
    val_dataset = CardDataset(
        val_data,
        tokenizer,
        max_length=config["max_length_bert_tokenizer"],
        tags_len=config.get("tag_output_dim", None),
        dataset_name="val_tag",
        tag_to_index_file=config.get("tag_to_index_file", None),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    print(f"Using tag model with output dimension: {config.get('tag_output_dim', 271)}")
    print_separator()

    if config.get("tag_model_pos_weight", None) is not None:
        tag_model_pos_weight = torch.tensor([config["tag_model_pos_weight"]]).to(device)
    else:
        tag_counts = train_dataset.tag_counts  # shape: (tags_len,)
        total = train_dataset.total_tag_samples  # scalar

        neg_counts = total - tag_counts  # how many times each tag is not present
        tag_model_pos_weight = (neg_counts / (tag_counts + 1e-6)).to(device)  # avoid div-by-zero

    optimizer, loss_tag_fn = build_training_components_tag(
        config, bert_model, tag_model, device, tag_model_pos_weight=tag_model_pos_weight
    )

    print_separator()
    end_epoch = config.get("epochs_tag", 0)+ start_epoch
    print(f"Starting training for tag model for {config['epochs_tag']} epochs...\n")
    print(f"starting epoch: {start_epoch}, to epoch: {end_epoch}")
    if end_epoch <= start_epoch:
        print("No epochs to train for tag model, exiting.")
        return

    for epoch in tqdm(
        range(start_epoch, end_epoch),
        desc="Epochs Tag",
        initial=start_epoch,
        unit="epoch",
    ):
        # Unfreeze BERT when freeze period is over
        if freeze_epochs and epoch-start_epoch == freeze_epochs:
            print(f"Unfreezing BERT at epoch {epoch}")
            bert_model.unfreeze_bert()
            print_models_param_summary(
                [("bert_model", bert_model), ("tag_model", tag_model)],
                optimizer
            )
            print_separator()

        train_tag_loop(
            bert_model,
            tag_model,
            train_loader,
            optimizer,
            loss_tag_fn,
            epoch,
            writer,
            device,
            accumulation_steps=config["accumulation_steps"],
            use_empty_cache=config.get("use_empty_cache", False),
            calc_metrics=config.get("train_calc_metrics", False),
        )

        if epoch % config.get("save_every_n_epochs", 1) == 0:
            
            torch.save(bert_model.state_dict(), os.path.join(save_full_dir, f"bert_tag_only_epoch_{epoch + 1}.pt"))
            torch.save(tag_model.state_dict(), os.path.join(save_full_dir, f"tag_tag_only_model_epoch_{epoch + 1}.pt"))
            print(f"Saved Bert and Tag models at epoch {epoch + 1}.")

        print_separator()
        if epoch % config.get("eval_every_n_epochs", 1) == 0:
            eval_tag_loop(
                bert_model,
                tag_model,
                val_loader,
                loss_tag_fn,
                epoch,
                writer,
                device
            )
        
    print_separator()
    print("Training completed for tag model.")
    print("Saving final models...")
    torch.save(bert_model.state_dict(), os.path.join(save_full_dir, "bert_tag_only_final.pt"))
    torch.save(tag_model.state_dict(), os.path.join(save_full_dir, "tag_tag_only_final_model.pt"))
    print("Final models saved.")
    print_separator()
        
def run_training_multitask(config):
    """
    Main function to run the training of the multitask model.
    It sets up directories, loads data, initializes models, and starts training.
    """
    print_separator()
    print("Starting Multitask Training")
    print_separator()

    # Set up directories and writer
    writer, save_full_dir, start_epoch = setup_dirs_writer(config)

    #Initialize BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
     # Step 2: Initialize BERT model and tokenizer
    model_name = config["bert_model_name"]
    embedding_dim = config.get("bert_embedding_dim")
    bert_model, tokenizer, device = build_bert_model(model_name, embedding_dim)
    print(f"Using BERT model: {model_name} with embedding dimension: {embedding_dim}")

    
    tag_model = build_tag_model(
        "tagModel",
        input_dim=config.get("bert_embedding_dim"),
        output_dim=config.get("tag_output_dim"),
        hidden_dims=config.get("tag_hidden_dims"),
        dropout=config.get("tag_dropout"),
        use_batchnorm=config.get("tag_use_batchnorm"),
        use_sigmoid_output=config.get("tag_use_sigmoid_output")
    ).to(device)

    # Train the tag model if specified
    if config.get("train_tag_model", False):

        print("Training tag model...")
        print_separator()
        if config.get("bert_checkpoint_tag", None) and config["bert_checkpoint_tag"] != "":
            bert_model.load_state_dict(torch.load(config["bert_checkpoint_tag"]))
            print(f"Loaded BERT checkpoint for tag model: {config['bert_checkpoint_tag']}")
        
        if config.get("tag_checkpoint_tag", None) and config["tag_checkpoint_tag"] != "":
            tag_model.load_state_dict(torch.load(config["tag_checkpoint_tag"]))
            print(f"Loaded tag model checkpoint: {config['tag_checkpoint_tag']}")
        
        train_tag_model(config, writer, save_full_dir, bert_model, tag_model, tokenizer, device, start_epoch)
    else:

        print("Skipping tag model training as per configuration.")

    if config["bert_checkpoint_multi"] and config["bert_checkpoint_multi"] != "":
        bert_model.load_state_dict(torch.load(config["bert_checkpoint_multi"]))
        print(f"Loaded BERT checkpoint: {config['bert_checkpoint_multi']}")
    
    if config.get("tag_checkpoint_multi", None) and config["tag_checkpoint_multi"] != "":
        tag_model.load_state_dict(torch.load(config["tag_checkpoint_multi"]))
        print(f"Loaded tag model checkpoint: {config['tag_checkpoint_multi']}")
    
    train_multitask_model(
        config, writer, save_full_dir, start_epoch+config.get("epochs_tag", 0), bert_model, tokenizer, device, tag_model
    )

def train_multitask_model(config, writer, save_full_dir, start_epoch, bert_model, tokenizer, device, tag_model):
    
    
    set_color("green")

    bert_model.unfreeze_bert()  # Ensure BERT is unfrozen for multitask training

    print_separator()
    print("Starting Multitask Model Training")
    print_separator()

    # Step 1: Load and split indices
    real_indices, fake_indices = get_real_fake_indices(config["synergy_file"])

    # --- Freeze BERT based on config ---
    freeze_epochs = config.get("freeze_bert_epochs_multi", None)
    freeze_layers = config.get("freeze_bert_layers_multi", None)

    if freeze_layers == "all":
        print(f"Freezing all BERT layers for tag model training, for {freeze_epochs} epochs.")
        bert_model.freeze_bert()
    elif isinstance(freeze_layers, int):
        print(f"Freezing the first {freeze_layers} BERT layers for tag model training, for {freeze_epochs} epochs.")
        bert_model.freeze_bert_layers(freeze_layers)


    # Define split proportions
    if config.get("splits", None) is not None:
        splits = config["splits"]
    else:
        print("Using default splits")
        splits = {
            "train": {"real": 0.8, "fake": 0.1},
            "val_real": {"real": 0.1, "fake": 0.0},
            "val_real_fake": {"real": 0.1, "fake": 0.03},
        }

    split_indices_result = split_indices(
        real_indices, fake_indices, splits, log_splits=True
    )

    print_separator()

    data_loaders = create_dataloaders_multi(config, tokenizer, split_indices_result)

    print_separator()

    train_loader = data_loaders["train"]

    synergy_model = build_synergy_model(config["synergy_arch"], config["bert_embedding_dim"], config["tag_projector_output_dim"]).to(
        device
    )

    print(f"Using synergy model architecture: {config['synergy_arch']}")

    tag_projector_model = build_tag_projector_model(
        num_tags=config.get("tag_output_dim"),
        output_dim=config.get("tag_projector_output_dim"),
        hidden_dim=config.get("tag_projector_hidden_dim"),
        dropout=config.get("tag_projector_dropout"),
    ).to(device)

    print(f"Using tag projector model with output dimension: {config.get('tag_projector_output_dim')}")


    if config.get("synergy_checkpoint", None) and config["synergy_checkpoint"] != "":
        synergy_model.load_state_dict(torch.load(config["synergy_checkpoint"]))
        print(f"Loaded synergy model checkpoint: {config['synergy_checkpoint']}")

    if config.get("tag_projector_checkpoint", None) and config["tag_projector_checkpoint"] != "":
        tag_projector_model.load_state_dict(torch.load(config["tag_projector_checkpoint"]))
        print(f"Loaded tag projector model checkpoint: {config['tag_projector_checkpoint']}")

    train_dataset = train_loader.dataset
    
    if config.get("synergy_pos_weight", None) is not None:
        synergy_model_pos_weight = torch.tensor(
            [config["synergy_pos_weight"]]
        )
    else:
        synergy_1_counts = train_dataset.counts[0] + train_dataset.counts[2]
        synergy_0_counts = train_dataset.counts[1] + train_dataset.counts[3]

        synergy_model_pos_weight = torch.tensor([synergy_0_counts / (synergy_1_counts+ 1e-6)])
        print(f"Using synergy model pos weight: {synergy_model_pos_weight.item()}")


    if config.get("tag_model_pos_weight", None) is not None:
        tag_model_pos_weight = torch.tensor([config["tag_model_pos_weight"]]).to(device)
    else:
        tag_counts = train_dataset.tag_counts  # shape: (tags_len,)
        total = train_dataset.total_tag_samples  # scalar

        neg_counts = total - tag_counts  # how many times each tag is not present
        tag_model_pos_weight = (neg_counts / (tag_counts + 1e-6)).to(device)  # avoid div-by-zero


    optimizer, loss_sin_fn, loss_tag_fn = build_training_components_multitask(
        config=config,
        bert_model=bert_model,
        synergy_model=synergy_model,
        tag_model=tag_model,
        tag_projector_model=tag_projector_model,
        device=device,
        tag_model_pos_weight=tag_model_pos_weight,
        synergy_model_pos_weight=synergy_model_pos_weight
    )

    if config.get("use_scheduler", False):
        num_training_steps = len(train_loader) * (config["epochs_multi"] - start_epoch)
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        print(f"Using scheduler with {num_warmup_steps} warmup steps and {num_training_steps} total steps.")
    else:
        scheduler = None
        print("Not using scheduler.")

    print_separator()

    test_at_start_sets = config.get("test_at_start_sets", None)
    if test_at_start_sets is not None and isinstance(test_at_start_sets, list) and len(test_at_start_sets) > 0:
        print_separator()
        print("Running initial evaluation on test sets...")
        for split_name, loader in data_loaders.items():
            if split_name in test_at_start_sets:
                eval_multitask_loop(
                    bert_model=bert_model,
                    synergy_model=synergy_model,
                    tag_model=tag_model,
                    tag_projector_model=tag_projector_model,
                    dataloader=loader,
                    loss_synergy_model=loss_sin_fn,
                    loss_tag_model=loss_tag_fn,
                    epoch=-1,  # No epoch for initial eval
                    writer=writer,
                    device=device,
                    label=split_name,
                    false_positive_penalty=config.get("synergy_false_positive_penalty"),
                    tag_loss_weight=config.get("tag_loss_weight"),
                )

    end_epoch = start_epoch + config.get("epochs_multi", 0)
    print_separator()
    print(f"Starting training for multitask model {config['epochs_multi']} epochs...\n")
    print(f"starting epoch: {start_epoch}, to epoch: {end_epoch}")
    for epoch in tqdm(
        range(start_epoch, end_epoch),
        desc="Epochs",
        initial=start_epoch
    ):
        print_separator()

        if freeze_epochs and epoch-start_epoch == freeze_epochs:
            print(f"Unfreezing BERT at epoch {epoch}")
            bert_model.unfreeze_bert()
            print_models_param_summary(
                [("bert_model", bert_model), ("synergy_model", synergy_model), ("tag_model", tag_model), ("tag_projector_model", tag_projector_model)],
                optimizer
            )

        train_multitask_loop(
            bert_model=bert_model,
            synergy_model=synergy_model,
            tag_model=tag_model,
            tag_projector_model=tag_projector_model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_synergy_model=loss_sin_fn,
            loss_tag_model=loss_tag_fn,
            epoch=epoch,
            writer=writer,
            device=device,
            false_positive_penalty=config.get("synergy_false_positive_penalty"),
            tag_loss_weight=config.get("tag_loss_weight"),
            accumulation_steps=config["accumulation_steps"],
            use_empty_cache=config.get("use_empty_cache", False),
            calc_metrics=config.get("train_calc_metrics", False),
            scheduler=scheduler
        )
        
        if (epoch + 1) % config["save_every"] == 0:
            print_separator()
            bert_model_path = os.path.join(
                save_full_dir, f"bert_multi_model_epoch_{epoch + 1}.pth"
            )
            synergy_model_path = os.path.join(
                save_full_dir, f"synergy_model_epoch_{epoch + 1}.pth"
            )
            tag_model_path = os.path.join(
                save_full_dir, f"tag_multi_model_epoch_{epoch + 1}.pth"
            )
            tag_projector_model_path = os.path.join(
                save_full_dir, f"tag_projector_model_epoch_{epoch + 1}.pth"
            )

            torch.save(tag_projector_model.state_dict(), tag_projector_model_path)
            torch.save(tag_model.state_dict(), tag_model_path)
            torch.save(bert_model.state_dict(), bert_model_path)
            torch.save(synergy_model.state_dict(), synergy_model_path)
            print(f"Saved Bert, Synergy, Tag, and Tag Projector models at epoch {epoch + 1}.")

        if (epoch + 1) % config["eval_every"] == 0:
            print_separator()
            for split_name, loader in data_loaders.items():
                if split_name.startswith("val"):
                    eval_multitask_loop(
                        bert_model=bert_model,
                        synergy_model=synergy_model,
                        tag_model=tag_model,
                        tag_projector_model=tag_projector_model,
                        dataloader=loader,
                        loss_synergy_model=loss_sin_fn,
                        loss_tag_model=loss_tag_fn,
                        epoch=epoch,
                        writer=writer,
                        device=device,
                        label=split_name,
                        false_positive_penalty=config.get("synergy_false_positive_penalty"),
                        tag_loss_weight=config.get("tag_loss_weight"),
                    )

        

    # Final save
    bert_model_path = os.path.join(save_full_dir, "bert_model_final.pth")
    synergy_model_path = os.path.join(save_full_dir, "synergy_model_final.pth")
    tag_model_path = os.path.join(save_full_dir, "tag_model_final.pth")
    tag_projector_model_path = os.path.join(save_full_dir, "tag_projector_model_final.pth")
    torch.save(bert_model.state_dict(), bert_model_path)
    torch.save(synergy_model.state_dict(), synergy_model_path)
    torch.save(tag_model.state_dict(), tag_model_path)
    torch.save(tag_projector_model.state_dict(), tag_projector_model_path)
    print(f"Final models saved at {save_full_dir}")
    writer.close()

def run_all_configs(config_path):
    with open(config_path) as f:
        config_list = json.load(f)

    for config in config_list:
        config["run_name"] += time.strftime("_%Y%m%d_%H%M%S")  # Append timestamp here
        
        run_training_multitask(config)

if __name__ == "__main__":
    set_seed(1006)
    if len(sys.argv) != 2:
        print("Usage: python joint_train.py <config_file.json>")
        exit(1)

    config_file = sys.argv[1]
    run_all_configs(config_file)
