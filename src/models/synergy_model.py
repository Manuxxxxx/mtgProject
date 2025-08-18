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
class ModelSimple(
    nn.Module
):  # Used for simple embedding concatenation with only bert, currently not used
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
            # Removed Sigmoid: use BCEWithLogitsLoss + torch.sigmoid in forward or after logits
        )

    def forward(self, embed1, embed2):
        x = torch.cat([embed1, embed2], dim=-1)
        return self.net(x)


class ModelComplex(nn.Module):
    def __init__(self, embedding_dim, tag_projector_dim):
        super().__init__()
        input_dim = (
            4 * embedding_dim + 4 * tag_projector_dim
        )  # embed1, embed2, embed1*embed2, abs(embed1-embed2), tag_projector1, tag_projector2, tag_projector1*tag_projector2, abs(tag_projector1-tag_projector2)
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
            # Removed Sigmoid here too
        )

    def forward(self, embed1, embed2, tag_projector1, tag_projector2):
        x = torch.cat(
            [
                embed1,
                embed2,
                embed1 * embed2,
                torch.abs(embed1 - embed2),
                tag_projector1,
                tag_projector2,
                tag_projector1 * tag_projector2,
                torch.abs(tag_projector1 - tag_projector2),
            ],
            dim=-1,
        )
        return self.net(x)


class ModelComplexSymmetrical(nn.Module):
    def __init__(self, embedding_dim, tag_projector_dim):
        super().__init__()
        input_dim = (
            3 * embedding_dim + 3 * tag_projector_dim
        )  # embed1, embed2, embed1*embed2, abs(embed1-embed2), tag_projector1, tag_projector2, tag_projector1*tag_projector2, abs(tag_projector1-tag_projector2)
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
            # Removed Sigmoid here too
        )

    def forward(self, embed1, embed2, tag_projector1, tag_projector2):
        x = torch.cat(
            [
                embed1 + embed2,
                embed1 * embed2,
                torch.abs(embed1 - embed2),
                tag_projector1 + tag_projector2,
                tag_projector1 * tag_projector2,
                torch.abs(tag_projector1 - tag_projector2),
            ],
            dim=-1,
        )
        return self.net(x)


class ModelComplexTagHidden(nn.Module):
    def __init__(self, embedding_dim, tag_hidden_dim):
        super().__init__()
        input_dim = 4 * embedding_dim + 4 * tag_hidden_dim
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
        )

    def forward(self, embed1, embed2, tag_hidden1, tag_hidden2):
        x = torch.cat(
            [
                embed1,
                embed2,
                embed1 * embed2,
                torch.abs(embed1 - embed2),
                tag_hidden1,
                tag_hidden2,
                tag_hidden1 * tag_hidden2,
                torch.abs(tag_hidden1 - tag_hidden2),
            ],
            dim=-1,
        )
        return self.net(x)


class ModelComplexTagHiddenSymmetrical(nn.Module):
    def __init__(self, embedding_dim, tag_hidden_dim):
        super().__init__()
        input_dim = 3 * embedding_dim + 3 * tag_hidden_dim
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
        )

    def forward(self, embed1, embed2, tag_hidden1, tag_hidden2):
        x = torch.cat(
            [
                embed1 + embed2,
                embed1 * embed2,
                torch.abs(embed1 - embed2),
                tag_hidden1 + tag_hidden2,
                tag_hidden1 * tag_hidden2,
                torch.abs(tag_hidden1 - tag_hidden2),
            ],
            dim=-1,
        )
        return self.net(x)


# ----------------------
# Weight Initialization
# ----------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ----------------------
# Model Factory
# ----------------------
def build_synergy_model(
    arch_name,
    embedding_dim,
    tag_projector_dim=None,
    initialize_weights=True,
    hidden_tag_dim=None,
):
    if arch_name == "modelSimple":
        model = ModelSimple(embedding_dim)
    elif arch_name == "modelComplex":
        if tag_projector_dim is None:
            raise ValueError("tag_projector_dim must be provided for modelComplex")
        model = ModelComplex(embedding_dim, tag_projector_dim)
    elif arch_name == "modelComplexSymmetrical":
        if tag_projector_dim is None:
            raise ValueError(
                "tag_projector_dim must be provided for modelComplexSymmetrical"
            )
        model = ModelComplexSymmetrical(embedding_dim, tag_projector_dim)
    elif arch_name == "modelComplexTagHidden":
        if hidden_tag_dim is None:
            raise ValueError(
                "hidden_tag_dim must be provided for modelComplexTagHidden"
            )
        model = ModelComplexTagHidden(embedding_dim, hidden_tag_dim)
    elif arch_name == "modelComplexTagHiddenSymmetrical":
        if hidden_tag_dim is None:
            raise ValueError(
                "hidden_tag_dim must be provided for modelComplexTagHiddenSymmetrical"
            )
        model = ModelComplexTagHiddenSymmetrical(embedding_dim, hidden_tag_dim)
    else:
        raise ValueError(f"Unknown model architecture: {arch_name}")

    if initialize_weights:
        model.apply(init_weights)

    return model


# ----------------------
# Weighted Loss Calculation Helper
# ----------------------
def calculate_synergy_weighted_FP_loss(
    logits, labels, loss_fn, false_positive_penalty=1.0
):
    """
    logits: raw output from model (no sigmoid applied yet)
    labels: ground truth labels (0 or 1)
    loss_fn: loss function (expects raw logits for BCEWithLogitsLoss)
    false_positive_penalty: multiplier applied to false positives in the batch
    """
    loss = loss_fn(logits, labels)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    # print all the predictions and labels
    # print(f"Predictions: {preds.cpu().numpy()}, Labels: {labels.cpu().numpy()}")
    # print("Logits:", logits[:10].detach().cpu().numpy())

    weights = torch.where(labels == 0, false_positive_penalty, 1.0)
    weighted_loss = (loss * weights).mean()

    return weighted_loss, preds
