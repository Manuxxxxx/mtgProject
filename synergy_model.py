import json
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

# ----------------------
# Config
# ----------------------
ID_RUN = time.strftime("%Y%m%d-%H%M%S")
EMBEDDING_DIM = 384
BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 1e-4
LOG_DIR = f"runs/synergy_classifier_{ID_RUN}"
SAVE_DIR = f"checkpoints/synergy_classifier_{ID_RUN}"
SAVE_EVERY = 10
EVAL_EVERY = 5
LABEL_FILE = "edhrec_data/labeled/with_random/random_real_synergies.json"
BULK_EMBEDDING_FILE = "datasets/processed/embedding_predicted/all_commander_legal_cards20250609112722.json"
WEIGHT_FALSE_POSITIVE = 5.0  # Increase penalty for false positives

# ----------------------
# Dataset
# ----------------------
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

# ----------------------
# Model
# ----------------------
class SynergyClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, emb1, emb2):
        x = torch.cat((emb1, emb2), dim=1)
        return self.net(x)

# ----------------------
# Training Utilities
# ----------------------
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    dataset = SynergyDataset(LABEL_FILE, BULK_EMBEDDING_FILE)

    # Shuffle and split real pairs (80/20)
    real_pairs = dataset.real_pairs
    random.shuffle(real_pairs)
    split_idx = int(0.8 * len(real_pairs))
    real_train = real_pairs[:split_idx]
    real_val = real_pairs[split_idx:]

    # 100% of fake pairs
    fake_train = dataset.fake_pairs

    # Create datasets
    train_dataset = SynergyDataset(LABEL_FILE, BULK_EMBEDDING_FILE)
    train_dataset.set_data(fake_train + real_train)
    val_dataset = SynergyDataset(LABEL_FILE, BULK_EMBEDDING_FILE)
    val_dataset.set_data(real_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Now continue with your training loop using train_loader and val_loader...


    print(f"Dataset size: {len(dataset)}")
    print(f"Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    print(f"Real pairs: {len(real_pairs)}, Fake pairs: {len(dataset.fake_pairs)}")
    print(f"Batch size: {BATCH_SIZE}, Embedding dim: {EMBEDDING_DIM}")


    #number of positive labels in train_dataset
    num_pos = sum(1 for _, _, label in train_dataset if label.item() == 1)
    num_neg = sum(1 for _, _, label in train_dataset if label.item() == 0)
    print(f"TRAIN - Number of positive labels: {num_pos}, Number of negative labels: {num_neg}")

    # pos_weight = torch.tensor([num_neg / num_pos])
    pos_weight = torch.tensor([1.0])  # Use 1.0 for balanced loss, or adjust as needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SynergyClassifier(EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(EPOCHS):
        number_preds_ones = 0
        number_preds_zeros = 0
        model.train()
        total_loss = 0
        total_weighted_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        correct_training = 0
        total_training = 0
        for emb1, emb2, label in loop:
            emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(emb1, emb2)
            loss = loss_fn(logits, label)
            # Increase penalty for false positives: label == 0
            weights = torch.where(label == 0, torch.tensor(WEIGHT_FALSE_POSITIVE).to(device), torch.tensor(1.0).to(device))
            weighted_loss = (loss * weights).mean()

            weighted_loss.backward()
            optimizer.step()

            total_loss += loss.mean().item()
            total_weighted_loss += weighted_loss.item()
            loop.set_postfix(loss=loss.mean().item(), weighted_loss=weighted_loss.item())
            if (epoch + 1) % EVAL_EVERY == 0:
                correct_training += (torch.sigmoid(logits) > 0.5).float().eq(label).sum().item()
                total_training += label.size(0)

        avg_train_loss = total_loss / len(train_loader)
        avg_weighted_loss = total_weighted_loss / len(train_loader)
        accuracy_training = correct_training / total_training if total_training > 0 else 0
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        writer.add_scalar("Train/WeightedLoss", avg_weighted_loss, epoch)
        

        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            val_loss = 0
            weighted_val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for emb1, emb2, label in tqdm(val_loader, desc=f"Epoch {epoch+1} Eval"):
                    emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device)
                    logits = model(emb1, emb2)
                    loss = loss_fn(logits, label)
                    val_loss += loss.mean().item()

                    weights = torch.where(label == 0, torch.tensor(WEIGHT_FALSE_POSITIVE).to(device), torch.tensor(1.0).to(device))
                    weighted_loss = (loss * weights).mean()
                    weighted_val_loss += weighted_loss.item()

                    preds = (torch.sigmoid(logits) > 0.5).float()
                    number_preds_ones += (preds == 1).sum().item()
                    number_preds_zeros += (preds == 0).sum().item()

                    correct += (preds == label).sum().item()
                    total += label.size(0)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_weighted_loss = weighted_val_loss / len(val_loader)
            accuracy_val = correct / total
            writer.add_scalar("Eval/Loss", avg_val_loss, epoch)
            writer.add_scalar("Eval/Accuracy", accuracy_val, epoch)
            print(f"Validation Loss: {avg_val_loss:.4f}, Weighted Loss: {avg_val_weighted_loss:.4f}, Accuracy: {accuracy_val:.4f}")
            print(f"Training Loss: {avg_train_loss:.4f}, Weighted Loss: {avg_weighted_loss:.4f}, Accuracy: {accuracy_training:.4f}")
            print(f"Validation Number of predictions - Ones: {number_preds_ones}, Zeros: {number_preds_zeros}")
            print(f"Training Number of predictions - Ones: {correct_training}, Zeros: {total_training - correct_training}")
            

        if (epoch + 1) % SAVE_EVERY == 0:
            path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), path)
            print(f"Saved checkpoint: {path}")

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    train()
