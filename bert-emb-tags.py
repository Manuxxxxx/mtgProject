import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.tensorboard import SummaryWriter
import os

import bert_parsing

# ----------------------
# Config
# ----------------------

ID_RUN = time.strftime("%Y%m%d-%H%M%S")
MODEL_NAME = "bert-base-uncased"
EMBEDDING_DIM = 384  # Should match the dimension of tag embeddings
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 2e-5
CONFIG_SAVE_DIR = "configs/bert_embed_regression" + ID_RUN
SAVE_DIR = "checkpoints/bert_embed_regression" + ID_RUN
LOG_DIR = "runs/bert_embed_regression"+ID_RUN
EVAL_EVERY = 3
SAVE_EVERY = 5
DATA_EMBED_JSON = "datasets/processed/embedding/"

# ----------------------
# Dataset
# ----------------------
class CardDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        card = self.data[idx]
        text = bert_parsing.format_card_for_bert(card)
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        target = torch.tensor(card['target_embedding'], dtype=torch.float)
        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "target": target
        }

# ----------------------
# Model
# ----------------------
class BertEmbedRegressor(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.linear(self.dropout(pooled))

# ----------------------
# Training Utilities
# ----------------------
def train_loop(model, dataloader, optimizer, loss_fn, device, writer, epoch):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar("Train/Batch_Loss", loss.item(), epoch * len(dataloader) + i)

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Train/Epoch_Loss", avg_loss, epoch)
    return avg_loss

def eval_loop(model, dataloader, loss_fn, device, writer, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Eval/Loss", avg_loss, epoch)
    return avg_loss

def cosine_sim_test(model, dataloader, device):
    model.eval()
    similarities = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].cpu().numpy()
            preds = model(input_ids, attention_mask).cpu().numpy()
            for p, t in zip(preds, targets):
                sim = cosine_similarity([p], [t])[0][0]
                similarities.append(sim)
    return np.mean(similarities)

# ----------------------
# Load Data
# ----------------------
def load_data(json_path):
    with open(json_path, 'r') as f:
        cards = json.load(f)
    return [card for card in cards if 'target_embedding' in card]

# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    writer = SummaryWriter(log_dir=LOG_DIR)

    data = load_data(DATA_EMBED_JSON)
    np.random.shuffle(data)
    split1 = int(0.8 * len(data))
    split2 = int(0.9 * len(data))

    train_data, val_data, test_data = data[:split1], data[split1:split2], data[split2:]

    train_ds = CardDataset(train_data, tokenizer)
    val_ds = CardDataset(val_data, tokenizer)
    test_ds = CardDataset(test_data, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = BertEmbedRegressor(output_dim=EMBEDDING_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)
    with open(os.path.join(CONFIG_SAVE_DIR, "config.json"), "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE
        }, f, indent=4)

    for epoch in range(EPOCHS):
        train_loss = train_loop(model, train_loader, optimizer, loss_fn, device, writer, epoch)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

        if (epoch + 1) % EVAL_EVERY == 0:
            val_loss = eval_loop(model, val_loader, loss_fn, device, writer, epoch)
            print(f"           Val Loss   = {val_loss:.4f}")
        
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth"))
            print(f"Model saved at epoch {epoch+1}")

    test_sim = cosine_sim_test(model, test_loader, device)
    writer.add_scalar("Test/CosineSimilarity", test_sim, EPOCHS)
    print(f"Average Cosine Similarity on Test Set: {test_sim:.4f}")

    writer.close()