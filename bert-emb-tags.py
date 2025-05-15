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
from tqdm import tqdm
import conf

import bert_parsing

# ----------------------
# Config
# ----------------------

ID_RUN = time.strftime("%Y%m%d-%H%M%S")
MODEL_NAME = "bert-base-uncased"
EMBEDDING_DIM = 384  # Should match the dimension of tag embeddings
MAX_LEN = 320
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 2e-5
CONFIG_SAVE_DIR = "configs/bert_embed_regression" + ID_RUN
SAVE_DIR = "checkpoints/bert_embed_regression" + ID_RUN
LOG_DIR = "runs/bert_embed_regression"+ID_RUN
EVAL_EVERY = 2
SAVE_EVERY = 4
SPLIT_DIR = conf.split_dir


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
# Training Utilities with tqdm
# ----------------------
def train_loop(model, dataloader, optimizer, loss_fn, device, writer, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} Train")
    similarities = []
    for i, batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)

        batch_similarities = []

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        for p, t in zip(outputs, targets):
            sim = cosine_similarity([p.cpu().detach().numpy()], [t.cpu().detach().numpy()])[0][0]
            similarities.append(sim)
            batch_similarities.append(sim)
        
        avg_sim_batch = np.mean(batch_similarities)
        writer.add_scalar("Train/Batch_CosineSimilarity", avg_sim_batch, epoch * len(dataloader) + i)
        writer.add_scalar("Train/Batch_Loss", loss.item(), epoch * len(dataloader) + i)
        pbar.set_postfix(loss=loss.item())

    avg_sim = np.mean(similarities)
    writer.add_scalar("Train/CosineSimilarity", avg_sim, epoch)
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Train/Epoch_Loss", avg_loss, epoch)
    return avg_loss

def eval_loop(model, dataloader, loss_fn, device, writer, epoch):
    model.eval()
    similarities = []
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Eval")
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            outputs = model(input_ids, attention_mask)
            for p, t in zip(outputs, targets):
                sim = cosine_similarity([p.cpu().detach().numpy()], [t.cpu().detach().numpy()])[0][0]
                similarities.append(sim)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    avg_sim = np.mean(similarities)
    writer.add_scalar("Eval/Loss", avg_loss, epoch)
    writer.add_scalar("Eval/CosineSimilarity", avg_sim, epoch)
    return avg_loss, avg_sim

def test_loop(model, dataloader, loss_fn, device, writer, desc):
    model.eval()
    total_loss = 0
    similarities = []
    pbar = tqdm(dataloader, desc="Testing "+desc)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            for p, t in zip(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy()):
                sim = cosine_similarity([p], [t])[0][0]
                similarities.append(sim)
            pbar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    avg_sim = np.mean(similarities)
    writer.add_scalar(f"Test/{desc}/Loss", avg_loss)
    writer.add_scalar(f"Test/{desc}/CosineSimilarity", avg_sim)
    return avg_loss, avg_sim


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

    # Load data
    train_data = load_data(os.path.join(SPLIT_DIR, "train_data.json"))
    val_data = load_data(os.path.join(SPLIT_DIR, "val_data.json"))
    test_data = load_data(os.path.join(SPLIT_DIR, "test_data.json"))

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

    test_loss, test_sim = test_loop(model, test_loader, loss_fn, device, writer, "Initial Test")
    print(f"Initial Test Loss: {test_loss:.4f}, Cosine Similarity: {test_sim:.4f}")

    for epoch in range(EPOCHS):
        train_loss = train_loop(model, train_loader, optimizer, loss_fn, device, writer, epoch)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

        if (epoch + 1) % EVAL_EVERY == 0:
            val_loss, val_sim = eval_loop(model, val_loader, loss_fn, device, writer, epoch)
            
            print(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4f}, Cosine Similarity = {val_sim:.4f}")
        
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth"))
            print(f"Model saved at epoch {epoch+1}")

    # Final test
    test_loss, test_sim = test_loop(model, test_loader, loss_fn, device, writer, "Final Test")
    print(f"Final Test Loss: {test_loss:.4f}, Cosine Similarity: {test_sim:.4f}")

    writer.close()
