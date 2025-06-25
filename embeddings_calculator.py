import os
import json
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import bert_parsing
import conf
from bert_emb_tags import BertEmbedRegressor, initialize_bert_model


MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_DIM = 384  # Should match the dimension of tag embeddings
MAX_LEN = 256
CHECKPOINT_FILE="checkpoints/joint_training_tag/complexSin_distilbert_tag_AdamW_20250622_175623/bert_model_epoch_9.pth"



def get_embedding_from_card(card, model, tokenizer, device):
    if card is not None:
        text = bert_parsing.format_card_for_bert(card)
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            return outputs.cpu().numpy()
    else:
        return None
    
def minify_emb_arrays(json_str):
    """
    Flattens any "emb": [[...]] arrays into one line, preserving the rest of the formatting.
    Assumes emb is always a 2D array with shape (1, 384).
    """
    # Match "emb": followed by exactly two brackets with numbers inside (with optional whitespace)
    pattern = re.compile(
        r'("emb": )\[\s*\[\s*((?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,?\s*){10,})\s*\]\s*\]',
        re.DOTALL
    )

    def replacer(match):
        key = match.group(1)
        values = match.group(2)
        # Strip all unnecessary whitespace/newlines
        compact_values = re.sub(r'\s+', '', values)
        return f'{key}[[{compact_values}]]'

    return pattern.sub(replacer, json_str)

def load_bulk_file(bulk_file):
    with open(bulk_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_embedding_file(bulk_file, save_every=2000):
    model, tokenizer, device = initialize_bert_model(MODEL_NAME, EMBEDDING_DIM)
    model.load_state_dict(torch.load(CHECKPOINT_FILE))
    model.eval()

    cards = load_bulk_file(bulk_file)
    counter_save = 0
    for card in tqdm(cards, desc="Processing cards"):
        # Check if the card already has an embedding
        if "emb" not in card:
            card_emb = get_embedding_from_card(card, model, tokenizer, device)
            card["emb"] = card_emb.tolist() if card_emb is not None else None

            counter_save += 1

            if counter_save > save_every:
                
                with open(bulk_file, "w", encoding="utf-8") as f:
                    pretty_json = json.dumps(cards, indent=4)
                    compacted = minify_emb_arrays(pretty_json)
                    f.write(compacted)
                counter_save = 0
                
        
        

    # #save

    with open(bulk_file, "w", encoding="utf-8") as f:
        pretty_json = json.dumps(cards, indent=2)
        compacted = minify_emb_arrays(pretty_json)
        f.write(compacted)

if __name__ == "__main__":
    create_embedding_file("datasets/processed/embedding_predicted/joint_tag/cards_with_tags_20250622170831_withuri.json", save_every=10000)
