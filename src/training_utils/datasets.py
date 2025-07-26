import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

from mtgProject.src.training_utils import bert_parsing

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
