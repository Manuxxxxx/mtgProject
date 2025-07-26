import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import DistilBertModel, DistilBertTokenizer

# ----------------------
# Model
# ----------------------
class BertEmbedRegressor(nn.Module):
    def __init__(self, output_dim, model_name):
        super().__init__()
        if model_name == "bert-base-uncased":
            print("Using BertModel")
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        elif model_name == "distilbert-base-uncased":
            print("Using DistilBertModel")
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.model_name = model_name

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)


        if self.model_name == "distilbert-base-uncased":
            cls_emb = outputs.last_hidden_state[:, 0, :]
        elif self.model_name == "bert-base-uncased":
            cls_emb = outputs.pooler_output
        else:
            #error handling for unsupported models
            raise ValueError(f"Model {self.model_name} is not supported for this task.")
        
        return self.linear(self.dropout(self.layer_norm(cls_emb)))
    
    def freeze_bert_layers(self, freeze_until_layer):
        """
        Freeze BERT or DistilBERT layers up to (but not including) freeze_until_layer.
        Use 0-11 for bert-base-uncased

        Use 0-5 for distilbert-base-uncased
        """
        if self.model_name == "bert-base-uncased":
            # Freeze embeddings
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # Freeze encoder layers
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_until_layer:
                    for param in layer.parameters():
                        param.requires_grad = False

        elif self.model_name == "distilbert-base-uncased":
            # Freeze embeddings
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # Freeze transformer layers
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_until_layer:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")


    def unfreeze_bert_layers(self, start_layer=0):
        """
        Unfreeze BERT or DistilBERT layers starting from start_layer.
        """
        if self.model_name == "bert-base-uncased":
            for i, layer in enumerate(self.bert.encoder.layer):
                if i >= start_layer:
                    for param in layer.parameters():
                        param.requires_grad = True

        elif self.model_name == "distilbert-base-uncased":
            for i, layer in enumerate(self.bert.transformer.layer):
                if i >= start_layer:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")


    def freeze_bert(self):
        """
        Freeze all BERT or DistilBERT parameters.
        """
        for param in self.bert.parameters():
            param.requires_grad = False


    def unfreeze_bert(self):
        """
        Unfreeze all BERT or DistilBERT parameters.
        """
        for param in self.bert.parameters():
            param.requires_grad = True



def build_bert_model(model_name, embedding_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = BertEmbedRegressor(output_dim=embedding_dim, model_name=model_name).to(device)
    return model, tokenizer, device

