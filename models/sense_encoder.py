"""
sense_encoder.py
Stage 1: Sense-aware encoder fine-tuned on WiC data.
Produces contextual embeddings sensitive to word sense.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import numpy as np
from utils.data_loader import WiCExample


class WiCDataset(Dataset):
    def __init__(self, examples: List[WiCExample], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Encode sentence pair
        enc = self.tokenizer(
            ex.sentence1, ex.sentence2,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "label": torch.tensor(ex.label, dtype=torch.float),
            "word": ex.word,
            "sentence1": ex.sentence1,
            "sentence2": ex.sentence2
        }


class SenseEncoder(nn.Module):
    """
    BERT-based sense encoder that produces a sense-aware embedding for a word in context.
    Fine-tuned on WiC (Word-in-Context) to distinguish same vs different senses.
    """

    def __init__(self, model_name: str = "bert-base-uncased", hidden_size: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, 256),  # [u, v, |u-v|]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_word_embedding(self, sentence: str, word: str) -> torch.Tensor:
        """Get contextual embedding of a specific word in a sentence."""
        inputs = self.tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)

        # Find token positions for the target word
        tokens = self.tokenizer.tokenize(sentence)
        word_tokens = self.tokenizer.tokenize(word)

        # Find the span
        word_idx = None
        for i in range(len(tokens) - len(word_tokens) + 1):
            if tokens[i:i + len(word_tokens)] == word_tokens:
                word_idx = i + 1  # +1 for [CLS]
                break

        if word_idx is not None:
            end_idx = word_idx + len(word_tokens)
            word_emb = hidden[word_idx:end_idx].mean(dim=0)
        else:
            # Fallback: use [CLS] token
            word_emb = hidden[0]

        return word_emb

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass for WiC classification."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        # Use [CLS] token representation
        cls1 = outputs.last_hidden_state[:, 0, :]

        # Split the pair embeddings using token_type_ids
        # Simple approach: use mean of each segment
        sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[1]

        batch_size = input_ids.shape[0]
        u_list, v_list = [], []

        for b in range(batch_size):
            seps = (input_ids[b] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            if len(seps) >= 2:
                mid = seps[0].item()
                end = seps[1].item()
                u = outputs.last_hidden_state[b, 1:mid].mean(dim=0)
                v = outputs.last_hidden_state[b, mid + 1:end].mean(dim=0)
            else:
                u = cls1[b]
                v = cls1[b]
            u_list.append(u)
            v_list.append(v)

        u = torch.stack(u_list)
        v = torch.stack(v_list)
        combined = torch.cat([u, v, torch.abs(u - v)], dim=-1)
        return self.classifier(combined).squeeze(-1)


def train_sense_encoder(
    model: SenseEncoder,
    train_examples: List[WiCExample],
    dev_examples: List[WiCExample],
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: str = "cpu",
    save_path: str = "models/sense_encoder.pt"
) -> SenseEncoder:
    """Train sense encoder on WiC data."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_dataset = WiCDataset(train_examples, model.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Training sense encoder on {len(train_examples)} examples...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            preds = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((preds > 0.5).float() == labels).sum().item()

        acc = correct / len(train_examples)
        print(f"  Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f} | Train Acc: {acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved sense encoder to {save_path}")
    return model
