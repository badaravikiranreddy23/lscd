"""
data_loader.py
Loads SemEval-2020 Task 1 corpora and WiC dataset.
"""

import os
import json
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class WordUsage:
    word: str
    sentence: str
    period: int  # 0 = corpus1 (old), 1 = corpus2 (new)


@dataclass
class WiCExample:
    word: str
    sentence1: str
    sentence2: str
    label: int  # 1 = same sense, 0 = different sense


def load_semeval_corpora(data_dir: str, target_word: str) -> Tuple[List[str], List[str]]:
    """
    Load all sentences for a target word from corpus1 and corpus2.
    Expects:
        data_dir/corpus1/lemma/  -> .txt files
        data_dir/corpus2/lemma/  -> .txt files
    """
    usages_c1, usages_c2 = [], []

    for period, folder, usages in [(1, "corpus1", usages_c1), (2, "corpus2", usages_c2)]:
        corpus_path = os.path.join(data_dir, folder, "lemma")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")
        for fname in os.listdir(corpus_path):
            if fname.endswith(".txt"):
                fpath = os.path.join(corpus_path, fname)
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if target_word.lower() in line.lower():
                            usages.append(line)

    return usages_c1, usages_c2


def load_targets(data_dir: str) -> List[str]:
    """Load target word list."""
    path = os.path.join(data_dir, "targets.txt")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_graded_labels(data_dir: str) -> Dict[str, float]:
    """Load graded change scores (gold labels for Subtask 2)."""
    path = os.path.join(data_dir, "graded.txt")
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                labels[parts[0]] = float(parts[1])
    return labels


def load_binary_labels(data_dir: str) -> Dict[str, int]:
    """Load binary change labels (gold labels for Subtask 1)."""
    path = os.path.join(data_dir, "binary.txt")
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                labels[parts[0]] = int(parts[1])
    return labels


def load_wic_dataset(wic_dir: str, split: str = "train") -> List[WiCExample]:
    """
    Load WiC dataset.
    split: 'train', 'dev', or 'test'
    Expects files: {split}/{split}.data.txt and {split}/{split}.gold.txt
    """
    data_path = os.path.join(wic_dir, split, f"{split}.data.txt")
    gold_path = os.path.join(wic_dir, split, f"{split}.gold.txt")

    examples = []
    with open(data_path, encoding="utf-8") as df, open(gold_path, encoding="utf-8") as gf:
        for data_line, gold_line in zip(df, gf):
            parts = data_line.strip().split("\t")
            if len(parts) < 5:
                continue
            word = parts[0]
            sent1 = parts[3]
            sent2 = parts[4]
            label = 1 if gold_line.strip() == "T" else 0
            examples.append(WiCExample(word=word, sentence1=sent1, sentence2=sent2, label=label))

    return examples


def build_usage_pairs(
    usages_c1: List[str],
    usages_c2: List[str],
    word: str,
    max_pairs: int = 200
) -> List[Tuple[WordUsage, WordUsage]]:
    """
    Build all cross-period usage pairs for graded scoring.
    For graded change: compare C1 usages vs C2 usages pairwise.
    """
    pairs = []
    c1_usages = [WordUsage(word=word, sentence=s, period=0) for s in usages_c1]
    c2_usages = [WordUsage(word=word, sentence=s, period=1) for s in usages_c2]

    for u1 in c1_usages:
        for u2 in c2_usages:
            pairs.append((u1, u2))

    # Subsample if too many
    if len(pairs) > max_pairs:
        random.seed(42)
        pairs = random.sample(pairs, max_pairs)

    return pairs
