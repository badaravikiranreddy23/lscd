"""
definition_gen.py
FIX FOR FLAW 4: Interpretable definition generation.

Given usage clusters for a word across two periods,
generate natural language definitions for each sense cluster
to explain HOW the word's meaning changed.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.cluster import KMeans


class DefinitionGenerator:
    """
    Uses a pre-trained generative model to produce definitions
    for each sense cluster of a target word.

    Model: 'google/flan-t5-base' (lightweight, no API needed)
    """

    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cpu"):
        self.device = device
        print(f"Loading definition generator: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        print("Definition generator loaded.")

    def generate_definition(self, word: str, example_sentences: List[str]) -> str:
        """
        Generate a definition for a word given a few example sentences.

        Prompt format:
        'Define the word "cell" as used in these sentences:
         1. The prisoner sat in his cell.
         2. The guard locked the cell door.
         Definition:'
        """
        examples = "\n".join(
            [f"{i + 1}. {s}" for i, s in enumerate(example_sentences[:3])]
        )
        prompt = (
            f'Define the word "{word}" as used in the following sentences:\n'
            f'{examples}\n'
            f'Definition:'
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=4,
                early_stopping=True
            )

        definition = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return definition.strip()


def cluster_usages(
    sentences: List[str],
    embeddings: np.ndarray,
    n_clusters: int = 3
) -> Dict[int, List[str]]:
    """
    Cluster usage sentences by their embeddings.
    Returns dict: {cluster_id: [sentence1, sentence2, ...]}
    """
    k = min(n_clusters, len(sentences))
    if k < 2:
        return {0: sentences}

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        label = int(label)  # Convert NumPy int32 to Python int for JSON serialization
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[i])

    return clusters


def generate_change_explanation(
    def_gen: DefinitionGenerator,
    word: str,
    sentences_c1: List[str],
    embs_c1: np.ndarray,
    sentences_c2: List[str],
    embs_c2: np.ndarray,
    graded_score: float,
    n_clusters: int = 3
) -> Dict:
    """
    Main interpretability function.

    1. Cluster usages within each period
    2. Generate definitions for each cluster
    3. Summarize how the meaning changed

    Returns a structured explanation dict.
    """
    # Cluster old and new usages separately
    clusters_c1 = cluster_usages(sentences_c1, embs_c1, n_clusters)
    clusters_c2 = cluster_usages(sentences_c2, embs_c2, n_clusters)

    # Generate definitions for each cluster
    old_defs = {}
    for cid, sents in clusters_c1.items():
        defn = def_gen.generate_definition(word, sents)
        old_defs[cid] = {"definition": defn, "example_sentences": sents[:2]}

    new_defs = {}
    for cid, sents in clusters_c2.items():
        defn = def_gen.generate_definition(word, sents)
        new_defs[cid] = {"definition": defn, "example_sentences": sents[:2]}

    # Summarize the change
    change_label = (
        "HIGH CHANGE" if graded_score > 0.6 else
        "MODERATE CHANGE" if graded_score > 0.3 else
        "LOW/NO CHANGE"
    )

    explanation = {
        "word": word,
        "graded_score": round(graded_score, 4),
        "change_level": change_label,
        "old_period_senses": old_defs,
        "new_period_senses": new_defs,
    }

    return explanation


def print_explanation(explanation: Dict) -> None:
    """Pretty print the change explanation."""
    print("=" * 60)
    print(f"WORD: '{explanation['word']}'")
    print(f"Graded Change Score: {explanation['graded_score']} ({explanation['change_level']})")
    print("=" * 60)

    print("\n📖 OLD PERIOD SENSES:")
    for cid, info in explanation["old_period_senses"].items():
        print(f"  Sense {cid + 1}: {info['definition']}")
        for ex in info["example_sentences"]:
            print(f"    → \"{ex[:80]}...\"" if len(ex) > 80 else f"    → \"{ex}\"")

    print("\n📖 NEW PERIOD SENSES:")
    for cid, info in explanation["new_period_senses"].items():
        print(f"  Sense {cid + 1}: {info['definition']}")
        for ex in info["example_sentences"]:
            print(f"    → \"{ex[:80]}...\"" if len(ex) > 80 else f"    → \"{ex}\"")

    print("=" * 60)
