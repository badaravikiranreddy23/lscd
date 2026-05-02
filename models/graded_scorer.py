"""
graded_scorer.py
FIX FOR FLAW 2: Graded semantic change scoring.

Instead of binary (changed/not changed), we compute a continuous
graded score using Jensen-Shannon Divergence between usage distributions
across two time periods.
"""

import torch
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from models.sense_encoder import SenseEncoder


def get_usage_embeddings(
    model: SenseEncoder,
    word: str,
    sentences: List[str],
    device: str = "cpu"
) -> np.ndarray:
    """
    Get sense-aware embeddings for all usages of a word.
    Returns array of shape (n_sentences, hidden_size).
    """
    model.eval()
    embeddings = []
    for sent in sentences:
        emb = model.get_word_embedding(sent, word)
        embeddings.append(emb.cpu().numpy())
    if not embeddings:
        return np.zeros((1, 768))
    return np.stack(embeddings)


def compute_apd(embs_c1: np.ndarray, embs_c2: np.ndarray) -> float:
    """
    Average Pairwise Distance (APD) between two usage distributions.
    Higher = more change.
    """
    sims = cosine_similarity(embs_c1, embs_c2)
    return float(1.0 - sims.mean())


def compute_jsd_graded_score(
    embs_c1: np.ndarray,
    embs_c2: np.ndarray,
    n_clusters: int = 5
) -> float:
    """
    Jensen-Shannon Divergence between sense cluster distributions.

    This is our GRADED SCORE approach (Flaw 2 fix):
    1. Cluster all usage embeddings into sense clusters
    2. Compute sense distribution in C1 and C2
    3. JSD between distributions = graded change score

    JSD is bounded in [0, 1]:
    - 0 = identical distributions (no change)
    - 1 = completely different distributions (maximum change)
    """
    all_embs = np.vstack([embs_c1, embs_c2])

    # Adaptive cluster count
    k = min(n_clusters, len(all_embs) // 2, len(embs_c1), len(embs_c2))
    k = max(k, 2)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_embs)

    labels_c1 = labels[:len(embs_c1)]
    labels_c2 = labels[len(embs_c1):]

    # Compute sense frequency distributions
    def sense_distribution(labels_arr, n_clusters):
        counts = np.bincount(labels_arr, minlength=n_clusters).astype(float)
        counts += 1e-10  # Laplace smoothing
        return counts / counts.sum()

    dist_c1 = sense_distribution(labels_c1, k)
    dist_c2 = sense_distribution(labels_c2, k)

    # Jensen-Shannon Divergence
    jsd = float(jensenshannon(dist_c1, dist_c2))
    return jsd


def compute_graded_change_score(
    model: SenseEncoder,
    word: str,
    sentences_c1: List[str],
    sentences_c2: List[str],
    device: str = "cpu",
    method: str = "jsd"
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Main entry point: compute graded semantic change score for a word.

    Args:
        model: Trained sense encoder
        word: Target word
        sentences_c1: Sentences from corpus 1 (old period)
        sentences_c2: Sentences from corpus 2 (new period)
        device: torch device
        method: 'jsd' (Jensen-Shannon) or 'apd' (Average Pairwise Distance)

    Returns:
        (score, embs_c1, embs_c2)
    """
    if not sentences_c1 or not sentences_c2:
        return 0.0, np.zeros((1, 768)), np.zeros((1, 768))

    embs_c1 = get_usage_embeddings(model, word, sentences_c1, device)
    embs_c2 = get_usage_embeddings(model, word, sentences_c2, device)

    if method == "jsd":
        score = compute_jsd_graded_score(embs_c1, embs_c2)
    elif method == "apd":
        score = compute_apd(embs_c1, embs_c2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return score, embs_c1, embs_c2


def graded_to_binary(score: float, threshold: float = 0.3) -> int:
    """Convert graded score to binary label."""
    return 1 if score >= threshold else 0
