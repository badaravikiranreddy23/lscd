"""
evaluate.py
Evaluate on SemEval-2020 Task 1 benchmark.
Computes Spearman rho for graded scoring (Subtask 2).
"""

import argparse
import re
import torch
import numpy as np
from models.sense_encoder import SenseEncoder
from models.graded_scorer import compute_graded_change_score
from utils.data_loader import load_targets, load_graded_labels, load_binary_labels, load_semeval_corpora
from utils.evaluation import print_results


def strip_pos_tag(word: str) -> str:
    """
    Strip POS tag suffixes like _nn, _vb, _jj from SemEval word IDs.
    Examples: 'chairman_nn' -> 'chairman', 'circle_vb' -> 'circle'
    """
    return re.sub(r'_(nn|vb|jj|rb)$', '', word)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="english",
                        choices=["english", "german", "latin", "swedish"])
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to SemEval2020 language directory")
    parser.add_argument("--model_weights", type=str, default="models/sense_encoder.pt",
                        help="Path to trained sense encoder weights")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--method", type=str, default="jsd",
                        choices=["jsd", "apd"])
    parser.add_argument("--max_sentences", type=int, default=100,
                        help="Max sentences per word per corpus (for speed)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Language: {args.lang} | Method: {args.method}")

    # Load model
    model = SenseEncoder(model_name=args.model_name)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model = model.to(device)
    model.eval()

    # Load data
    targets = load_targets(args.data_dir)
    gold_graded = load_graded_labels(args.data_dir)
    gold_binary = load_binary_labels(args.data_dir)

    print(f"Evaluating {len(targets)} target words...")
    word_scores = {}
    skipped_nan = []

    for word in targets:
        try:
            # Strip POS tag for corpus search and BERT tokenization
            clean_word = strip_pos_tag(word)

            sents_c1, sents_c2 = load_semeval_corpora(args.data_dir, clean_word)
            # Also search with the original POS-tagged form (corpus uses both)
            if not sents_c1 or not sents_c2:
                sents_c1_alt, sents_c2_alt = load_semeval_corpora(args.data_dir, word)
                if not sents_c1:
                    sents_c1 = sents_c1_alt
                if not sents_c2:
                    sents_c2 = sents_c2_alt

            # Limit for speed
            sents_c1 = sents_c1[:args.max_sentences]
            sents_c2 = sents_c2[:args.max_sentences]

            if not sents_c1 or not sents_c2:
                print(f"  [SKIP] {word}: no usages found")
                skipped_nan.append(word)
                continue

            score, embs_c1, embs_c2 = compute_graded_change_score(
                model=model,
                word=clean_word,  # Use clean word for BERT
                sentences_c1=sents_c1,
                sentences_c2=sents_c2,
                device=device,
                method=args.method
            )

            # Check for NaN in score or embeddings
            if np.isnan(score) or np.any(np.isnan(embs_c1)) or np.any(np.isnan(embs_c2)):
                print(f"  [WARN] {word}: NaN detected in embeddings, skipping")
                skipped_nan.append(word)
                continue

            word_scores[word] = score
            print(f"  {word:20s} score={score:.4f}  gold={gold_graded.get(word, '?')}")

        except Exception as e:
            error_msg = str(e)
            # Shorten the verbose sklearn NaN messages
            if "NaN" in error_msg:
                print(f"  [WARN] {word}: NaN in embeddings (word not found by BERT), skipping")
            else:
                print(f"  [ERROR] {word}: {error_msg[:100]}")
            skipped_nan.append(word)

    # Summary
    print(f"\n{'='*50}")
    print(f"Evaluated: {len(word_scores)} / {len(targets)} words")
    if skipped_nan:
        print(f"Skipped ({len(skipped_nan)}): {', '.join(skipped_nan)}")
    print(f"{'='*50}")

    print_results(word_scores, gold_graded, gold_binary)


if __name__ == "__main__":
    main()
