"""
predict.py
Run full pipeline on a custom word: graded score + interpretable definitions.
This is the showcase script combining BOTH fixes (Flaw 2 + Flaw 4).
"""

import argparse
import torch
import json
from models.sense_encoder import SenseEncoder
from models.graded_scorer import compute_graded_change_score
from models.definition_gen import DefinitionGenerator, generate_change_explanation, print_explanation
from utils.data_loader import load_semeval_corpora


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict graded change + interpretable definitions for a word"
    )
    parser.add_argument("--word", type=str, required=True,
                        help="Target word to analyze")
    parser.add_argument("--corpus1", type=str, default=None,
                        help="Path to old corpus text file (one sentence per line)")
    parser.add_argument("--corpus2", type=str, default=None,
                        help="Path to new corpus text file (one sentence per line)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="SemEval data dir (alternative to --corpus1/2)")
    parser.add_argument("--model_weights", type=str, default="models/sense_encoder.pt")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--def_model", type=str, default="google/flan-t5-base",
                        help="Model for definition generation")
    parser.add_argument("--method", type=str, default="jsd", choices=["jsd", "apd"])
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional: save output to JSON file")
    parser.add_argument("--max_sentences", type=int, default=80)
    return parser.parse_args()


def load_sentences_from_file(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load sentences
    if args.corpus1 and args.corpus2:
        sents_c1 = load_sentences_from_file(args.corpus1)
        sents_c2 = load_sentences_from_file(args.corpus2)
    elif args.data_dir:
        sents_c1, sents_c2 = load_semeval_corpora(args.data_dir, args.word)
    else:
        raise ValueError("Provide either --corpus1/--corpus2 or --data_dir")

    sents_c1 = sents_c1[:args.max_sentences]
    sents_c2 = sents_c2[:args.max_sentences]

    print(f"Word: '{args.word}' | C1: {len(sents_c1)} sentences | C2: {len(sents_c2)} sentences")

    # Load sense encoder
    print("Loading sense encoder...")
    model = SenseEncoder(model_name=args.model_name)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model = model.to(device)
    model.eval()

    # --- FLAW 2 FIX: Graded change score ---
    print("\nComputing graded change score...")
    score, embs_c1, embs_c2 = compute_graded_change_score(
        model=model,
        word=args.word,
        sentences_c1=sents_c1,
        sentences_c2=sents_c2,
        device=device,
        method=args.method
    )
    print(f"Graded Score ({args.method.upper()}): {score:.4f}")

    # --- FLAW 4 FIX: Interpretable definitions ---
    print("\nGenerating interpretable definitions...")
    def_gen = DefinitionGenerator(model_name=args.def_model, device=device)

    explanation = generate_change_explanation(
        def_gen=def_gen,
        word=args.word,
        sentences_c1=sents_c1,
        embs_c1=embs_c1,
        sentences_c2=sents_c2,
        embs_c2=embs_c2,
        graded_score=score
    )

    print_explanation(explanation)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(explanation, f, indent=2)
        print(f"\nOutput saved to {args.output_json}")


if __name__ == "__main__":
    main()
