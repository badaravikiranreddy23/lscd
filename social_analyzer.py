"""
social_analyzer.py
Full pipeline for social media semantic change analysis.
Combines: graded scoring + interpretable definitions on pre/post COVID Reddit corpus.

This is the MAIN script that demonstrates the novel contribution:
applying graded + interpretable LSCD to modern social media language.

Usage:
    python social_analyzer.py --word mask
    python social_analyzer.py --all
    python social_analyzer.py --word remote --output_json outputs/remote_analysis.json
"""

import argparse
import json
import os
import torch
import numpy as np
from typing import List, Dict

from models.sense_encoder import SenseEncoder
from models.graded_scorer import compute_graded_change_score
from models.definition_gen import DefinitionGenerator, generate_change_explanation, print_explanation
from data.social_corpus_builder import TARGET_WORDS, load_social_corpus


# Words we expect to show HIGH change (pandemic-affected)
HIGH_CHANGE_EXPECTED = {"mask", "variant", "lockdown", "essential", "bubble"}
# Words we expect to show LOW change (control group)
LOW_CHANGE_EXPECTED  = {"remote", "viral", "spread"}


def load_sentences(word: str, corpus_dir: str = "data/social_corpus"):
    """Load pre/post sentences from saved corpus."""
    pre_path  = os.path.join(corpus_dir, word, "pre2020.txt")
    post_path = os.path.join(corpus_dir, word, "post2020.txt")

    def read_lines(path):
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    return read_lines(pre_path), read_lines(post_path)


def analyze_word(
    word: str,
    model: SenseEncoder,
    def_gen: DefinitionGenerator,
    corpus_dir: str = "data/social_corpus",
    device: str = "cpu",
    method: str = "jsd"
) -> Dict:
    """Run full analysis pipeline for a single word."""
    print(f"\n{'='*60}")
    print(f"Analyzing: '{word}'")

    sents_pre, sents_post = load_sentences(word, corpus_dir)

    if not sents_pre or not sents_post:
        print(f"  ⚠️  No corpus data found for '{word}'. Run generate_sample_data.py first.")
        return {}

    print(f"  Pre-2020:  {len(sents_pre)} sentences")
    print(f"  Post-2020: {len(sents_post)} sentences")

    # Graded change score (Flaw 2 fix)
    score, embs_pre, embs_post = compute_graded_change_score(
        model=model,
        word=word,
        sentences_c1=sents_pre,
        sentences_c2=sents_post,
        device=device,
        method=method
    )

    # Interpretable definitions (Flaw 4 fix)
    explanation = generate_change_explanation(
        def_gen=def_gen,
        word=word,
        sentences_c1=sents_pre,
        embs_c1=embs_pre,
        sentences_c2=sents_post,
        embs_c2=embs_post,
        graded_score=score
    )

    print_explanation(explanation)
    return explanation


def run_all_words(
    model: SenseEncoder,
    def_gen: DefinitionGenerator,
    corpus_dir: str = "data/social_corpus",
    device: str = "cpu",
    method: str = "jsd",
    output_dir: str = "outputs"
) -> None:
    """Analyze all target words and produce a comparison report."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    words = list(TARGET_WORDS.keys())
    for word in words:
        result = analyze_word(word, model, def_gen, corpus_dir, device, method)
        if result:
            all_results[word] = result

    # Print comparison table
    print("\n" + "="*65)
    print("SOCIAL MEDIA SEMANTIC CHANGE — FULL RESULTS")
    print("="*65)
    print(f"{'Word':<15} {'Score':>8}  {'Level':<20}  {'Expected'}")
    print("-"*65)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get("graded_score", 0), reverse=True)
    for word, res in sorted_results:
        score = res.get("graded_score", 0)
        level = res.get("change_level", "?")
        expected = "HIGH ✓" if word in HIGH_CHANGE_EXPECTED else "LOW ✓"
        print(f"{word:<15} {score:>8.4f}  {level:<20}  {expected}")

    print("="*65)

    # Save full report
    report_path = os.path.join(output_dir, "full_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📄 Full report saved to: {report_path}")

    # Save CSV summary
    csv_path = os.path.join(output_dir, "scores_summary.csv")
    with open(csv_path, "w") as f:
        f.write("word,graded_score,change_level,expected_change\n")
        for word, res in sorted_results:
            f.write(f"{word},{res.get('graded_score',0):.4f},{res.get('change_level','?')},{'high' if word in HIGH_CHANGE_EXPECTED else 'low'}\n")
    print(f"📊 CSV summary saved to: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Social Media Semantic Change Analyzer")
    parser.add_argument("--word", type=str, default=None,
                        help="Specific word to analyze (e.g. 'mask')")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all target words")
    parser.add_argument("--corpus_dir", type=str, default="data/social_corpus")
    parser.add_argument("--model_weights", type=str, default="models/sense_encoder.pt")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--def_model", type=str, default="google/flan-t5-base")
    parser.add_argument("--method", type=str, default="jsd", choices=["jsd", "apd"])
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--use_sample_data", action="store_true",
                        help="Generate and use sample data (no Reddit API needed)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Optionally generate sample data first
    if args.use_sample_data:
        print("Generating sample data...")
        from data.generate_sample_data import generate_sample_corpus
        generate_sample_corpus(output_dir=args.corpus_dir)

    # Load models
    print("Loading sense encoder...")
    model = SenseEncoder(model_name=args.model_name)
    if os.path.exists(args.model_weights):
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
        print(f"  Loaded weights from {args.model_weights}")
    else:
        print(f"  ⚠️  No weights found at {args.model_weights}. Using pretrained BERT (run train.py first for best results).")
    model = model.to(device)
    model.eval()

    print("Loading definition generator...")
    def_gen = DefinitionGenerator(model_name=args.def_model, device=device)

    # Run analysis
    if args.all:
        run_all_words(model, def_gen, args.corpus_dir, device, args.method, args.output_dir)
    elif args.word:
        result = analyze_word(args.word, model, def_gen, args.corpus_dir, device, args.method)
        if args.output_json and result:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to {args.output_json}")
    else:
        print("Provide --word <word> or --all")
        print(f"Available words: {list(TARGET_WORDS.keys())}")


if __name__ == "__main__":
    main()
