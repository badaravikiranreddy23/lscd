"""
train.py
Training pipeline: fine-tune the sense encoder on WiC data.
"""

import argparse
import torch
from models.sense_encoder import SenseEncoder, train_sense_encoder
from utils.data_loader import load_wic_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train sense encoder on WiC")
    parser.add_argument("--wic_dir", type=str, required=True,
                        help="Path to WiC dataset directory")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_path", type=str, default="models/sense_encoder.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading WiC dataset...")
    train_examples = load_wic_dataset(args.wic_dir, split="train")
    dev_examples = load_wic_dataset(args.wic_dir, split="dev")
    print(f"Train: {len(train_examples)} | Dev: {len(dev_examples)}")

    print(f"Loading model: {args.model_name}")
    model = SenseEncoder(model_name=args.model_name)

    model = train_sense_encoder(
        model=model,
        train_examples=train_examples,
        dev_examples=dev_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        save_path=args.save_path
    )
    print("Training complete!")


if __name__ == "__main__":
    main()
