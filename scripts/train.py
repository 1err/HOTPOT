#!/usr/bin/env python3
"""Fine-tune MobileNet on collected images with session-held-out validation."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotpot.training import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--unfreeze-last", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = train_model(
            args.data, args.models, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, unfreeze_last=args.unfreeze_last,
            validation_fraction=args.validation_fraction, seed=args.seed,
        )
    except (ValueError, ImportError) as error:
        parser.exit(2, f"Training could not start: {error}\n")
    print(json.dumps({k: result[k] for k in ("checkpoint", "report_path", "accuracy", "macro_recall")}, indent=2))


if __name__ == "__main__":
    main()
