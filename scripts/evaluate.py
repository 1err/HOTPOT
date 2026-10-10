#!/usr/bin/env python3
"""Measure recognition on explicitly selected, independent collection sessions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotpot.recognition import Recognizer
from hotpot.training import CLASS_IDS, read_samples


def evaluate(data_dir: Path, model_dir: Path, sessions: list[str]) -> dict:
    samples = read_samples(data_dir)
    selected = [sample for sample in samples if sample["session_id"] in sessions]
    missing = set(sessions) - {sample["session_id"] for sample in selected}
    if missing:
        raise ValueError(f"No samples found for requested sessions: {sorted(missing)}")
    recognizer = Recognizer(data_dir, model_dir)
    status = recognizer.status()
    if not status["ready"]:
        raise ValueError(status["message"] + " " + "; ".join(status["warnings"]))
    if status["mode"] == "fewshot":
        raise ValueError("Example-similarity mode uses the captured dataset as references. "
                         "Prepare CLIP or train a MobileNet before evaluating this manifest.")
    metadata = status["metadata"]
    seen_sessions = set(metadata.get("train_sessions", [])) | set(metadata.get("validation_sessions", []))
    overlap = seen_sessions & set(sessions)
    if overlap:
        raise ValueError(f"Test sessions were already used for training or calibration: {sorted(overlap)}. "
                         "Capture a fresh session before evaluating.")

    confusion = [[0 for _ in CLASS_IDS] for _ in CLASS_IDS]
    predictions = []
    correct = accepted = accepted_correct = known = known_accepted = background = background_rejected = 0
    for sample in selected:
        with Image.open(sample["absolute_path"]) as image:
            result = recognizer.predict(image)
        predicted = result["scores"][0]["id"]
        actual = sample["label"]
        confusion[CLASS_IDS.index(actual)][CLASS_IDS.index(predicted)] += 1
        correct += int(predicted == actual)
        is_accepted = not result["uncertain"]
        accepted += int(is_accepted)
        accepted_correct += int(is_accepted and predicted == actual)
        if actual == "background":
            background += 1
            background_rejected += int(result["uncertain"])
        else:
            known += 1
            known_accepted += int(is_accepted)
        predictions.append({
            "sample_id": sample.get("sample_id"), "path": sample["path"], "actual": actual,
            "predicted": predicted, "accepted": is_accepted, "confidence": result["confidence"],
            "inference_ms": result["inference_ms"],
        })
    per_class = {}
    for index, label in enumerate(CLASS_IDS):
        support = sum(confusion[index])
        guessed = sum(row[index] for row in confusion)
        per_class[label] = {
            "support": support, "recall": confusion[index][index] / support if support else None,
            "precision": confusion[index][index] / guessed if guessed else None,
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "mode": status["mode"],
        "sessions": sessions, "samples": len(selected), "class_ids": list(CLASS_IDS),
        "top1_accuracy": correct / len(selected), "confusion_matrix": confusion,
        "accepted_accuracy": accepted_correct / accepted if accepted else None,
        "accepted_coverage": accepted / len(selected),
        "known_ingredient_coverage": known_accepted / known if known else None,
        "background_rejection_rate": background_rejected / background if background else None,
        "per_class": per_class, "predictions": predictions,
        "note": "Top-1 measures the highest-scoring class even when rejected. Accepted accuracy excludes "
                "uncertain predictions; inspect coverage alongside it. Test sessions were excluded from "
                "training and calibration, but independence also depends on how images were collected.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--session", required=True, nargs="+", help="Fresh collection session IDs")
    parser.add_argument("--output", type=Path, default=Path("data/evaluation_report.json"))
    args = parser.parse_args()
    try:
        report = evaluate(args.data, args.models, args.session)
    except (ValueError, ImportError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
