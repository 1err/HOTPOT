#!/usr/bin/env python3
"""Compare local and GPT recognition on the same saved, labeled camera crops.

The default only prints a plan. Explicit --run enables API requests, which may
incur charges. No ingredient accuracy is inferred from model names or prompts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import getpass
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotpot.credentials import resolve_openai_api_key, validate_api_key
from hotpot.ingredients import SAMPLE_LABELS
from hotpot.training import read_samples


CLASS_IDS = tuple(SAMPLE_LABELS)
CLOUD_MODELS = ("gpt-5.4", "gpt-5.4-mini")
AVAILABLE_MODELS = ("local", *CLOUD_MODELS)


def parse_models(value: str) -> list[str]:
    models = [model.strip() for model in value.split(",") if model.strip()]
    if not models or len(models) != len(set(models)):
        raise ValueError("Choose distinct models, separated by commas.")
    unknown = set(models) - set(AVAILABLE_MODELS)
    if unknown:
        raise ValueError(f"Unsupported models: {', '.join(sorted(unknown))}")
    return models


def select_samples(data_dir: Path, sessions: list[str], limit: int) -> list[dict]:
    """Choose a reproducible, class-balanced subset without resampling images."""
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    if not sessions:
        raise ValueError("Select at least one explicitly labeled collection session.")
    records = read_samples(data_dir)
    selected = [record for record in records if record["session_id"] in sessions]
    missing = set(sessions) - {record["session_id"] for record in selected}
    if missing:
        raise ValueError(f"No samples found for sessions: {', '.join(sorted(missing))}")
    grouped = {label: [] for label in CLASS_IDS}
    for sample in sorted(selected, key=lambda row: (row["session_id"], row["path"])):
        grouped[sample["label"]].append(sample)
    balanced = []
    for index in range(max(map(len, grouped.values()), default=0)):
        for label in CLASS_IDS:
            if index < len(grouped[label]):
                balanced.append(grouped[label][index])
                if len(balanced) == limit:
                    return balanced
    return balanced


def make_plan(samples: list[dict], models: list[str], sessions: list[str]) -> dict:
    counts = Counter(sample["label"] for sample in samples)
    cloud_count = sum(model in CLOUD_MODELS for model in models)
    missing = [label for label in CLASS_IDS if not counts[label]]
    return {
        "mode": "plan", "sessions": sessions, "models": models,
        "samples": len(samples), "class_ids": list(CLASS_IDS),
        "class_counts": {label: counts[label] for label in CLASS_IDS},
        "missing_classes": missing, "all_nine_ingredients_present": not any(
            label != "background" for label in missing
        ),
        "planned_api_calls": len(samples) * cloud_count,
        "api_calls_per_model": {model: len(samples) if model in CLOUD_MODELS else 0 for model in models},
        "requires_api_key": bool(cloud_count),
        "cost_note": "Each cloud prediction is one paid request. Actual cost is computed from returned "
                     "token usage using the configured model prices; image and text tokens vary.",
        "note": "Every model receives the same captured source images. Its own image preprocessing "
                "still applies. Missing classes are untested. Add --run to execute this plan.",
    }


def prepare_images(samples: list[dict]) -> list[dict]:
    """Freeze and validate the complete input set before making a paid request."""
    prepared = []
    for sample in samples:
        try:
            image_bytes = Path(sample["absolute_path"]).read_bytes()
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError("Image exceeds the prototype's 16-megapixel limit")
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ValueError(f"Unreadable benchmark image: {sample['path']}") from error
        prepared.append({**sample, "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
                         "_image_bytes": image_bytes})
    return prepared


def validate_status(name: str, status: dict, sessions: list[str]) -> None:
    if not status.get("ready"):
        raise ValueError(f"{name} is not ready. Configure its API key/dependencies or prepare its local weights.")
    if name != "local":
        return
    if status.get("mode") == "fewshot":
        raise ValueError("Local example-similarity uses these stored images as references. "
                         "Prepare CLIP or use an independently trained classifier for this comparison.")
    metadata = status.get("metadata", {})
    seen = set(metadata.get("train_sessions", [])) | set(metadata.get("validation_sessions", []))
    if seen.intersection(sessions):
        raise ValueError("Selected sessions were used for local model training or calibration. "
                         "Capture a fresh collection session for a fair comparison.")


def build_recognizers(models: list[str], data_dir: Path, model_dir: Path,
                      sessions: list[str], sample_count: int) -> dict:
    """Preflight every requested model before either local or paid prediction."""
    cloud_count = sum(model in CLOUD_MODELS for model in models)
    if cloud_count:
        if not resolve_openai_api_key():
            raise ValueError("Configure OPENAI_API_KEY_FILE or run python3 scripts/configure_key.py before --run with GPT models.")
        if importlib.util.find_spec("openai") is None:
            raise ValueError("The OpenAI SDK is missing. Install requirements-vision.txt first.")
        from hotpot.openai_recognition import OpenAIRecognizer, OpenAIUsage
        usage = OpenAIUsage(request_limit=sample_count * cloud_count)
    recognizers = {}
    for model in models:
        if model == "local":
            from hotpot.recognition import Recognizer
            recognizer = Recognizer(data_dir, model_dir)
        else:
            # The runner spaces calls itself so waiting time is separate from
            # prediction latency and cannot appear as a cooldown classification failure.
            recognizer = OpenAIRecognizer(model, usage=usage, min_interval_ms=0)
        validate_status(model, recognizer.status(), sessions)
        recognizers[model] = recognizer
    return recognizers


def ensure_cloud_credentials(models: list[str]) -> None:
    """Prompt privately only for an explicitly requested paid execution."""
    if not any(model in CLOUD_MODELS for model in models):
        return
    if importlib.util.find_spec("openai") is None:
        raise ValueError("The OpenAI SDK is missing. Install requirements-vision.txt first.")
    if resolve_openai_api_key():
        return
    if not sys.stdin.isatty():
        raise ValueError("An OpenAI API key is missing. Run --run in an interactive Terminal "
                         "for a hidden key prompt, or configure OPENAI_API_KEY_FILE privately.")
    try:
        api_key = getpass.getpass("OpenAI API key (hidden; used only by this process): ").strip()
    except (EOFError, KeyboardInterrupt) as error:
        raise ValueError("No API key entered; no recognition requests were made.") from error
    if not api_key:
        raise ValueError("No API key entered; no recognition requests were made.")
    os.environ["OPENAI_API_KEY"] = validate_api_key(api_key)


def _prediction_id(result: dict) -> str | None:
    predicted = result.get("predicted_id")
    if predicted is None and result.get("scores"):
        predicted = result["scores"][0].get("id")
    if predicted is None:
        predicted = result.get("ingredient_id")
    # An uncertain null result alone is not evidence of a correct background
    # classification. Providers should expose the predicted class explicitly.
    if predicted is not None and predicted not in CLASS_IDS:
        raise ValueError("Recognizer returned a class outside the ingredient vocabulary")
    return predicted


def summarize(predictions: list[dict]) -> dict:
    successful = [row for row in predictions if row.get("error") is None]
    classified = [row for row in successful if row["predicted"] is not None]
    accepted = [row for row in classified if not row["uncertain"]]
    correct = sum(row["predicted"] == row["actual"] for row in classified)
    accepted_correct = sum(row["predicted"] == row["actual"] for row in accepted)
    columns = (*CLASS_IDS, "unclassified")
    confusion = [[0 for _ in columns] for _ in CLASS_IDS]
    for row in successful:
        prediction_index = columns.index(row["predicted"] or "unclassified")
        confusion[CLASS_IDS.index(row["actual"])][prediction_index] += 1
    per_class = {}
    for label in CLASS_IDS:
        rows = [row for row in successful if row["actual"] == label]
        all_rows = [row for row in predictions if row["actual"] == label]
        class_correct = sum(row["predicted"] == label for row in rows)
        per_class[label] = {
            "samples": len(all_rows), "successful_requests": len(rows),
            "errors": len(all_rows) - len(rows),
            "recall": class_correct / len(rows) if rows else None,
        }
    known = [row for row in successful if row["actual"] != "background"]
    background = [row for row in successful if row["actual"] == "background"]
    latencies = [row["latency_ms"] for row in successful]
    return {
        "samples": len(predictions), "successful_requests": len(successful),
        "errors": len(predictions) - len(successful), "classified": len(classified),
        "unclassified": len(successful) - len(classified),
        "top1_accuracy": correct / len(successful) if successful else None,
        "accepted_accuracy": accepted_correct / len(accepted) if accepted else None,
        "accepted_coverage": len(accepted) / len(successful) if successful else None,
        "known_ingredient_coverage": sum(not row["uncertain"] and row["predicted"] is not None
                                         for row in known) / len(known) if known else None,
        "background_rejection_rate": sum(row["uncertain"] for row in background) / len(background)
                                     if background else None,
        "confusion_rows": list(CLASS_IDS), "confusion_columns": list(columns),
        "confusion_matrix": confusion, "per_class": per_class,
        "unevaluated_classes": [label for label in CLASS_IDS if not per_class[label]["successful_requests"]],
        "all_nine_ingredients_evaluated": all(per_class[label]["successful_requests"] > 0
                                             for label in CLASS_IDS if label != "background"),
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "note": "API/prediction errors are excluded from accuracy and reported separately. "
                "Unclassified successful responses count as incorrect. Accepted accuracy excludes "
                "uncertain results; inspect coverage, per-class support, and error counts alongside it.",
    }


def _usage_delta(before: dict, after: dict) -> dict:
    keys = ("attempts", "completed", "input_tokens", "output_tokens", "estimated_usd", "unreported_attempts")
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys} | {
        "price_note": after.get("price_note"),
    }


def run_benchmark(samples: list[dict], recognizers: dict, sessions: list[str], *,
                  interval_seconds: float = 3.0, clock=time.monotonic, sleep=time.sleep) -> dict:
    if not 0 <= interval_seconds <= 3:
        raise ValueError("Cloud call spacing must be between 0 and 3 seconds")
    # Perform all model and image preflight checks before the first prediction.
    for name, recognizer in recognizers.items():
        validate_status(name, recognizer.status(), sessions)
    prepared = prepare_images(samples)
    report = make_plan(samples, list(recognizers), sessions)
    report.update({"mode": "executed", "created_at": datetime.now(timezone.utc).isoformat(),
                   "results": {}})
    last_cloud_started = None
    for name, recognizer in recognizers.items():
        status = recognizer.status()
        before = status.get("usage", {})
        predictions = []
        for sample in prepared:
            if name in CLOUD_MODELS and last_cloud_started is not None:
                waiting = interval_seconds - (clock() - last_cloud_started)
                if waiting > 0:
                    sleep(min(waiting, 3.0))
            if name in CLOUD_MODELS:
                last_cloud_started = clock()
            row = {"sample_id": sample.get("sample_id"), "path": sample["path"],
                   "session_id": sample["session_id"], "source_sha256": sample["source_sha256"],
                   "actual": sample["label"], "predicted": None, "uncertain": True,
                   "confidence": None, "error": None}
            started = clock()
            try:
                with Image.open(io.BytesIO(sample["_image_bytes"])) as image:
                    result = recognizer.predict(image)
                row.update({"predicted": _prediction_id(result), "uncertain": bool(result["uncertain"]),
                            "confidence": result.get("confidence"), "model": result.get("model", name)})
            except Exception as error:
                # Never print arbitrary SDK exception text: it can include URLs,
                # server echoes, or request details. The backend logs safe diagnostics.
                row["error"] = {"type": type(error).__name__, "message": "Prediction failed; excluded from accuracy."}
                code = getattr(error, "status_code", None)
                if isinstance(code, int):
                    row["error"]["status_code"] = code
            row["latency_ms"] = round((clock() - started) * 1000, 3)
            predictions.append(row)
        after = recognizer.status().get("usage", {})
        report["results"][name] = {
            "model_mode": status.get("mode"), "metrics": summarize(predictions),
            "usage": _usage_delta(before, after) if name in CLOUD_MODELS else {"estimated_usd": 0.0},
            "predictions": predictions,
        }
    report["note"] = "This report measures the selected saved camera images only. " \
                     "Missing ingredients remain untested; a small session does not establish production accuracy. " \
                     "All models received identical saved image content before their own preprocessing."
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--models", default="local,gpt-5.4", help="Comma-separated local,gpt-5.4,gpt-5.4-mini")
    parser.add_argument("--session", required=True, nargs="+", help="Fresh saved camera capture session IDs")
    parser.add_argument("--limit", type=int, default=30, help="Maximum images, balanced across present labels")
    parser.add_argument("--run", action="store_true", help="Execute recognition, including paid GPT requests")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without any inference (default)")
    parser.add_argument("--output", type=Path, help="JSON destination; execution defaults to data/benchmark_report.json")
    args = parser.parse_args(argv)
    if args.run and args.dry_run:
        parser.error("Use either --run or --dry-run")
    try:
        models = parse_models(args.models)
        samples = select_samples(args.data, args.session, args.limit)
        report = make_plan(samples, models, args.session)
        if args.run:
            # Validate all image files before constructing even a local model.
            prepare_images(samples)
            ensure_cloud_credentials(models)
            recognizers = build_recognizers(models, args.data, args.model_dir, args.session, len(samples))
            report = run_benchmark(samples, recognizers, args.session)
    except (ValueError, ImportError, OSError) as error:
        parser.exit(2, f"Comparison failed: {error}\n")
    output = args.output or (args.data / "benchmark_report.json" if args.run else None)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    display = {key: value for key, value in report.items() if key != "results"}
    if "results" in report:
        display["results"] = {
            model: {key: value for key, value in result.items() if key != "predictions"}
            for model, result in report["results"].items()
        }
    print(json.dumps(display, ensure_ascii=False, indent=2))
    if output is not None:
        print(f"Full report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
