#!/usr/bin/env python3
"""Reproduce a 27-image public benchmark. Default: plan only; --run is paid.

Public images are evaluation inputs, never automatically imported into training.
Source pixels and attribution are retained; filenames and labels are not sent to GPT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from datetime import datetime, timezone
import hashlib
import html
from io import BytesIO
import json
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hotpot.ingredients import INGREDIENT_IDS, INGREDIENTS, SAMPLE_LABELS
from scripts.prepare_public_images import atomic_write

DEFAULT_MANIFEST = ROOT / "training_data" / "public_benchmark.json"
MODELS = ("gpt-5.4", "gpt-5.4-mini")
IMAGE_HOSTS = frozenset({"upload.wikimedia.org", "www.zhangfeihotpot.com"})
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 60_000_000
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
NAMES = {item["id"]: item["name"] for item in INGREDIENTS}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def safe_https(value: str, *, image: bool = False) -> str:
    parsed = urlparse(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.fragment
            or (image and parsed.hostname not in IMAGE_HOSTS)):
        raise ValueError("Expected an HTTPS source URL on an allowed image host" if image
                         else "Source and license links must be HTTPS URLs")
    return value


def load_manifest(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sources = manifest.get("sources") if isinstance(manifest, dict) else manifest
    if not isinstance(sources, list) or len(sources) != 27:
        raise ValueError("Benchmark requires exactly 27 sources: three per ingredient")
    seen_ids, seen_urls = set(), set()
    required = ("id", "label", "page_url", "image_url", "description", "author", "license",
                "license_url", "review_status")
    for source in sources:
        if not isinstance(source, dict) or any(not isinstance(source.get(k), str) or not source[k].strip()
                                               for k in required):
            raise ValueError("Every source requires complete label, description, attribution, and review metadata")
        if not SAFE_ID.fullmatch(source["id"]) or source["label"] not in INGREDIENT_IDS:
            raise ValueError("Invalid source ID or ingredient label")
        if source["review_status"] not in ("unreviewed", "accepted", "rejected"):
            raise ValueError("Invalid source review status")
        safe_https(source["image_url"], image=True)
        safe_https(source["page_url"])
        safe_https(source["license_url"])
        if source["id"] in seen_ids or source["image_url"] in seen_urls:
            raise ValueError("Duplicate image ID or URL; each photograph must be distinct")
        seen_ids.add(source["id"])
        seen_urls.add(source["image_url"])
        if "crop_box" in source:
            crop = source["crop_box"]
            if (not isinstance(crop, list) or len(crop) != 4
                    or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in crop)
                    or crop[0] >= crop[2] or crop[1] >= crop[3]):
                raise ValueError("crop_box must be normalized [left, top, right, bottom] coordinates")
    if Counter(s["label"] for s in sources) != Counter({label: 3 for label in INGREDIENT_IDS}):
        raise ValueError("Benchmark requires exactly three sources for each of the nine ingredients")
    return sources


def plan(sources: list[dict], model: str) -> dict:
    return {"mode": "plan", "model": model, "images": len(sources),
            "counts": dict(Counter(s["label"] for s in sources)), "planned_api_calls": len(sources),
            "reviewed_images": sum(s["review_status"] == "accepted" for s in sources),
            "sources": sources, "results": [{"id": s["id"], "expected_label": s["label"],
                "page_url": s["page_url"], "image_url": s["image_url"], "status": "not_run",
                "predicted_label": None, "reason": None} for s in sources],
            "note": "No downloads or API requests made. --download-only stages images; "
            "--run executes paid recognition. This small public set does not establish camera accuracy."}


class SourceRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_https(newurl, image=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_image(url: str) -> bytes:
    safe_https(url, image=True)
    request = Request(url, headers={"User-Agent": "HotpotLab-public-benchmark/1.0"})
    with build_opener(SourceRedirects()).open(request, timeout=30) as response:
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Source image exceeds 20 MiB")
    return payload


def prepare_image(payload: bytes, crop_box=None) -> tuple[bytes, dict]:
    """Stage lossless RGB pixels; compute exactly the JPEG the app will send."""
    if len(payload) > MAX_BYTES:
        raise ValueError("Source image exceeds 20 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Source image exceeds 60 megapixels")
                original_size = list(source.size)
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.info.clear()
                if crop_box:
                    box = tuple(round(v * image.size[i % 2]) for i, v in enumerate(crop_box))
                    if box[0] >= box[2] or box[1] >= box[3]:
                        raise ValueError("Crop is empty at source-image resolution")
                    image = image.crop(box)
                image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                png, jpeg = BytesIO(), BytesIO()
                image.save(png, "PNG")
                image.save(jpeg, "JPEG", quality=90)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Source does not decode as a supported image") from error
    return png.getvalue(), {"original_size": original_size, "input_size": list(image.size),
                            "request_image_sha256": digest(jpeg.getvalue()),
                            "pixel_sha256": digest(image.tobytes())}


def write_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def stage_images(sources: list[dict], output: Path, *, fetch=fetch_image) -> list[dict]:
    """Validate every source and cached file before any model execution."""
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    old = json.loads(provenance_path.read_text()) if provenance_path.exists() else []
    cached = {record["id"]: record for record in old}
    records, seen_pixels = [], set()
    for source in sources:
        raw_path = output / "sources" / f"{source['id']}.bin"
        image_path = output / "images" / f"{source['id']}.png"
        existing = cached.get(source["id"])
        if existing:
            if any(existing.get(k) != source.get(k) for k in ("image_url", "crop_box")):
                raise ValueError("Cached source or crop changed; choose a new output folder")
            if (not raw_path.exists() or digest(raw_path.read_bytes()) != existing["source_sha256"]
                    or not image_path.exists() or digest(image_path.read_bytes()) != existing["processed_sha256"]):
                raise ValueError("A cached source or processed image was modified; it will not be overwritten")
            prepared, metadata = prepare_image(raw_path.read_bytes(), source.get("crop_box"))
            if digest(prepared) != existing["processed_sha256"]:
                raise ValueError("Image preprocessing differs from the cached inputs; use a new output folder")
            record = {**existing, **source, **metadata}
        else:
            payload = fetch(source["image_url"])
            prepared, metadata = prepare_image(payload, source.get("crop_box"))
            record = {**source, **metadata, "source_sha256": digest(payload),
                      "processed_sha256": digest(prepared), "source_path": raw_path.relative_to(output).as_posix(),
                      "input_path": image_path.relative_to(output).as_posix(),
                      "downloaded_at": datetime.now(timezone.utc).isoformat()}
            atomic_write(raw_path, payload)
            atomic_write(image_path, prepared)
        if record["pixel_sha256"] in seen_pixels:
            raise ValueError("Duplicate decoded image pixels; do not count the same photograph twice")
        seen_pixels.add(record["pixel_sha256"])
        records.append(record)
        cached[source["id"]] = record
        write_json(provenance_path, list(cached.values()))
    write_json(provenance_path, records)
    write_gallery(output, records)
    return records


def write_gallery(output: Path, records: list[dict], predictions: list[dict] = ()) -> None:
    rows = {row["id"]: row for row in predictions}
    cards = []
    for record in records:
        escape = html.escape
        row = rows.get(record["id"], {})
        result = "Not evaluated" if not row else (
            f"Prediction: {row.get('predicted_label')}; uncertain: {row.get('uncertain')}; "
            f"model self-assessment: {row.get('confidence_level') or 'not supplied'}; "
            f"reason: {row.get('reason') or row.get('error')}")
        cards.append(f'<article><h2>{escape(NAMES[record["label"]])}: {escape(record["id"])}</h2>'
                     f'<img src="{escape(record["input_path"], quote=True)}" alt="Benchmark input">'
                     f'<p>{escape(record["description"])}</p><p>{escape(result)}</p>'
                     f'<p><a href="{escape(record["page_url"], quote=True)}">Source</a> · '
                     f'{escape(record["author"])} · <a href="{escape(record["license_url"], quote=True)}">'
                     f'{escape(record["license"])}</a></p></article>')
    page = '<!doctype html><meta charset="utf-8"><title>Hotpot public benchmark</title>' \
           '<style>body{font:16px system-ui;max-width:1000px;margin:30px auto}article{padding:20px;border-bottom:1px solid #ddd}' \
           'img{max-width:100%;max-height:360px}</style><h1>Exact public benchmark inputs</h1>' + "".join(cards)
    atomic_write(output / "gallery.html", page.encode())


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row.get("status") == "completed"]
    attempted = [row for row in rows if row.get("status") != "not_run"]
    correct = sum(row["correct"] for row in successful)
    accepted = [row for row in successful if not row["uncertain"]]
    return {"attempted": len(attempted), "not_run": len(rows) - len(attempted),
            "completed": len(successful), "errors": len(attempted) - len(successful),
            "correct": correct, "accuracy": correct / len(successful) if successful else None,
            "accepted_accuracy": sum(row["correct"] for row in accepted) / len(accepted) if accepted else None,
            "accepted_coverage": len(accepted) / len(successful) if successful else None,
            "estimated_usd": sum(row.get("estimated_usd", 0) for row in rows),
            "mean_latency_ms": statistics.mean(row["latency_ms"] for row in successful) if successful else None,
            "per_ingredient": {label: {"completed": sum(r["expected_label"] == label for r in successful),
                                       "correct": sum(r["expected_label"] == label and r["correct"] for r in successful)}
                               for label in INGREDIENT_IDS},
            "note": "Accuracy excludes request errors and unattempted images. Public-image results do not measure live-camera accuracy. "
                    "Confidence levels are model self-assessments, not calibrated probabilities."}


def write_report(output: Path, path: Path, report: dict) -> None:
    report["summary"] = summarize(report["predictions"])
    write_json(path, report)
    lines = [f"# Public ingredient benchmark: {report['model']}", "", "```json",
             json.dumps(report["summary"], ensure_ascii=False, indent=2), "```", "",
             "| Image | Expected | Predicted | Uncertain | Self-assessment | Result | Reason | USD | Latency ms |", "|---|---|---|---|---|---|---|---|---|"]
    for row in report["predictions"]:
        reason = (row.get("reason") or row.get("error") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| [{row['id']}]({row['page_url']}) | {row['expected_label']} | {row.get('predicted_label')} | "
                     f"{row.get('uncertain')} | {row.get('confidence_level') or 'not supplied'} | "
                     f"{('correct' if row.get('correct') else 'incorrect') if row['status'] == 'completed' else row['status']} | {reason} | "
                     f"{row.get('estimated_usd', 0):.6f} | {row.get('latency_ms', '')} |")
    atomic_write(path.with_suffix(".md"), ("\n".join(lines) + "\n").encode())
    write_gallery(output, report["inputs"], report["predictions"])


def run_benchmark(records: list[dict], output: Path, model: str, *, recognizer=None,
                  rerun=False, clock=time.perf_counter) -> dict:
    """Allow one writer per output folder so simultaneous runs cannot double bill."""
    with (output / ".benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another benchmark is using this output folder; no requests made") from None
        try:
            return _run_benchmark(records, output, model, recognizer=recognizer, rerun=rerun, clock=clock)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _run_benchmark(records: list[dict], output: Path, model: str, *, recognizer=None,
                   rerun=False, clock=time.perf_counter) -> dict:
    if model not in MODELS:
        raise ValueError("Choose gpt-5.4 or gpt-5.4-mini")
    if any(record["review_status"] != "accepted" for record in records):
        raise ValueError("Visually review all 27 labels and mark accepted before --run")
    from hotpot.openai_recognition import INSTRUCTIONS, RESULT_SCHEMA
    request_config = {"instructions_sha256": digest(INSTRUCTIONS.encode()),
                      "result_schema_sha256": digest(canonical(RESULT_SCHEMA)),
                      "image_detail": "high", "max_dimension": 768, "jpeg_quality": 90,
                      "reasoning_effort": "none", "max_output_tokens": 160}
    fingerprint = digest(canonical({"inputs": records, "request_config": request_config}))
    result_path = output / f"results-{model}.json"
    report = {"schema_version": 1, "model": model, "input_fingerprint": fingerprint,
              "created_at": datetime.now(timezone.utc).isoformat(), "inputs": records,
              "request_config": request_config, "predictions": []}
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("input_fingerprint") != fingerprint:
            raise ValueError("Benchmark inputs or attribution changed; use a separate output folder")
        if rerun:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            atomic_write(output / f"results-{model}-{stamp}.json", result_path.read_bytes())
        else:
            report = existing
    completed_ids = {row["id"] for row in report["predictions"] if row["status"] != "not_run"}
    pending = [record for record in records if record["id"] not in completed_ids]
    report["predictions"] = [row for row in report["predictions"] if row["status"] != "not_run"]
    # Revalidate every frozen input before readiness checks or the first billable call.
    images = {}
    for record in records:
        payload = (output / record["input_path"]).read_bytes()
        if digest(payload) != record["processed_sha256"]:
            raise ValueError("Benchmark input changed; no requests made")
        with Image.open(BytesIO(payload)) as image:
            images[record["id"]] = image.convert("RGB")
        jpeg = BytesIO()
        images[record["id"]].save(jpeg, "JPEG", quality=90)
        if digest(jpeg.getvalue()) != record["request_image_sha256"]:
            raise ValueError("Request-image preprocessing changed; no requests made")
    if not pending:
        return report
    owned = recognizer is None
    if owned:
        from hotpot.openai_recognition import OpenAIRecognizer, OpenAIUsage
        recognizer = OpenAIRecognizer(model, usage=OpenAIUsage(len(pending)), min_interval_ms=0)
    if not recognizer.status().get("ready"):
        raise ValueError("GPT is not ready. Configure a private OPENAI_API_KEY or OPENAI_API_KEY_FILE and install vision dependencies.")
    try:
        for pending_index, record in enumerate(pending):
            row = {k: record[k] for k in ("id", "page_url", "image_url", "description", "author", "license", "license_url",
                                          "source_sha256", "processed_sha256", "request_image_sha256")}
            row.update({"expected_label": record["label"], "status": "started", "predicted_label": None,
                        "uncertain": None, "reason": None, "correct": False, "estimated_usd": 0,
                        "confidence_level": None, "confidence_basis": "model_self_assessment_uncalibrated",
                        "error": "Interrupted attempt; usage may have been charged. No automatic retry."})
            report["predictions"].append(row)
            write_report(output, result_path, report)  # Persist before sending; crashes must not repeat paid attempts.
            before = recognizer.status().get("usage", {})
            started = clock()
            try:
                result = recognizer.predict(images[record["id"]])
                predicted = result.get("predicted_id")
                if predicted not in SAMPLE_LABELS:
                    raise ValueError("Recognizer returned an invalid label")
                row.update({"status": "completed", "predicted_label": predicted,
                            "uncertain": result["uncertain"], "reason": result["reason"],
                            "confidence_level": result.get("confidence_level"),
                            "confidence_basis": result.get("confidence_basis", "model_self_assessment_uncalibrated"),
                            "correct": predicted == record["label"], "error": None})
            except Exception as error:
                from hotpot.openai_recognition import RecognitionError
                row.update({"status": "error", "error": str(error) if isinstance(error, RecognitionError)
                            else f"Recognition failed ({type(error).__name__}); no automatic retry."})
            row["latency_ms"] = round((clock() - started) * 1000, 1)
            after = recognizer.status().get("usage", {})
            row["estimated_usd"] = round(after.get("estimated_usd", 0) - before.get("estimated_usd", 0), 8)
            row["usage"] = {key: after.get(key, 0) - before.get(key, 0)
                            for key in ("attempts", "completed", "input_tokens", "output_tokens", "unreported_attempts")}
            if row["status"] == "error":
                for skipped in pending[pending_index + 1:]:
                    report["predictions"].append({**skipped, "expected_label": skipped["label"],
                        "status": "not_run", "predicted_label": None, "uncertain": None,
                        "confidence_level": None, "confidence_basis": "model_self_assessment_uncalibrated",
                        "reason": None, "correct": False, "estimated_usd": 0,
                        "error": "Batch stopped after a failed request; no automatic retry."})
            write_report(output, result_path, report)
            if row["status"] == "error":
                break
    finally:
        if owned:
            recognizer.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "public-benchmark")
    parser.add_argument("--model", choices=MODELS, default="gpt-5.4")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--download-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Explicitly repeat all paid attempts; archive prior results")
    args = parser.parse_args()
    try:
        sources = load_manifest(args.manifest)
        if args.rerun and not args.run:
            raise ValueError("--rerun requires explicit --run")
        if not args.download_only and not args.run:
            result = plan(sources, args.model)
        else:
            records = stage_images(sources, args.output)
            if args.run:
                report = run_benchmark(records, args.output, args.model, rerun=args.rerun)
                result = {"mode": "executed", "model": args.model, "summary": report["summary"],
                          "report": str(args.output / f"results-{args.model}.json"), "gallery": str(args.output / "gallery.html")}
            else:
                result = {"mode": "downloaded", "images": len(records), "api_calls": 0,
                          "gallery": str(args.output / "gallery.html"), "provenance": str(args.output / "provenance.json")}
    except (ValueError, OSError) as error:
        parser.exit(2, f"Public benchmark failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
