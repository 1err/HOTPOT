#!/usr/bin/env python3
"""Import labeled local/public images into the same manifest as camera captures.

Expected folders: source/beef_rolls, source/crab_sticks, ..., source/background.
Import separate acquisition sessions separately; changing a session ID does not
make adjacent frames independent. Honor the source dataset's license.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotpot.training import CLASS_IDS
from hotpot.ingredients import INGREDIENTS

CHINESE_NAMES = {item["name"]: item["id"] for item in INGREDIENTS}
CHINESE_NAMES["背景"] = "background"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def import_images(source: Path, data_dir: Path, session_id: str, *, label: str | None = None) -> dict:
    source, data_dir = Path(source).resolve(), Path(data_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"Source must be a directory: {source}")
    if not session_id.strip() or len(session_id) > 100:
        raise ValueError("Use a nonempty session ID of at most 100 characters")
    if label is not None and label not in CLASS_IDS:
        raise ValueError(f"Unknown label: {label}")
    sample_dir = data_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "samples.jsonl"
    seen = set()
    if manifest.exists():
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Existing manifest is malformed at line {number}; repair it before importing") from error
            if record.get("source_sha256"):
                seen.add(record["source_sha256"])
    files = []
    if label:
        files = [(path, label) for path in sorted(source.rglob("*")) if path.suffix.lower() in EXTENSIONS]
    else:
        for folder in sorted(source.iterdir()):
            resolved_label = CHINESE_NAMES.get(folder.name, folder.name)
            if folder.is_dir() and resolved_label in CLASS_IDS:
                files.extend((path, resolved_label) for path in sorted(folder.rglob("*")) if path.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError("No matching images. Use ingredient-ID folders or --label for a single-class directory.")

    imported = 0
    duplicates = 0
    errors = []
    counts = {}
    for path, image_label in files:
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hash in seen:
            duplicates += 1
            continue
        sample_id = uuid.uuid4().hex
        destination = sample_dir / f"{sample_id}.jpg"
        try:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image.thumbnail((1600, 1600))
                image.save(destination, "JPEG", quality=93)
            record = {
                "sample_id": sample_id, "label": image_label, "session_id": session_id,
                "path": destination.relative_to(data_dir).as_posix(),
                "created_at": datetime.now(timezone.utc).isoformat(), "source": "import",
                "source_filename": path.name, "source_sha256": source_hash,
            }
            # A complete JSONL record is appended in one write. Imports should be
            # completed before starting a training run, which snapshots the manifest.
            with manifest.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen.add(source_hash)
            imported += 1
            counts[image_label] = counts.get(image_label, 0) + 1
        except Exception as error:
            destination.unlink(missing_ok=True)
            errors.append({"path": str(path), "error": str(error)})
    return {"imported": imported, "duplicates_skipped": duplicates, "counts": counts,
            "session_id": session_id, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--session", required=True, help="Acquisition-session ID; adjacent frames must share it")
    parser.add_argument("--label", choices=CLASS_IDS, help="Treat the source directory as one class")
    args = parser.parse_args()
    try:
        result = import_images(args.source, args.data, args.session, label=args.label)
    except ValueError as error:
        parser.exit(2, f"Import failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
