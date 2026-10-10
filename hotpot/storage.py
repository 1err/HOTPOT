"""Explicit sample capture and configuration, independent of the ML runtime."""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import threading
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .ingredients import DEFAULT_DURATIONS, INGREDIENT_IDS, SAMPLE_LABELS


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "samples").mkdir(exist_ok=True)
        self.manifest = self.root / "samples.jsonl"
        self.config_path = self.root / "config.json"
        self.lock = threading.RLock()

    def durations(self) -> dict:
        with self.lock:
            if not self.config_path.exists():
                return dict(DEFAULT_DURATIONS)
            saved = json.loads(self.config_path.read_text(encoding="utf-8"))
            return {key: saved.get("durations", {}).get(key, DEFAULT_DURATIONS[key]) for key in INGREDIENT_IDS}

    def set_durations(self, updates: dict) -> dict:
        for key, value in updates.items():
            if key not in INGREDIENT_IDS:
                raise ValueError(f"Unknown ingredient: {key}")
            if value is not None and (type(value) is not int or not 1 <= value <= 7200):
                raise ValueError("Durations must be whole seconds between 1 and 7200, or null.")
        with self.lock:
            durations = self.durations()
            durations.update(updates)
            payload = {"durations": durations}
            self._write_json(self.config_path, payload)
            return durations

    @staticmethod
    def _write_json(path: Path, value: dict):
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def records(self) -> list[dict]:
        with self.lock:
            if not self.manifest.exists():
                return []
            return [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summary(self) -> dict:
        records = self.records()
        counts = Counter(record["label"] for record in records)
        sessions = Counter(record["session_id"] for record in records)
        return {
            "counts": {key: counts[key] for key in SAMPLE_LABELS},
            "total": len(records),
            "sessions": [{"id": key, "count": count} for key, count in sorted(sessions.items())],
        }

    def add_sample(self, image: Image.Image, label: str, session_id: str) -> dict:
        if label not in SAMPLE_LABELS:
            raise ValueError("Choose a supported ingredient or background.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", session_id):
            raise ValueError("Session names must be 1–80 letters, digits, dots, underscores or hyphens.")
        sample_id = uuid.uuid4().hex
        relative = f"samples/{sample_id}.jpg"
        record = {
            "sample_id": sample_id,
            "label": label,
            "session_id": session_id,
            "path": relative,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            # Re-encoding strips metadata and ensures all stored samples are RGB JPEGs.
            image.convert("RGB").save(self.root / relative, format="JPEG", quality=94)
            try:
                with self.manifest.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                (self.root / relative).unlink(missing_ok=True)
                raise
            return {**record, **self.summary()}

    def export(self) -> Path:
        # Single-worker app: hold the lock so the manifest and image set agree.
        with self.lock:
            records = self.records()
            fd, name = tempfile.mkstemp(prefix="hotpot-dataset-", suffix=".zip")
            os.close(fd)
            destination = Path(name)
            try:
                with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    manifest_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
                    archive.writestr("samples.jsonl", manifest_text)
                    archive.writestr("config.json", json.dumps({"durations": self.durations()}, ensure_ascii=False, indent=2))
                    for record in records:
                        path = (self.root / record["path"]).resolve()
                        if not path.is_relative_to(self.root / "samples"):
                            raise ValueError("Invalid sample path in manifest.")
                        archive.write(path, record["path"])
                return destination
            except Exception:
                destination.unlink(missing_ok=True)
                raise


def decode_image(payload: bytes) -> Image.Image:
    if not payload:
        raise ValueError("Choose an image or start the camera first.")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Image is too large; use at most 16 megapixels.")
            image.load()
            from PIL import ImageOps
            return ImageOps.exif_transpose(image).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("Image dimensions exceed the supported limit.") from error
    except (OSError, SyntaxError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error
