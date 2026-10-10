#!/usr/bin/env python3
"""Download a small licensed public-image review set, without importing or training it."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hotpot.ingredients import INGREDIENTS, SAMPLE_LABELS

DEFAULT_MANIFEST = ROOT / "training_data" / "public_sources.json"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_SOURCE_PIXELS = 60_000_000
MAX_IMAGE_SIDE = 1600
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
IMAGE_HOST = "upload.wikimedia.org"
SOURCE_HOST = "commons.wikimedia.org"
ALLOWED_LICENSES = {
    "CC0-1.0", "CC-BY-2.0", "CC-BY-4.0", "CC-BY-SA-2.0",
    "CC-BY-SA-2.5", "CC-BY-SA-3.0", "CC-BY-SA-4.0", "Public-domain",
}


def validate_url(value: str, host: str) -> str:
    parsed = urlparse(value)
    if (parsed.scheme != "https" or parsed.hostname != host or parsed.username
            or parsed.password or parsed.port not in (None, 443) or parsed.query or parsed.fragment):
        raise ValueError(f"Expected an HTTPS URL on {host}: {value}")
    return value


def load_sources(path: Path = DEFAULT_MANIFEST) -> list[dict]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Expected a version-1 public-source manifest")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("The source manifest must contain a nonempty sources list")
    seen_ids, seen_urls = set(), set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source must be an object")
        required = ("id", "label", "filename", "page_url", "image_url", "author", "license",
                    "license_url", "source_group", "description", "review_status", "review_notes")
        if any(not isinstance(source.get(key), str) or not source[key].strip() for key in required):
            raise ValueError(f"Incomplete source attribution or review metadata: {source.get('id')}")
        if not SAFE_COMPONENT.fullmatch(source["id"]) or not SAFE_COMPONENT.fullmatch(source["source_group"]):
            raise ValueError("IDs and source groups must be safe ASCII directory names of at most 64 characters")
        if source["label"] not in SAMPLE_LABELS:
            raise ValueError(f"Unknown ingredient label: {source['label']}")
        if source["license"] not in ALLOWED_LICENSES:
            raise ValueError(f"Unsupported license: {source['license']}")
        if source["review_status"] not in ("unreviewed", "accepted", "rejected"):
            raise ValueError(f"Invalid review status: {source['review_status']}")
        validate_url(source["image_url"], IMAGE_HOST)
        validate_url(source["page_url"], SOURCE_HOST)
        if not source["license_url"].startswith("https://"):
            raise ValueError("License URLs must use HTTPS")
        if source["id"] in seen_ids or source["image_url"] in seen_urls:
            raise ValueError("Duplicate source ID or image URL; do not count the same photo twice")
        seen_ids.add(source["id"])
        seen_urls.add(source["image_url"])
    return sources


def summarize(sources: list[dict]) -> dict:
    return {
        "candidate_images": len(sources),
        "counts": {label: sum(s["label"] == label for s in sources) for label in SAMPLE_LABELS},
        "source_groups_per_class": {
            label: len({s["source_group"] for s in sources if s["label"] == label})
            for label in SAMPLE_LABELS
        },
        "review_status_counts": dict(Counter(s["review_status"] for s in sources)),
        "training_ready": False,
        "note": "A small review set. No downloaded-image validation or measured food accuracy is implied.",
    }


class ImageHostRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl, IMAGE_HOST)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_image(url: str) -> bytes:
    validate_url(url, IMAGE_HOST)
    request = Request(url, headers={"User-Agent": "HotpotLab-public-image-review/1.0"})
    with build_opener(ImageHostRedirects()).open(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Source image exceeds the 20 MiB download limit")
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Source image exceeds the 20 MiB download limit")
    return payload


def normalize_image(payload: bytes) -> tuple[bytes, list[int], list[int]]:
    if not isinstance(payload, bytes) or len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Expected image bytes below the 20 MiB download limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Expected a JPEG, PNG or WebP source photograph")
                original_size = list(source.size)
                if source.width * source.height > MAX_SOURCE_PIXELS:
                    raise ValueError("Source photograph exceeds the 60 megapixel limit")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
                output = BytesIO()
                image.save(output, format="JPEG", quality=93)
                return output.getvalue(), original_size, list(image.size)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError(f"Source is not a decodable photograph: {error}") from error


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_review_files(output_dir: Path, records: dict[str, dict]) -> None:
    ordered = list(records.values())
    atomic_write(output_dir / "provenance.jsonl", ("".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in ordered
    )).encode("utf-8"))
    cards = []
    names = {ingredient["id"]: ingredient["name"] for ingredient in INGREDIENTS}
    for record in ordered:
        title = f"{names.get(record['label'], record['label'])} — {record['id']}"
        image_tag = (f'<img loading="lazy" src="{quote(record["saved_path"], safe="/")}" alt="{html.escape(title)}">'
                     if record.get("saved_path") else "<p>Download failed.</p>")
        cards.append(f'<article><h2>{html.escape(title)}</h2>{image_tag}'
                     f'<p>{html.escape(record["review_status"])}: {html.escape(record["review_notes"])}</p>'
                     f'<p>{html.escape(record["description"])}</p><code>{html.escape(record.get("saved_path", ""))}</code>'
                     f'<p>{html.escape(record["author"])} · <a href="{html.escape(record["license_url"])}">'
                     f'{html.escape(record["license"])}</a> · <a href="{html.escape(record["page_url"])}">Source</a></p></article>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Hotpot image review</title>'
                '<style>body{font:16px system-ui;margin:24px;max-width:1400px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}'
                'article{border:1px solid #ccc;border-radius:10px;padding:16px;overflow-wrap:anywhere}img{width:100%;height:250px;object-fit:contain}h2{font-size:18px}code{font-size:12px}</style>'
                '<h1>Public photograph candidates — review before importing</h1>'
                '<p>These are source-caption candidates, not verified training examples. Review ingredient identity and composition. '
                'Copy approved images into a separate reviewed folder, preserving source-group and label folders. '
                'Keep this provenance file with every derived dataset. Nothing is imported or trained by this script.</p>'
                '<main>' + "".join(cards) + '</main></html>')
    atomic_write(output_dir / "gallery.html", document.encode("utf-8"))


def download_sources(sources: list[dict], output_dir: Path, *, fetch: Callable[[str], bytes] = fetch_image) -> dict:
    output_dir = Path(output_dir).resolve()
    records = {}
    previous = output_dir / "provenance.jsonl"
    if previous.exists():
        for line in previous.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    downloaded, skipped, failures = 0, 0, []
    for source in sources:
        relative = Path("images") / source["source_group"] / source["label"] / f"{source['id']}.jpg"
        destination = (output_dir / relative).resolve()
        if not destination.is_relative_to(output_dir):
            raise ValueError("Image destination escapes the output directory")
        existing = records.get(source["id"])
        try:
            if existing and destination.exists():
                if any(existing.get(key) != source[key] for key in ("image_url", "label", "source_group")):
                    raise ValueError("Source metadata changed; use a separate output directory")
                if hashlib.sha256(destination.read_bytes()).hexdigest() != existing.get("processed_sha256"):
                    raise ValueError("Local image was modified; it will not be overwritten")
                skipped += 1
                continue
            payload = fetch(source["image_url"])
            image_bytes, original_size, saved_size = normalize_image(payload)
            record = {
                **source, "download_status": "downloaded", "saved_path": relative.as_posix(),
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "processed_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "original_size": original_size, "saved_size": saved_size,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "changes": "EXIF orientation applied; resized to at most 1600 pixels per side; re-encoded as JPEG; metadata removed.",
            }
            atomic_write(destination, image_bytes)
            records[source["id"]] = record
            downloaded += 1
        except Exception as error:
            failures.append({"id": source["id"], "error": str(error)})
            if not existing:
                records[source["id"]] = {**source, "download_status": "failed", "error": str(error)}
        write_review_files(output_dir, records)
    write_review_files(output_dir, records)
    return {"downloaded": downloaded, "already_present": skipped, "failures": failures,
            "output": str(output_dir), "gallery": str(output_dir / "gallery.html"),
            "imported": 0, "trained": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "public-images")
    parser.add_argument("--labels", nargs="+", choices=SAMPLE_LABELS, help="Download only these ingredient IDs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count candidates without network access or file writes")
    args = parser.parse_args()
    try:
        sources = load_sources(args.manifest)
        if args.labels:
            sources = [source for source in sources if source["label"] in args.labels]
        if not sources:
            raise ValueError("No public candidates for the selected labels; capture your own background examples")
        result = summarize(sources) if args.dry_run else download_sources(sources, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Public image preparation failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("failures"):
        parser.exit(1, "Some images could not be downloaded. Check source URLs and network access; successful images were preserved.\n")


if __name__ == "__main__":
    main()
