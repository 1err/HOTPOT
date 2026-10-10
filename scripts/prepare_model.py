#!/usr/bin/env python3
"""Explicitly download public pretrained weights; the server never downloads them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import urllib.request


SOURCES = {
    "clip": {
        "filename": "clip_rn50_yfcc15m.pt",
        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt",
        "minimum_bytes": 300_000_000,
        "description": "OpenCLIP RN50-quickgelu, pretrained on YFCC15M; zero-shot baseline",
    },
    "mobilenet": {
        "filename": "mobilenet_v3_small_imagenet.pth",
        "url": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        "minimum_bytes": 5_000_000,
        "sha256_prefix": "047dcff4",
        "description": "Torchvision MobileNetV3-Small ImageNet weights; training initializer",
    },
}


def prepare(kind: str, model_dir: Path, *, force: bool = False) -> dict:
    spec = SOURCES[kind]
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / spec["filename"]
    if destination.exists() and not force:
        print(f"Already exists: {destination}. Use --force to download again.", flush=True)
        return {"path": str(destination), "existing": True}

    temporary = destination.with_suffix(destination.suffix + ".download")
    digest = hashlib.sha256()
    count = 0
    last_report = 0
    print(f"Downloading {spec['description']}\n{spec['url']}", flush=True)
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "HotpotLab/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            expected = int(response.headers.get("Content-Length", "0"))
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if count - last_report >= 25 * 1024 * 1024:
                    total = f" / {expected / 1024**2:.0f} MB" if expected else " MB"
                    print(f"  {count / 1024**2:.0f}{total}", flush=True)
                    last_report = count
        if count < spec["minimum_bytes"] or (expected and count != expected):
            raise ValueError(f"Incomplete or invalid weight download: received {count} bytes")
        checksum = digest.hexdigest()
        if spec.get("sha256_prefix") and not checksum.startswith(spec["sha256_prefix"]):
            raise ValueError("Downloaded weights failed the publisher's SHA256-prefix check")
        os.replace(temporary, destination)
        metadata = {
            "path": destination.name,
            "url": spec["url"],
            "bytes": count,
            "sha256": checksum,
            "description": spec["description"],
        }
        destination.with_suffix(destination.suffix + ".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Ready: {destination} ({count / 1024**2:.1f} MB)", flush=True)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--mode", choices=["clip", "mobilenet", "both"], default="clip")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for kind in (["clip", "mobilenet"] if args.mode == "both" else [args.mode]):
        prepare(kind, args.models, force=args.force)


if __name__ == "__main__":
    main()
