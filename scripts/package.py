#!/usr/bin/env python3
"""Bundle portable source, tests, printable parts and previews (without weights/data)."""
from pathlib import Path
import argparse
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Source-only upgrade; excludes data, models, environment and hardware")
    parser.add_argument("--vision", action="store_true", help="Source-only GPT vision upgrade")
    parser.add_argument("--docker", action="store_true", help="Source-only Docker and Git workflow upgrade")
    args = parser.parse_args()
    source_only = args.update or args.vision or args.docker
    filename = "hotpot-docker-update.zip" if args.docker else "hotpot-gpt-update.zip" if args.vision else "hotpot-update.zip" if args.update else "hotpot-prototype.zip"
    output = ROOT / "artifacts" / filename
    output.parent.mkdir(exist_ok=True)
    files = [ROOT / name for name in (
        "README.md", "requirements.txt", "requirements-dev.txt", "run.py", ".gitignore",
    )]
    files.extend(ROOT / name for name in (
        "requirements-vision.txt", "requirements-vision.lock", "Dockerfile", "compose.yaml",
        ".dockerignore", ".env.example",
    ) if (ROOT / name).exists())
    if not source_only:
        files.extend(ROOT / name for name in ("data/.gitkeep", "models/.gitkeep"))
    directories = ["hotpot", "scripts", "static", "tests", "training_data", ".github"]
    if not source_only:
        directories.append("hardware")
    for directory in directories:
        files.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    if not source_only:
        files.extend((ROOT / "models").glob("*.json"))
    previews = ("verification.json",) if source_only else ("hotpot-lab.png", "hotpot-lab-mobile.png", "verification.json")
    for name in previews:
        path = ROOT / "artifacts" / name
        if path.exists():
            files.append(path)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(set(files)):
            bundle.write(path, "HOTPOT/" + path.relative_to(ROOT).as_posix())
    print(f"Created {output} ({output.stat().st_size / 1024:.0f} KiB, {len(files)} files)")
    print("Model weights and collected images are excluded; see README for preparation.")


if __name__ == "__main__":
    main()
