#!/usr/bin/env python3
"""Save an OpenAI API key privately for Docker and local launches."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotpot.credentials import CredentialsError, DEFAULT_KEY_FILE, write_api_key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Replace a previously saved key")
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        parser.exit(1, "Run this command in an interactive Terminal for the hidden API-key prompt.\n")
    try:
        if any(part.is_symlink() for part in (DEFAULT_KEY_FILE, *DEFAULT_KEY_FILE.parents)):
            parser.exit(1, "Choose an API-key file path without symlinks.\n")
        if DEFAULT_KEY_FILE.exists():
            file_info = DEFAULT_KEY_FILE.stat()
            if not stat.S_ISREG(file_info.st_mode):
                parser.exit(1, "The API-key destination must be a regular file.\n")
            if file_info.st_size and not args.replace:
                parser.exit(1, "An API key is already saved. Use --replace only when changing it.\n")
    except OSError:
        parser.exit(1, "The private API-key file could not be accessed. Check its permissions.\n")
    try:
        key = getpass.getpass("OpenAI API key (hidden; saved privately on this computer): ")
        destination = write_api_key(key, replace=args.replace)
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, "\nKey setup cancelled.\n")
    except CredentialsError as error:
        parser.exit(1, f"Key setup failed: {error}\n")
    print(f"API key saved privately to {destination} (file permissions 600).")
    print("Docker and the local launcher will read it automatically. Do not share this file or paste the key into chat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
