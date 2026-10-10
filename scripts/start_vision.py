#!/usr/bin/env python3
"""Start GPT recognition locally; prompt privately for an API key if necessary."""

import argparse
import getpass
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotpot.credentials import CredentialsError, resolve_openai_api_key, validate_api_key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gpt-5.4", "gpt-5.4-mini"), default="gpt-5.4")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-limit", type=int, default=100,
                        help="Maximum cloud attempts during this server process (default: 100)")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.request_limit <= 10000:
        parser.error("--request-limit must be between 1 and 10000")
    try:
        import openai  # noqa: F401
        import uvicorn
    except ImportError:
        parser.exit(1, "Install GPT dependencies first: .venv/bin/python -m pip install -r requirements-vision.txt\n")
    try:
        configured_key = resolve_openai_api_key()
    except CredentialsError as error:
        parser.exit(1, f"API-key setup failed: {error}\n")
    if not configured_key:
        if not sys.stdin.isatty():
            parser.exit(1, "Run python3 scripts/configure_key.py in an interactive Terminal, or configure OPENAI_API_KEY_FILE privately.\n")
        try:
            key = validate_api_key(getpass.getpass("OpenAI API key (hidden; stored only for this process): "))
        except (EOFError, KeyboardInterrupt):
            parser.exit(1, "\nStartup cancelled.\n")
        except CredentialsError as error:
            parser.exit(1, f"API-key setup failed: {error}\n")
        os.environ["OPENAI_API_KEY"] = key
    os.environ["HOTPOT_RECOGNITION_PROVIDER"] = "openai"
    os.environ["HOTPOT_OPENAI_MODEL"] = args.model
    os.environ["HOTPOT_OPENAI_REQUEST_LIMIT"] = str(args.request_limit)
    print(f"Starting {args.model}. Camera recognition is manual by default.")
    print(f"Cloud attempt limit: {args.request_limit}; restarting the server resets this limit.")
    print(f"Open http://localhost:{args.port}. API billing is separate from ChatGPT.")
    uvicorn.run("hotpot.app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
