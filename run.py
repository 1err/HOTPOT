#!/usr/bin/env python3
"""Launch Hotpot Lab on this computer. Open http://localhost:8000."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Developer mode; restarts on source changes")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run("hotpot.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
