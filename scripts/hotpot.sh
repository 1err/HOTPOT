#!/usr/bin/env bash
# Manage the local container; dependencies stay inside Docker.
set -euo pipefail

HOTPOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HOTPOT_ROOT"
export HOTPOT_UID="${HOTPOT_UID:-$(id -u)}"
export HOTPOT_GID="${HOTPOT_GID:-$(id -g)}"

usage() {
  cat <<'HELP'
Usage: bash scripts/hotpot.sh {setup|start|update|key|status|logs|stop}
  setup   Save a private API key if needed, build, and start.
  start   Build if needed and start using the saved key.
  update  Pull this Git branch safely, rebuild, and restart.
  key     Privately enter or replace your saved API key.
  status  Show container health.
  logs    Follow server logs (Ctrl+C leaves the app running).
  stop    Stop the container; saved images and configuration remain.
HELP
}

docker_ready() {
  if ! command -v docker >/dev/null 2>&1; then
    printf '%s\n' 'Install Docker Desktop for Apple Silicon: https://docs.docker.com/desktop/setup/install/mac-install/' >&2
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    printf '%s\n' 'Docker Compose v2 is required. Update Docker Desktop.' >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    printf '%s\n' 'Start Docker Desktop, wait until its engine is running, then retry.' >&2
    exit 1
  fi
}

configure_key() {
  if ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' 'Python 3 is needed only for the private key setup helper.' >&2
    exit 1
  fi
  python3 scripts/configure_key.py "$@"
}

ensure_key() {
  if [[ ! -s .secrets/openai_api_key ]]; then
    configure_key
  fi
}

start_app() {
  ensure_key
  mkdir -p data
  docker compose up --detach --build --force-recreate --wait --wait-timeout 90
  printf '%s\n' 'Hotpot is running. Open the published localhost URL shown below:'
  docker compose port hotpot 8000
}

case "${1:-help}" in
  setup|start)
    docker_ready
    start_app
    ;;
  update)
    docker_ready
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1 || ! git rev-parse --verify HEAD >/dev/null 2>&1; then
      printf '%s\n' 'Updates require a Git clone of this project. See README for one-time setup.' >&2
      exit 1
    fi
    if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
      printf '%s\n' 'Source files have local changes. Commit or stash them before updating; your data and key are ignored by Git.' >&2
      exit 1
    fi
    git pull --ff-only
    start_app
    ;;
  key)
    configure_key --replace
    printf '%s\n' 'Key saved. If the app is already running, run the start command to recreate it with the new key.'
    ;;
  status)
    docker_ready
    docker compose ps
    ;;
  logs)
    docker_ready
    docker compose logs --follow --tail 100 hotpot
    ;;
  stop)
    docker_ready
    docker compose stop hotpot
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
