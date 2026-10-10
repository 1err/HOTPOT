"""Git updates preserve private state and refuse to overwrite source edits."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def command(*args, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, check=True)


@pytest.fixture
def cloned_project(tmp_path):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("Git and Bash are required for the update workflow")
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    checkout = tmp_path / "checkout"
    command("git", "init", "--bare", str(remote))
    command("git", "init", "-b", "main", str(source))
    command("git", "config", "user.name", "Workflow test", cwd=source)
    command("git", "config", "user.email", "workflow@example.invalid", cwd=source)
    (source / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/hotpot.sh", source / "scripts/hotpot.sh")
    shutil.copy(ROOT / ".gitignore", source / ".gitignore")
    (source / "version.txt").write_text("first")
    command("git", "add", ".", cwd=source)
    command("git", "commit", "-m", "Initial test source", cwd=source)
    command("git", "remote", "add", "origin", str(remote), cwd=source)
    command("git", "push", "-u", "origin", "main", cwd=source)
    command("git", "clone", "-b", "main", str(remote), str(checkout))
    (checkout / "data").mkdir()
    (checkout / "data/config.json").write_text('{"custom":true}')
    (checkout / ".secrets").mkdir(mode=0o700)
    key = checkout / ".secrets/openai_api_key"
    key.write_text("dummy-private-test-key\n")
    key.chmod(0o600)
    (source / "version.txt").write_text("second")
    command("git", "add", "version.txt", cwd=source)
    command("git", "commit", "-m", "Updated test source", cwd=source)
    command("git", "push", cwd=source)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$HOTPOT_TEST_DOCKER_LOG\"\nif [[ \"${1:-}\" == compose && \"${2:-}\" == port ]]; then printf '%s\\n' '127.0.0.1:8000'; fi\n")
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    environment = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       HOTPOT_TEST_DOCKER_LOG=str(log))
    return checkout, environment, log


def test_update_pulls_code_and_preserves_private_key_and_captured_state(cloned_project):
    checkout, environment, log = cloned_project
    completed = command("bash", "scripts/hotpot.sh", "update", cwd=checkout, env=environment)
    assert (checkout / "version.txt").read_text() == "second"
    assert (checkout / "data/config.json").read_text() == '{"custom":true}'
    assert (checkout / ".secrets/openai_api_key").read_text() == "dummy-private-test-key\n"
    assert "dummy-private-test-key" not in completed.stdout + completed.stderr
    assert "--force-recreate" in log.read_text()
    assert command("git", "status", "--porcelain", cwd=checkout).stdout == ""


def test_update_keeps_running_container_when_source_has_local_edits(cloned_project):
    checkout, environment, log = cloned_project
    (checkout / "version.txt").write_text("my uncommitted change")
    result = subprocess.run(["bash", "scripts/hotpot.sh", "update"], cwd=checkout,
                            env=environment, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Commit or stash" in result.stderr
    assert (checkout / "version.txt").read_text() == "my uncommitted change"
    assert "compose up" not in log.read_text()
