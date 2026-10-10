"""Private server-key setup must not expose keys or silently overwrite them."""

import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from hotpot import credentials
from scripts import configure_key, start_vision


@pytest.fixture(autouse=True)
def private_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    key_file = tmp_path / ".secrets" / "openai_api_key"
    monkeypatch.setattr(credentials, "DEFAULT_KEY_FILE", key_file)
    monkeypatch.setattr(configure_key, "DEFAULT_KEY_FILE", key_file)
    return key_file


def test_unconfigured_default_returns_none_and_explicit_missing_file_is_a_setup_error(tmp_path):
    assert credentials.resolve_openai_api_key({}) is None
    with pytest.raises(credentials.CredentialsError, match="OPENAI_API_KEY_FILE.*missing"):
        credentials.resolve_openai_api_key({"OPENAI_API_KEY_FILE": str(tmp_path / "missing")})


def test_environment_key_takes_precedence_without_reading_invalid_file(tmp_path):
    key = credentials.resolve_openai_api_key({
        "OPENAI_API_KEY": "  environment-dummy-key  ",
        "OPENAI_API_KEY_FILE": str(tmp_path / "missing"),
    })
    assert key == "environment-dummy-key"


def test_blank_environment_falls_back_to_secret_file_without_changing_environment(tmp_path, monkeypatch):
    path = tmp_path / "key-file"
    path.write_text("project-dummy-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(path))
    assert credentials.resolve_openai_api_key() == "project-dummy-key"
    assert "OPENAI_API_KEY" not in os.environ
    assert credentials.resolve_openai_api_key({"OPENAI_API_KEY": "   "}, default_file=path) == "project-dummy-key"


@pytest.mark.parametrize("value", ["", "   ", "dummy-secret second-value", "dummy-secret\nsecond-value", "dummy\x00secret", "x" * 8193])
def test_invalid_key_values_fail_with_sanitized_errors(value):
    with pytest.raises(credentials.CredentialsError) as error:
        credentials.validate_api_key(value)
    assert "dummy-secret" not in str(error.value)
    assert "second-value" not in str(error.value)


def test_project_key_formats_are_not_restricted_to_a_specific_prefix():
    assert credentials.validate_api_key("project-key-varying-format") == "project-key-varying-format"


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo", "oversized", "invalid-utf8", "empty"])
def test_secret_reader_rejects_nonregular_or_invalid_files_without_exposing_contents(tmp_path, kind):
    path = tmp_path / "private-file"
    if kind == "symlink":
        source = tmp_path / "source"
        source.write_text("dummy-secret", encoding="utf-8")
        path.symlink_to(source)
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "oversized":
        path.write_bytes(b"dummy-secret" * 1000)
    elif kind == "invalid-utf8":
        path.write_bytes(b"dummy-secret\xff")
    else:
        path.write_text("\n", encoding="utf-8")
    with pytest.raises(credentials.CredentialsError) as error:
        credentials.resolve_openai_api_key({"OPENAI_API_KEY_FILE": str(path)})
    assert "dummy-secret" not in str(error.value)


def test_private_writer_saves_complete_file_with_secure_permissions_and_no_temporary_files(private_environment):
    saved = credentials.write_api_key("  project-dummy-secret  ")
    assert saved == private_environment
    assert saved.read_text(encoding="utf-8") == "project-dummy-secret\n"
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert stat.S_IMODE(saved.parent.stat().st_mode) == 0o700
    assert list(saved.parent.iterdir()) == [saved]
    assert credentials.resolve_openai_api_key({}) == "project-dummy-secret"


def test_existing_nonempty_key_requires_explicit_replace(private_environment):
    credentials.write_api_key("original-dummy-key")
    with pytest.raises(credentials.CredentialsError, match="--replace"):
        credentials.write_api_key("replacement-dummy-key")
    assert private_environment.read_text(encoding="utf-8") == "original-dummy-key\n"
    credentials.write_api_key("replacement-dummy-key", replace=True)
    assert private_environment.read_text(encoding="utf-8") == "replacement-dummy-key\n"


def test_existing_empty_placeholder_can_be_configured(private_environment):
    private_environment.parent.mkdir()
    private_environment.touch()
    credentials.write_api_key("project-dummy-key")
    assert credentials.resolve_openai_api_key({}) == "project-dummy-key"


def test_secret_writer_refuses_both_file_and_parent_symlinks(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    key = destination / "key"
    key.write_text("original-dummy-key", encoding="utf-8")
    symlink_file = tmp_path / "symlink-key"
    symlink_file.symlink_to(key)
    symlink_dir = tmp_path / "symlink-dir"
    symlink_dir.symlink_to(destination, target_is_directory=True)
    for path in (symlink_file, symlink_dir / "new-key"):
        with pytest.raises(credentials.CredentialsError, match="symlink"):
            credentials.write_api_key("replacement-dummy-key", path, replace=True)
    assert key.read_text(encoding="utf-8") == "original-dummy-key"
    assert not (destination / "new-key").exists()


def test_failed_atomic_save_cleans_up_temporary_file(private_environment, monkeypatch):
    monkeypatch.setattr(credentials.os, "link", lambda *_: (_ for _ in ()).throw(PermissionError("private detail")))
    with pytest.raises(credentials.CredentialsError, match="could not be saved") as error:
        credentials.write_api_key("project-dummy-key")
    assert "private detail" not in str(error.value)
    assert not private_environment.exists()
    assert not list(private_environment.parent.iterdir())


def test_configure_command_prompts_privately_and_does_not_echo_key(private_environment, monkeypatch, capsys):
    monkeypatch.setattr(configure_key.sys.stdin, "isatty", lambda: True)
    prompts = []
    monkeypatch.setattr(configure_key.getpass, "getpass", lambda prompt: prompts.append(prompt) or "private-dummy-key")
    assert configure_key.main([]) == 0
    assert "hidden" in prompts[0]
    assert private_environment.read_text(encoding="utf-8") == "private-dummy-key\n"
    output = capsys.readouterr()
    assert "private-dummy-key" not in output.out + output.err


def test_configure_command_refuses_noninteractive_input_and_does_not_prompt(monkeypatch):
    monkeypatch.setattr(configure_key.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(configure_key.getpass, "getpass", lambda *_: pytest.fail("Unexpected key prompt"))
    with pytest.raises(SystemExit) as error:
        configure_key.main([])
    assert error.value.code == 1


def test_configure_command_refuses_existing_key_before_prompt(private_environment, monkeypatch):
    credentials.write_api_key("original-dummy-key")
    monkeypatch.setattr(configure_key.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(configure_key.getpass, "getpass", lambda *_: pytest.fail("Unexpected key prompt"))
    with pytest.raises(SystemExit) as error:
        configure_key.main([])
    assert error.value.code == 1
    assert private_environment.read_text(encoding="utf-8") == "original-dummy-key\n"


def test_local_launcher_uses_saved_file_without_prompt_or_environment_key(private_environment, monkeypatch, capsys):
    credentials.write_api_key("saved-dummy-key")
    launches = []
    for name in ("HOTPOT_RECOGNITION_PROVIDER", "HOTPOT_OPENAI_MODEL", "HOTPOT_OPENAI_REQUEST_LIMIT"):
        # Register these variables for cleanup before the launcher writes them.
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *args, **kwargs: launches.append((args, kwargs))))
    monkeypatch.setattr(start_vision.getpass, "getpass", lambda *_: pytest.fail("Unexpected key prompt"))
    monkeypatch.setattr(start_vision.sys.stdin, "isatty", lambda: False)
    start_vision.main(["--model", "gpt-5.4-mini"])
    assert len(launches) == 1
    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ["HOTPOT_OPENAI_MODEL"] == "gpt-5.4-mini"
    assert "saved-dummy-key" not in capsys.readouterr().out
