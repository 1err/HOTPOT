"""Read a server-side API key from the environment or a private local file.

This module uses only the standard library, so the setup prompt also works
before installing the application dependencies. It never logs credential data.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping


DEFAULT_KEY_FILE = Path(__file__).resolve().parents[1] / ".secrets" / "openai_api_key"
MAX_KEY_BYTES = 8192
SETUP_HELP = (
    "Run python3 scripts/configure_key.py to save a private key, or configure "
    "OPENAI_API_KEY_FILE (OPENAI_API_KEY also works)."
)


class CredentialsError(ValueError):
    """A setup error whose message never contains the key or raw file contents."""


def validate_api_key(value: str) -> str:
    """Accept variable key formats while rejecting empty or malformed input."""
    key = value.strip()
    if not key:
        raise CredentialsError("The OpenAI API key is empty. " + SETUP_HELP)
    try:
        too_long = len(key.encode("utf-8")) > MAX_KEY_BYTES
    except UnicodeError:
        raise CredentialsError("The OpenAI API key must be valid UTF-8 text. " + SETUP_HELP) from None
    if too_long or any(character.isspace() or not character.isprintable() for character in key):
        raise CredentialsError("The OpenAI API key must be one nonempty value without whitespace. " + SETUP_HELP)
    return key


def resolve_openai_api_key(environ: Mapping[str, str] | None = None,
                           *, default_file: Path | None = None) -> str | None:
    """Use a nonempty environment key, explicit secret file, then project file.

    A missing default file means the key has not been configured. An explicit
    but unreadable file is a configuration error; it must not silently fall
    back to a prompt or another credential source. No key is copied into the
    process environment when a file is used.
    """
    environment = os.environ if environ is None else environ
    environment_key = environment.get("OPENAI_API_KEY", "")
    if environment_key.strip():
        return validate_api_key(environment_key)
    configured_file = environment.get("OPENAI_API_KEY_FILE", "").strip()
    path = Path(configured_file) if configured_file else (
        DEFAULT_KEY_FILE if default_file is None else Path(default_file)
    )
    descriptor = None
    try:
        # O_NONBLOCK prevents an unexpected pipe/device from hanging before
        # fstat can reject it. O_NOFOLLOW rejects a final symlink on macOS/Linux.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        if path.is_symlink():
            raise CredentialsError("The API-key file must be a regular file, not a symlink. " + SETUP_HELP)
        descriptor = os.open(path, flags)
        file_info = os.fstat(descriptor)
        if not stat.S_ISREG(file_info.st_mode):
            raise CredentialsError("The API-key file must be a regular file. " + SETUP_HELP)
        if file_info.st_size > MAX_KEY_BYTES:
            raise CredentialsError("The API-key file exceeds the 8 KB limit. " + SETUP_HELP)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            content = stream.read(MAX_KEY_BYTES + 1)
        if len(content) > MAX_KEY_BYTES:
            raise CredentialsError("The API-key file exceeds the 8 KB limit. " + SETUP_HELP)
        return validate_api_key(content.decode("utf-8"))
    except FileNotFoundError:
        if not configured_file:
            return None
        raise CredentialsError("The OPENAI_API_KEY_FILE file is missing. " + SETUP_HELP) from None
    except UnicodeDecodeError:
        raise CredentialsError("The API-key file must contain a UTF-8 text key. " + SETUP_HELP) from None
    except OSError:
        raise CredentialsError("The API-key file could not be read. Check its file permissions. " + SETUP_HELP) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reject_symlink_path(path: Path) -> None:
    """Avoid following either a secret symlink or a symlinked parent."""
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise CredentialsError("Choose an API-key file path without symlinks.")


def write_api_key(value: str, path: Path | None = None, *, replace: bool = False) -> Path:
    """Atomically save a private key, refusing accidental replacement."""
    key = validate_api_key(value)
    target = Path(path) if path is not None else DEFAULT_KEY_FILE
    target = target.absolute()
    _reject_symlink_path(target)
    try:
        if target.exists():
            file_info = target.stat()
            if not stat.S_ISREG(file_info.st_mode):
                raise CredentialsError("The API-key destination must be a regular file.")
            if file_info.st_size and not replace:
                raise CredentialsError("An API key is already saved. Use --replace only when changing it.")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _reject_symlink_path(target)
        target.parent.chmod(0o700)
        temporary_path = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(prefix=".openai-key-", dir=target.parent)
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(key + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _reject_symlink_path(target)
            if replace or (target.exists() and target.stat().st_size == 0):
                os.replace(temporary_path, target)
            else:
                # An exclusive hard link publishes the completed file atomically
                # and cannot overwrite a key created after our initial check.
                try:
                    os.link(temporary_path, target)
                except FileExistsError:
                    raise CredentialsError("An API key is already saved. Use --replace only when changing it.") from None
                temporary_path.unlink()
            temporary_path = None
            target.chmod(0o600)
            return target
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except OSError:
        raise CredentialsError("The private API-key file could not be saved. Check the destination permissions.") from None
