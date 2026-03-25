from __future__ import annotations

import importlib.resources
import json
import os
import sys
from pathlib import Path
from typing import Any


AUTH_DIR_ENV_VAR = "BRAINGENEERS_AUTH_DIR"
SERVICE_ACCOUNT_TOKEN_FILENAME = "service-account-token.json"
USER_TOKEN_FILENAME = "user-token.json"


def _legacy_service_account_dir() -> Path:
    return Path(importlib.resources.files("braingeneers.iot")) / "service_account"


def legacy_service_account_token_path() -> Path:
    return _legacy_service_account_dir() / "config.json"


def default_auth_dir() -> Path:
    override = os.environ.get(AUTH_DIR_ENV_VAR)
    if override:
        return Path(os.path.expanduser(override))

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Braingeneers" / "auth"
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root) / "Braingeneers" / "auth"
        return home / "AppData" / "Local" / "Braingeneers" / "auth"

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state_home) if xdg_state_home else home / ".local" / "state"
    return base / "braingeneers" / "auth"


def ensure_auth_dir() -> Path:
    auth_dir = default_auth_dir()
    auth_dir.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        auth_dir.chmod(0o700)
    return auth_dir


def service_account_token_path() -> Path:
    return ensure_auth_dir() / SERVICE_ACCOUNT_TOKEN_FILENAME


def user_token_path() -> Path:
    return ensure_auth_dir() / USER_TOKEN_FILENAME

def resolve_service_account_token_path() -> Path:
    primary = service_account_token_path()
    if primary.exists():
        return primary

    legacy = legacy_service_account_token_path()
    if legacy.exists():
        return legacy
    return primary


def load_token(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_token(path: Path, token_data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(token_data, handle)
    if os.name != "nt":
        path.chmod(0o600)
