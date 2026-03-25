from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from braingeneers.utils.auth_endpoints import (
    MCP_HELPER_CLIENT_ID,
    MCP_REQUESTED_SCOPE,
    OAUTH2_BROKER_ISSUER_URL,
    OAUTH2_BROKER_JWKS_URL,
    OAUTH2_BROKER_TOKEN_URL,
)
from braingeneers.utils.auth_storage import load_token, save_token
from braingeneers.utils.oidc_tokens import normalize_oidc_token_response


ACCESS_TOKEN_REFRESH_SKEW = timedelta(minutes=5)


def _parse_expires_at(expires_at: str) -> datetime:
    return datetime.fromisoformat(expires_at.replace(" UTC", "")).replace(
        tzinfo=timezone.utc
    )


@dataclass
class UserTokenManager:
    token_path: Path
    service_name: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _authenticate_help(self) -> str:
        return "python -m braingeneers.iot.authenticate"

    def _load_token(self) -> dict:
        token_data = load_token(self.token_path)
        if token_data is None:
            raise PermissionError(
                "Interactive user token not found. Run "
                f"{self._authenticate_help()}"
            )
        return token_data

    def _refresh_token(self, token_data: dict) -> dict:
        refresh_token = token_data.get("refresh_token")
        token_url = str(token_data.get("token_url") or OAUTH2_BROKER_TOKEN_URL)
        client_id = str(token_data.get("client_id") or MCP_HELPER_CLIENT_ID)
        if not refresh_token:
            raise PermissionError(
                "Interactive user token is missing refresh metadata. Re-run "
                f"{self._authenticate_help()}."
            )

        response = requests.post(
            token_url,
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": refresh_token,
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise PermissionError(
                "Failed to refresh interactive user token. Re-run "
                f"{self._authenticate_help()}. Response: {response.text}"
            )
        refreshed = normalize_oidc_token_response(
            response.json(),
            issuer_url=str(token_data.get("issuer") or OAUTH2_BROKER_ISSUER_URL),
            jwks_url=str(token_data.get("jwks_url") or OAUTH2_BROKER_JWKS_URL),
            token_url=token_url,
            client_id=client_id,
            audience=str(token_data.get("audience") or ""),
            requested_scope=str(
                token_data.get("requested_scope") or MCP_REQUESTED_SCOPE
            ),
        )
        if "refresh_token" not in refreshed:
            refreshed["refresh_token"] = str(refresh_token)
        if "refresh_expires_at" not in refreshed and token_data.get(
            "refresh_expires_at"
        ):
            refreshed["refresh_expires_at"] = str(token_data["refresh_expires_at"])
        if token_data.get("selected_service") and "selected_service" not in refreshed:
            refreshed["selected_service"] = str(token_data["selected_service"])
        save_token(self.token_path, refreshed)
        return refreshed

    def force_refresh(self) -> str:
        with self._lock:
            token_data = self._refresh_token(self._load_token())
            return str(token_data["access_token"])

    def get_access_token(self) -> str:
        with self._lock:
            token_data = self._load_token()
            expires_at = _parse_expires_at(token_data["expires_at"])
            if expires_at - datetime.now(timezone.utc) <= ACCESS_TOKEN_REFRESH_SKEW:
                token_data = self._refresh_token(token_data)
            return str(token_data["access_token"])
