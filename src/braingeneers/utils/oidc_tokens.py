from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_oidc_token_response(
    token_response: dict[str, Any],
    *,
    issuer_url: str,
    jwks_url: str,
    token_url: str,
    client_id: str,
    audience: str,
    requested_scope: str,
) -> dict[str, Any]:
    issued_at = utcnow()
    access_expires_at = issued_at + timedelta(
        seconds=int(token_response.get("expires_in", 0))
    )
    normalized: dict[str, Any] = {
        "issuer": issuer_url,
        "jwks_url": jwks_url,
        "token_url": token_url,
        "client_id": client_id,
        "audience": audience,
        "requested_scope": requested_scope,
        "access_token": token_response["access_token"],
        "expires_at": format_utc(access_expires_at),
    }

    if token_response.get("token_type"):
        normalized["token_type"] = str(token_response["token_type"])
    if token_response.get("scope"):
        normalized["granted_scope"] = str(token_response["scope"])
    if token_response.get("refresh_token"):
        normalized["refresh_token"] = str(token_response["refresh_token"])
    if token_response.get("refresh_expires_in") is not None:
        refresh_expires_at = issued_at + timedelta(
            seconds=int(token_response["refresh_expires_in"])
        )
        normalized["refresh_expires_at"] = format_utc(refresh_expires_at)
    if token_response.get("id_token"):
        normalized["id_token"] = str(token_response["id_token"])

    return normalized
