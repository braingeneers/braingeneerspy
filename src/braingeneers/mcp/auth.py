from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import jwt
from jwt import PyJWKClient
try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover
    class BaseModel:
        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

    def Field(*, default: Any = None, default_factory: Callable[[], Any] | None = None, **_: Any) -> Any:
        if default_factory is not None:
            return default_factory()
        return default

try:
    from mcp.server.auth.middleware.auth_context import get_access_token
    from mcp.server.auth.provider import AccessToken, TokenVerifier
    from mcp.server.auth.settings import AuthSettings
except ModuleNotFoundError:  # pragma: no cover
    def get_access_token():
        return None

    class AccessToken(BaseModel):
        token: str
        client_id: str
        scopes: list[str] = Field(default_factory=list)
        expires_at: int | None = None
        resource: str | None = None

    class TokenVerifier:
        async def verify_token(self, token: str) -> AccessToken | None:
            raise NotImplementedError

    @dataclass(frozen=True)
    class AuthSettings:
        issuer_url: str
        resource_server_url: str
        service_documentation_url: str

from .models import Principal


class VerifiedAccessToken(AccessToken):
    subject: str | None = None
    roles: list[str] = Field(default_factory=list)
    claims: dict[str, Any] = Field(default_factory=dict, exclude=True)


def _normalize_claim_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        separators = [",", " "]
        values = [value]
        for separator in separators:
            next_values: list[str] = []
            for item in values:
                next_values.extend(item.split(separator))
            values = next_values
        return [item.strip() for item in values if item.strip()]
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_normalize_claim_values(item))
        return values
    return [str(value)]


def extract_claim_values(claims: dict[str, Any], claim_path: str) -> list[str]:
    if claim_path in claims:
        return _normalize_claim_values(claims[claim_path])

    current_value: Any = claims
    for part in claim_path.split("."):
        if not isinstance(current_value, dict) or part not in current_value:
            return []
        current_value = current_value[part]
    return _normalize_claim_values(current_value)


class OIDCTokenVerifier(TokenVerifier):
    def __init__(
        self,
        *,
        issuer_url: str,
        jwks_url: str,
        audience: str,
        role_claim_paths: tuple[str, ...],
        algorithms: tuple[str, ...] = ("RS256",),
        key_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self.issuer_url = issuer_url
        self._acceptable_issuers = tuple(
            dict.fromkeys(
                [
                    issuer_url,
                    issuer_url.rstrip("/"),
                    issuer_url.rstrip("/") + "/",
                ]
            )
        )
        self.jwks_url = jwks_url
        self.audience = audience
        self.role_claim_paths = role_claim_paths
        self.algorithms = algorithms
        self.key_resolver = key_resolver
        self._jwks_client = None if key_resolver else PyJWKClient(self.jwks_url)

    def _resolve_key(self, token: str) -> Any:
        if self.key_resolver is not None:
            return self.key_resolver(token)
        if self._jwks_client is None:
            raise RuntimeError("JWKS client is not configured.")
        return self._jwks_client.get_signing_key_from_jwt(token).key

    def _decode_claims(self, token: str) -> dict[str, Any]:
        verification_key = self._resolve_key(token)
        claims = jwt.decode(
            token,
            verification_key,
            algorithms=list(self.algorithms),
            audience=self.audience,
            options={"verify_iss": False},
        )
        issuer = claims.get("iss")
        if issuer not in self._acceptable_issuers:
            raise jwt.InvalidIssuerError(f"Invalid issuer: {issuer}")
        return claims

    def _extract_roles(self, claims: dict[str, Any]) -> list[str]:
        roles: set[str] = set()
        for claim_path in self.role_claim_paths:
            roles.update(extract_claim_values(claims, claim_path))
        return sorted(roles)

    def _extract_scopes(self, claims: dict[str, Any], roles: list[str]) -> list[str]:
        scopes: set[str] = set()
        scopes.update(_normalize_claim_values(claims.get("scope")))
        scopes.update(_normalize_claim_values(claims.get("permissions")))
        scopes.update(_normalize_claim_values(claims.get("scp")))
        scopes.update(roles)
        return sorted(scopes)

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = await asyncio.to_thread(self._decode_claims, token)
        except Exception:
            return None

        roles = self._extract_roles(claims)
        scopes = self._extract_scopes(claims, roles)
        client_id = (
            claims.get("azp")
            or claims.get("client_id")
            or claims.get("sub")
            or "unknown"
        )
        return VerifiedAccessToken(
            token=token,
            client_id=str(client_id),
            subject=str(claims.get("sub") or client_id),
            scopes=scopes,
            roles=roles,
            expires_at=claims.get("exp"),
            resource=self.audience,
            claims=claims,
        )


@dataclass(frozen=True)
class MCPAuthRuntimeConfig:
    issuer_url: str
    jwks_url: str
    audience: str
    resource_server_url: str
    role_claim_paths: tuple[str, ...]
    algorithms: tuple[str, ...] = ("RS256",)

    def build_token_verifier(
        self,
        *,
        key_resolver: Callable[[str], Any] | None = None,
    ) -> OIDCTokenVerifier:
        return OIDCTokenVerifier(
            issuer_url=self.issuer_url,
            jwks_url=self.jwks_url,
            audience=self.audience,
            role_claim_paths=self.role_claim_paths,
            algorithms=self.algorithms,
            key_resolver=key_resolver,
        )

    def build_auth_settings(self, *, service_documentation_url: str) -> AuthSettings:
        return AuthSettings(
            issuer_url=self.issuer_url,
            resource_server_url=self.resource_server_url,
            service_documentation_url=service_documentation_url,
        )

    def audit_fields(self) -> dict[str, Any]:
        return {
            "issuer_url": self.issuer_url,
            "jwks_url": self.jwks_url,
            "audience": self.audience,
            "resource_server_url": self.resource_server_url,
            "role_claim_paths": list(self.role_claim_paths),
            "algorithms": list(self.algorithms),
        }


def principal_from_access_token(access_token: AccessToken | None) -> Principal:
    if access_token is None:
        raise PermissionError("Authentication required.")

    subject = str(getattr(access_token, "subject", None) or access_token.client_id)
    roles = tuple(sorted(set(getattr(access_token, "roles", []))))
    scopes = tuple(sorted(set(getattr(access_token, "scopes", []))))
    claims = dict(getattr(access_token, "claims", {}))
    return Principal(
        client_id=str(access_token.client_id),
        subject=subject,
        roles=roles,
        scopes=scopes,
        claims=claims,
    )


def principal_from_current_token() -> Principal:
    return principal_from_access_token(get_access_token())


# Backwards-compatible alias for the original Auth0-specific name used by the
# first MCP rollout. The verifier is provider-neutral as long as the issuer,
# JWKS, audience, and claim mapping are configured correctly.
Auth0TokenVerifier = OIDCTokenVerifier
