from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "Auth0TokenVerifier": "braingeneers.mcp.auth",
    "MCPAuthRuntimeConfig": "braingeneers.mcp.auth",
    "OIDCTokenVerifier": "braingeneers.mcp.auth",
    "VerifiedAccessToken": "braingeneers.mcp.auth",
    "extract_claim_values": "braingeneers.mcp.auth",
    "principal_from_access_token": "braingeneers.mcp.auth",
    "principal_from_current_token": "braingeneers.mcp.auth",
    "AuthorizationDecision": "braingeneers.mcp.iam",
    "AuthorizationPolicy": "braingeneers.mcp.iam",
    "IAMCommandSpec": "braingeneers.mcp.iam",
    "IAMPolicyAdapter": "braingeneers.mcp.iam",
    "PolicyLoader": "braingeneers.mcp.iam",
    "PolicyValidationError": "braingeneers.mcp.iam",
    "Principal": "braingeneers.mcp.models",
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    return getattr(module, name)
