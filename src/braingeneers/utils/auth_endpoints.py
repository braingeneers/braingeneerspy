from __future__ import annotations


SERVICE_ACCOUNT_TOKEN_URL = "https://service-accounts.braingeneers.gi.ucsc.edu/generate_token"
OAUTH2_BROKER_REALM = "braingeneers"
OAUTH2_BROKER_BASE_URL = "https://oauth2.braingeneers.gi.ucsc.edu"
OAUTH2_BROKER_ISSUER_URL = (
    f"{OAUTH2_BROKER_BASE_URL}/realms/{OAUTH2_BROKER_REALM}"
)
OAUTH2_BROKER_JWKS_URL = (
    f"{OAUTH2_BROKER_ISSUER_URL}/protocol/openid-connect/certs"
)
OAUTH2_BROKER_TOKEN_URL = (
    f"{OAUTH2_BROKER_ISSUER_URL}/protocol/openid-connect/token"
)
OAUTH2_BROKER_DEVICE_AUTHORIZATION_URL = (
    f"{OAUTH2_BROKER_ISSUER_URL}/protocol/openid-connect/auth/device"
)
MCP_HELPER_CLIENT_ID = "braingeneerspy-bridge"
MCP_REQUESTED_SCOPE = (
    "openid profile email offline_access "
    "mcp:tools mcp:resources mcp:prompts"
)

# Backward-compatible aliases for the current Keycloak client naming.
BRIDGE_CLIENT_ID = MCP_HELPER_CLIENT_ID
BRIDGE_REQUESTED_SCOPE = MCP_REQUESTED_SCOPE
