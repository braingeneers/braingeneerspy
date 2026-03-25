from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MCP_SERVICE_NAME = "integrated-system-mcp"


@dataclass(frozen=True)
class MCPServiceProfile:
    name: str
    remote_base_url: str
    audience: str
    display_name: str | None = None
    description: str | None = None
    mcp_path: str = "/mcp"

    @property
    def remote_mcp_url(self) -> str:
        base = self.remote_base_url.rstrip("/")
        path = self.mcp_path if self.mcp_path.startswith("/") else f"/{self.mcp_path}"
        return f"{base}{path}"

DEFAULT_SERVICE_PROFILE = MCPServiceProfile(
    name=DEFAULT_MCP_SERVICE_NAME,
    remote_base_url="https://integrated-system-mcp.braingeneers.gi.ucsc.edu",
    audience="https://integrated-system-mcp.braingeneers.gi.ucsc.edu/",
    display_name="Integrated System MCP",
    description="Braingeneers integrated-system tools and resources.",
)


def get_default_service_profile() -> MCPServiceProfile:
    return DEFAULT_SERVICE_PROFILE
