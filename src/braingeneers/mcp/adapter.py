from __future__ import annotations

import argparse
import base64
import logging
import sys
from dataclasses import dataclass
from typing import Iterable

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from mcp.types import (
    AnyUrl,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    ResourceTemplate,
)

import braingeneers
from braingeneers.mcp.service_profiles import (
    MCPServiceProfile,
    get_default_service_profile,
)
from braingeneers.mcp.token_manager import UserTokenManager
from braingeneers.utils.auth_storage import user_token_path


LOGGER = logging.getLogger(__name__)


class RefreshingBearerAuth(httpx.Auth):
    def __init__(self, token_manager: UserTokenManager) -> None:
        self._token_manager = token_manager

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = (
            f"Bearer {self._token_manager.get_access_token()}"
        )
        response = yield request
        if response.status_code != 401:
            return
        response.read()
        request.headers["Authorization"] = (
            f"Bearer {self._token_manager.force_refresh()}"
        )
        yield request


@dataclass
class RemoteMCPAdapter:
    profile: MCPServiceProfile
    remote: ClientSession

    async def list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        cursor = request.params.cursor if request.params else None
        return await self.remote.list_tools(cursor=cursor)

    async def call_tool(self, tool_name: str, arguments: dict | None):
        return await self.remote.call_tool(tool_name, arguments or {})

    async def list_resources(
        self, request: ListResourcesRequest
    ) -> ListResourcesResult:
        cursor = request.params.cursor if request.params else None
        return await self.remote.list_resources(cursor=cursor)

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        result = await self.remote.list_resource_templates()
        return list(result.resourceTemplates)

    async def read_resource(self, uri: AnyUrl) -> Iterable[ReadResourceContents]:
        result = await self.remote.read_resource(uri)
        contents: list[ReadResourceContents] = []
        for item in result.contents:
            mime_type = getattr(item, "mimeType", None)
            meta = getattr(item, "meta", None)
            if hasattr(item, "text"):
                contents.append(
                    ReadResourceContents(item.text, mime_type=mime_type, meta=meta)
                )
                continue
            blob = base64.b64decode(item.blob)
            contents.append(
                ReadResourceContents(blob, mime_type=mime_type, meta=meta)
            )
        return contents

    async def list_prompts(self) -> ListPromptsResult:
        return await self.remote.list_prompts()

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None
    ) -> GetPromptResult:
        return await self.remote.get_prompt(name, arguments)

    async def subscribe_resource(self, uri: AnyUrl) -> None:
        await self.remote.subscribe_resource(uri)

    async def unsubscribe_resource(self, uri: AnyUrl) -> None:
        await self.remote.unsubscribe_resource(uri)


def build_stdio_server(adapter: RemoteMCPAdapter) -> Server:
    profile = adapter.profile
    description = profile.description or (
        "Braingeneers stdio MCP adapter for a remote protected MCP service."
    )
    server = Server(
        name=f"braingeneers-{profile.name}",
        version=braingeneers.__version__,
        instructions=description,
    )

    @server.list_tools()
    async def list_tools(request: ListToolsRequest) -> ListToolsResult:
        return await adapter.list_tools(request)

    @server.call_tool()
    async def call_tool(tool_name: str, arguments: dict | None):
        return await adapter.call_tool(tool_name, arguments)

    @server.list_resources()
    async def list_resources(request: ListResourcesRequest) -> ListResourcesResult:
        return await adapter.list_resources(request)

    @server.list_resource_templates()
    async def list_resource_templates() -> list[ResourceTemplate]:
        return await adapter.list_resource_templates()

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> Iterable[ReadResourceContents]:
        return await adapter.read_resource(uri)

    @server.list_prompts()
    async def list_prompts() -> ListPromptsResult:
        return await adapter.list_prompts()

    @server.get_prompt()
    async def get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> GetPromptResult:
        return await adapter.get_prompt(name, arguments)

    @server.subscribe_resource()
    async def subscribe_resource(uri: AnyUrl) -> None:
        await adapter.subscribe_resource(uri)

    @server.unsubscribe_resource()
    async def unsubscribe_resource(uri: AnyUrl) -> None:
        await adapter.unsubscribe_resource(uri)

    return server


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _emit_interactive_status(message: str) -> None:
    if sys.stderr.isatty():
        print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Braingeneers local stdio MCP adapter."
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level for stderr output while the stdio adapter is running.",
    )
    return parser.parse_args()


async def _run_stdio_adapter(args: argparse.Namespace) -> None:
    profile = get_default_service_profile()
    token_manager = UserTokenManager(
        token_path=user_token_path(),
        service_name=profile.name,
    )
    auth = RefreshingBearerAuth(token_manager)
    timeout = httpx.Timeout(30.0, read=300.0)
    _emit_interactive_status(
        f"Connecting Braingeneers MCP adapter to {profile.remote_mcp_url} ..."
    )

    async with httpx.AsyncClient(auth=auth, timeout=timeout) as http_client:
        async with streamable_http_client(
            profile.remote_mcp_url,
            http_client=http_client,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as remote:
                await remote.initialize()
                adapter = RemoteMCPAdapter(profile=profile, remote=remote)
                server = build_stdio_server(adapter)
                LOGGER.info(
                    "Connected stdio adapter to %s (%s)",
                    profile.name,
                    profile.remote_mcp_url,
                )
                _emit_interactive_status(
                    "Braingeneers MCP adapter connected. Waiting for MCP client requests on stdio."
                )
                async with stdio_server() as (stdio_read, stdio_write):
                    await server.run(
                        stdio_read,
                        stdio_write,
                        server.create_initialization_options(
                            notification_options=NotificationOptions()
                        ),
                    )


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)
    anyio.run(_run_stdio_adapter, args)
