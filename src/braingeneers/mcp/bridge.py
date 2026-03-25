from __future__ import annotations

import sys
import warnings

from braingeneers.mcp.token_manager import UserTokenManager


def main() -> None:
    warnings.warn(
        "python -m braingeneers.mcp.bridge is deprecated. "
        "Use python -m braingeneers.mcp instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "braingeneers.mcp.bridge now runs the stdio MCP adapter. "
        "Use python -m braingeneers.mcp instead.",
        file=sys.stderr,
    )
    from braingeneers.mcp.adapter import main as adapter_main

    adapter_main()


if __name__ == "__main__":
    main()
