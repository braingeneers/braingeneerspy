from __future__ import annotations

import sys


def _missing_dependency_message(exc: ModuleNotFoundError) -> str:
    return (
        "braingeneers.mcp requires dependencies that should be included in a "
        "normal braingeneers install.\n"
        f"Missing module: {exc.name}\n\n"
        "Install or update braingeneerspy, for example:\n"
        "  python -m pip install -e "
        "/home/davidparks21/myprojects/Braingeneers/braingeneerspy"
    )


if __name__ == "__main__":
    try:
        from .adapter import main
    except ModuleNotFoundError as exc:
        print(_missing_dependency_message(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    main()
