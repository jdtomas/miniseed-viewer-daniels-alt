"""Development and production entry point."""

from __future__ import annotations

import uvicorn

from .config import Settings
from .web import create_application


def app_factory():
    """ASGI factory used by Uvicorn deployments."""
    server, _app = create_application(Settings())
    return server


def main() -> None:
    server = app_factory()
    uvicorn.run(server, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
