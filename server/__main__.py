"""Entry point: uv run python -m server"""

import asyncio
import logging

import uvicorn

from server.main import create_app
from server.settings import load_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    config = uvicorn.Config(create_app(settings), host=settings.host, port=settings.port)
    # Not uvicorn.run(): on Windows it picks the ProactorEventLoop, which async psycopg can't use.
    asyncio.run(uvicorn.Server(config).serve(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
