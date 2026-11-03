"""Smoke test so `uv run pytest` passes on the empty skeleton. Delete once real tests exist."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    ["server", "server.common", "server.ingest", "server.alerts", "server.assistant"],
)
def test_packages_import(module: str) -> None:
    importlib.import_module(module)
