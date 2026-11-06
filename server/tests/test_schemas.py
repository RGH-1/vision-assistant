"""Checks that server/common/schemas.py matches the WebSocket contract."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from server.common.schemas import PingMessage, WebSocketMessage
from server.settings import ROOT


def test_every_contract_example_is_a_valid_message() -> None:
    path = ROOT / "contracts" / "events.examples.json"
    examples = json.loads(path.read_text(encoding="utf-8"))
    adapter = TypeAdapter(WebSocketMessage)

    types = {type(adapter.validate_python(example)).__name__ for example in examples}

    assert len(types) == 7  # hello, alert, status, frame_stored, query_answered, ping, pong


def test_undocumented_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PingMessage(type="ping", extra_field=1)
