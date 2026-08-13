from __future__ import annotations

from typing import Any

import pytest
from openpi_client import msgpack_numpy

from pi0_minimal.teacher_client import LongInferenceWebsocketClientPolicy


class _FakeConnection:
    def recv(self) -> bytes:
        return msgpack_numpy.Packer().pack({"ready": True})


def test_long_inference_client_extends_ping_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call: dict[str, Any] = {}

    def fake_connect(uri: str, **kwargs: Any) -> _FakeConnection:
        call["uri"] = uri
        call.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr("websockets.sync.client.connect", fake_connect)
    client = LongInferenceWebsocketClientPolicy("127.0.0.1", 8000)

    assert client.get_server_metadata() == {"ready": True}
    assert call["uri"] == "ws://127.0.0.1:8000"
    assert call["ping_interval"] == 20.0
    assert call["ping_timeout"] == 180.0
    assert call["compression"] is None
    assert call["max_size"] is None


@pytest.mark.parametrize(
    ("ping_interval", "ping_timeout"),
    [(0.0, 180.0), (20.0, 0.0), (-1.0, 180.0), (20.0, -1.0)],
)
def test_long_inference_client_rejects_invalid_ping_settings(
    ping_interval: float,
    ping_timeout: float,
) -> None:
    with pytest.raises(ValueError, match="ping settings must be positive"):
        LongInferenceWebsocketClientPolicy(
            "127.0.0.1",
            8000,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        )
