"""WebSocket client settings for long-running official-teacher inference."""

from __future__ import annotations

import logging
import time
from typing import Any

import websockets.sync.client
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy

_LOGGER = logging.getLogger(__name__)


class LongInferenceWebsocketClientPolicy(WebsocketClientPolicy):
    """Keep liveness checks while allowing slow teacher responses."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        *,
        ping_interval: float = 20.0,
        ping_timeout: float = 180.0,
    ) -> None:
        if ping_interval <= 0.0 or ping_timeout <= 0.0:
            raise ValueError("WebSocket ping settings must be positive")
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        super().__init__(host=host, port=port, api_key=api_key)

    def _wait_for_server(self) -> tuple[Any, dict[str, Any]]:
        _LOGGER.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                connection = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                )
                metadata = msgpack_numpy.unpackb(connection.recv())
                return connection, metadata
            except ConnectionRefusedError:
                _LOGGER.info("Still waiting for server...")
                time.sleep(5)
