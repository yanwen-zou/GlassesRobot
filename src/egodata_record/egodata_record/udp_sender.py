#!/usr/bin/env python3
from __future__ import annotations

import socket
from typing import Final, Iterable, Tuple


class UDPSender:
    """Lightweight helper for sending UDP messages to a fixed endpoint."""

    # def __init__(self, target_host: str = '192.168.43.138', target_port: int = 5007) -> None:
    def __init__(self, target_host: str = '10.130.4.109', target_port: int = 5007) -> None:
        self._target: Final[Tuple[str, int]] = (target_host, target_port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, message: str) -> None:
        """Send a unicode message to the configured UDP endpoint."""
        if not isinstance(message, str):
            raise TypeError('message must be a string')
        data = message.encode('utf-8')
        self._socket.sendto(data, self._target)

    def batch_send(self, messages: Iterable[str]) -> None:
        """Send multiple messages in sequence."""
        for message in messages:
            self.send(message)

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> 'UDPSender':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

