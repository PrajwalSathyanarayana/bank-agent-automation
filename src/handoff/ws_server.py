"""The handoff feed: each announcement (a person is needed, taken over, handed back) sent on a
WebSocket on this machine, for anyone not looking at the run's window — an operations
dashboard, a ticket queue, or watch.py for the demo.

The feed only tells; control stays in the window, so nothing a listener sends is ever acted
on, and a listener that falls away never affects the run. A listener that connects while a
handoff is open is told about it straight away, so a dashboard opened late still sees who
is waiting.
"""
import errno
import json
from typing import Any, Optional

from websockets.asyncio.server import Server, ServerConnection, broadcast, serve
from websockets.exceptions import ConnectionClosed

# This machine only: announcements carry the task's goal, which names a member.
FEED_HOST = "127.0.0.1"
_RESOLVED = "HANDOFF_RESOLVED"
# "Address already in use": the POSIX code, and Windows' own (WSAEADDRINUSE).
_IN_USE = {errno.EADDRINUSE, 10048}


class FeedUnavailable(RuntimeError):
    """The feed couldn't start (most often: the port is in use). The run can go on without it."""


class HandoffFeed:
    """Use as `async with HandoffFeed(port) as feed:`; pass it to the handoff as its announcer.
    Port 0 lets the system pick a free port (tests)."""

    def __init__(self, port: int) -> None:
        self._port = port
        self._server: Optional[Server] = None
        self._listeners: set[ServerConnection] = set()
        # The latest announcement of a handoff not yet resolved, for listeners who join late.
        self._open: Optional[str] = None

    @property
    def port(self) -> int:
        """The port the feed listens on, once started."""
        return self._server.sockets[0].getsockname()[1] if self._server else self._port

    @property
    def listening(self) -> int:
        """How many listeners are connected now."""
        return len(self._listeners)

    async def __aenter__(self) -> "HandoffFeed":
        try:
            self._server = await serve(self._listen, FEED_HOST, self._port)
        except OSError as error:
            # The usual cause in plain words; the system's own wording otherwise.
            reason = "it is already in use" if error.errno in _IN_USE else (error.strerror or str(error))
            raise FeedUnavailable(f"the handoff feed couldn't listen on port {self._port} ({reason})") from None
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def announce(self, announcement: dict[str, Any]) -> None:
        """Send one announcement to every listener; one that can't receive it is skipped."""
        message = json.dumps(announcement, default=str)
        self._open = None if announcement.get("event") == _RESOLVED else message
        broadcast(self._listeners, message)

    async def _listen(self, connection: ServerConnection) -> None:
        self._listeners.add(connection)
        try:
            if self._open is not None:
                await connection.send(self._open)
            # Read and drop whatever a listener sends, until it leaves.
            async for _ in connection:
                pass
        except ConnectionClosed:
            pass
        finally:
            self._listeners.discard(connection)
