"""Prints each handoff announcement from a running system: a stand-in for an operations
dashboard. Start it in a second terminal, before or during a run with --operator:

    python -m src.handoff.watch

It waits for the feed to appear (a run with --operator opens it), and waits again after
each run ends. Ctrl+C stops it.
"""
import argparse
import asyncio
import json
import sys
from typing import Any, Optional, Sequence

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from src.config.env import env
from src.handoff.ws_server import FEED_HOST

_RETRY_S = 2.0
_ENDINGS = {
    "RESUMED": "handed back; the run carries on",
    "MANUAL_COMPLETED": "finished by the person",
    "ABORTED": "stopped",
    "OPERATOR_TIMED_OUT": "no one finished in time; the run stopped",
}


def describe(announcement: dict[str, Any]) -> str:
    """One announcement as a line a person reads."""
    event = announcement.get("event", "")
    at = str(announcement.get("at", ""))[11:19]
    run = f"run {str(announcement.get('run_id', ''))[:8]}"
    if event == "HANDOFF_REQUESTED":
        step = announcement.get("step_index")
        where = f" at step {step}" if step is not None else ""
        return (f"[{at}] A PERSON IS NEEDED{where} ({run}): {announcement.get('why')}. "
                f"Task: {announcement.get('goal')} Take over in the run's browser window.")
    if event == "HANDOFF_STARTED":
        return f"[{at}] Taken over by a person ({run})."
    if event == "HANDOFF_RESOLVED":
        ending = _ENDINGS.get(str(announcement.get("resolution")), str(announcement.get("resolution")))
        if announcement.get("window_closed"):
            ending = "stopped: the window was closed"
        return f"[{at}] Control is back with the automation ({run}): {ending}."
    return f"[{at}] {event} ({run})."


async def watch(port: int) -> None:
    url = f"ws://{FEED_HOST}:{port}"
    waiting_said = False
    while True:
        try:
            async with connect(url) as feed:
                print(f"Listening to the handoff feed at {url}.", flush=True)
                waiting_said = False
                async for message in feed:
                    print(describe(json.loads(message)), flush=True)
            print("The run ended.", flush=True)
        except (OSError, WebSocketException):
            if not waiting_said:
                print(f"Waiting for a run with --operator (feed at {url})...", flush=True)
                waiting_said = True
        await asyncio.sleep(_RETRY_S)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.handoff.watch")
    parser.add_argument("--port", type=int, default=env.ws_handoff_port,
                        help=f"the feed's port (default {env.ws_handoff_port})")
    args = parser.parse_args(argv)
    try:
        asyncio.run(watch(args.port))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
