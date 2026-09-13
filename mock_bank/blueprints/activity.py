import threading

from flask import current_app


class Activity:
    """Dashboard counters, kept in memory: reset on restart, like balances and sessions."""

    def __init__(self):
        # The server handles requests on several threads.
        self._lock = threading.Lock()
        self._members_viewed = set()
        self._payments_completed = 0
        self._payments_total = 0.0
        self._payments_blocked = 0

    def member_viewed(self, member_id: str) -> None:
        with self._lock:
            self._members_viewed.add(member_id)

    def payment_completed(self, amount: float) -> None:
        with self._lock:
            self._payments_completed += 1
            self._payments_total += amount

    def payment_blocked(self) -> None:
        with self._lock:
            self._payments_blocked += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "members_viewed": len(self._members_viewed),
                "payments_completed": self._payments_completed,
                "payments_total": self._payments_total,
                "payments_blocked": self._payments_blocked,
            }


def current_activity() -> Activity:
    return current_app.config["ACTIVITY"]
