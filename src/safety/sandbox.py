"""Where discovery may complete an irreversible step itself: only on a bank running on
this machine.

TARGET_ENVIRONMENT says whether the bank is a test copy; in a test copy discovery
completes irreversible steps, such as confirming a payment, to learn what follows them.
That claim is checked here, in code rather than configuration, so no .env mistake can
turn a real bank into a sandbox. A bank's test environment on its own server would need
an explicit list of approved test hosts, kept outside .env.
"""
from typing import Optional
from urllib.parse import urlparse

# This machine's names, as urlparse reports them: lower case, ::1 without its brackets.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def sandbox_refusal(url: str) -> Optional[str]:
    """Why the bank at this address can't be treated as a sandbox; None if it can.

    Only the address's host counts, so user info, a path or a query naming localhost
    can't make a remote bank look local. An address that can't be read is refused.
    """
    try:
        host = urlparse(url).hostname
    except ValueError:
        host = None
    if host in LOCAL_HOSTS:
        return None
    found = f"the start address's host is {host}" if host else "the start address has no host"
    return f"a sandbox must run on this machine (localhost, 127.0.0.1 or ::1), but {found}"
