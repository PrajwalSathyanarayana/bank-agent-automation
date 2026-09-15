import json
import secrets
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Running this as a plain script (python mock_bank/app.py) only puts
# this file's own directory on sys.path, not the repo root. Add the
# repo root explicitly so `src.config.env` resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request

from src.config.env import env

from blueprints.activity import Activity
from blueprints.auth import auth_bp
from blueprints.member import member_bp
from blueprints.billpay import billpay_bp

DATA_PATH = Path(__file__).parent / "data" / "members.json"
ACCOUNT_STATUSES = {"active", "restricted"}


def load_member_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    check_member_data(data)
    return data


def check_member_data(data: dict) -> None:
    """Refuses to start on contradictory data, so a data edit can't break the rules."""
    for member_id, member in data["members"].items():
        inactive = member["membership_status"] == "inactive"
        if inactive != bool(member.get("inactive_reason")):
            raise ValueError(
                f"member {member_id}: inactive_reason must be set exactly when membership is inactive"
            )
        for account in member["accounts"]:
            if account["status"] not in ACCOUNT_STATUSES:
                raise ValueError(
                    f"member {member_id}: unknown account status {account['status']!r}"
                )
            if inactive and account["status"] != "restricted":
                raise ValueError(
                    f"member {member_id}: inactive membership but account "
                    f"{account['account_id']} is not restricted"
                )


def create_app(renamed_menu: Optional[bool] = None, slow_pages_ms: Optional[int] = None) -> Flask:
    """The mock bank. The two test switches come from the settings unless given here, so a
    test can start a switched bank beside the normal one."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = env.mock_bank_secret_key.get_secret_value()
    # New value every startup; sessions from a previous run are rejected.
    app.config["BOOT_ID"] = secrets.token_hex(8)
    app.config["MEMBER_DATA"] = load_member_data()
    app.config["ACTIVITY"] = Activity()
    app.config["RENAMED_MENU"] = env.mock_bank_renamed_menu if renamed_menu is None else renamed_menu
    app.config["SLOW_PAGES_MS"] = env.mock_bank_slow_pages_ms if slow_pages_ms is None else slow_pages_ms

    @app.context_processor
    def menu_labels() -> dict:
        # A new vendor version relabels a menu item; its address stays the same.
        return {"member_search_label": "Find Member" if app.config["RENAMED_MENU"] else "Member Search"}

    @app.before_request
    def slow_pages() -> None:
        # Pages only: stylesheets and scripts arrive at their usual speed.
        if app.config["SLOW_PAGES_MS"] and not request.path.startswith("/static/"):
            time.sleep(app.config["SLOW_PAGES_MS"] / 1000)

    app.register_blueprint(auth_bp)
    app.register_blueprint(member_bp)
    app.register_blueprint(billpay_bp)

    return app


def switches_on(app: Flask) -> list[str]:
    """The test switches this bank runs with, for its start-up message."""
    on = []
    if app.config["RENAMED_MENU"]:
        on.append('menu item "Member Search" relabelled "Find Member"')
    if app.config["SLOW_PAGES_MS"]:
        on.append(f"every page served {app.config['SLOW_PAGES_MS']} ms late")
    return on


if __name__ == "__main__":
    app = create_app()
    # A switched bank must never look like the normal one.
    for switch in switches_on(app):
        print(f"TEST SWITCH ON: {switch}")
    # Bind to MOCK_BANK_BASE_URL's host/port so the printed link, the
    # allowlist's permitted domain, and the config always agree.
    base = urlparse(env.mock_bank_base_url)
    # Debug off: no in-browser code console, no auto-reload wiping state mid-run.
    app.run(host=base.hostname, port=base.port or 5000, debug=False)
