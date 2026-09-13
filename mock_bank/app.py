import json
import secrets
import sys
from pathlib import Path
from urllib.parse import urlparse

# Running this as a plain script (python mock_bank/app.py) only puts
# this file's own directory on sys.path, not the repo root. Add the
# repo root explicitly so `src.config.env` resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask

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


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = env.mock_bank_secret_key.get_secret_value()
    # New value every startup; sessions from a previous run are rejected.
    app.config["BOOT_ID"] = secrets.token_hex(8)
    app.config["MEMBER_DATA"] = load_member_data()
    app.config["ACTIVITY"] = Activity()

    app.register_blueprint(auth_bp)
    app.register_blueprint(member_bp)
    app.register_blueprint(billpay_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    # Bind to MOCK_BANK_BASE_URL's host/port so the printed link, the
    # allowlist's permitted domain, and the config always agree.
    base = urlparse(env.mock_bank_base_url)
    # Debug off: no in-browser code console, no auto-reload wiping state mid-run.
    app.run(host=base.hostname, port=base.port or 5000, debug=False)
