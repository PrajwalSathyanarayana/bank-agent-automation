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

from blueprints.auth import auth_bp
from blueprints.member import member_bp
from blueprints.billpay import billpay_bp

DATA_PATH = Path(__file__).parent / "data" / "members.json"


def load_member_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = env.mock_bank_secret_key
    # New value every startup; sessions from a previous run are rejected (D028).
    app.config["BOOT_ID"] = secrets.token_hex(8)
    app.config["MEMBER_DATA"] = load_member_data()

    app.register_blueprint(auth_bp)
    app.register_blueprint(member_bp)
    app.register_blueprint(billpay_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    # Bind to MOCK_BANK_BASE_URL's host/port so the printed link, the
    # allowlist's permitted domain (D021), and the config always agree.
    base = urlparse(env.mock_bank_base_url)
    app.run(host=base.hostname, port=base.port or 5000, debug=True)
