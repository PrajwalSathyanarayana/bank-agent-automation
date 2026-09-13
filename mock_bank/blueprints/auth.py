from functools import wraps

from flask import Blueprint, current_app, redirect, render_template, request, session, url_for

auth_bp = Blueprint("auth", __name__)

VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"


def has_valid_session() -> bool:
    """A session counts only if this server run issued it.

    The cookie is signed with the SECRET_KEY from .env, so it stays
    cryptographically valid across restarts; the boot ID is what ties it
    to one server lifetime, the way legacy in-memory sessions behaved.
    """
    return bool(session.get("logged_in")) and session.get("boot_id") == current_app.config["BOOT_ID"]


@auth_bp.app_context_processor
def inject_session_state():
    return {"signed_in": has_valid_session()}


def login_required(view_func):
    """Redirects to /session-timeout when no valid session exists.

    Deterministic stand-in for real session expiry: the test
    harness can produce this exact condition on demand by clearing
    cookies, instead of waiting out a real timer.
    """
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not has_valid_session():
            return redirect(url_for("auth.session_timeout"))
        return view_func(*args, **kwargs)
    return wrapped


@auth_bp.route("/", methods=["GET"])
def home():
    if has_valid_session():
        return redirect(url_for("member.dashboard"))
    return render_template("home.html")


@auth_bp.route("/login", methods=["GET"])
def login():
    message = "You have been signed off." if request.args.get("signed_off") else None
    return render_template("login.html", error=None, message=message)


@auth_bp.route("/login", methods=["POST"])
def login_submit():
    username = request.form.get("username", "")
    password = request.form.get("password", "")

    if username == VALID_USERNAME and password == VALID_PASSWORD:
        session["logged_in"] = True
        session["username"] = username
        session["boot_id"] = current_app.config["BOOT_ID"]
        return redirect(url_for("member.dashboard"))

    return render_template("login.html", error="Invalid username or password.", message=None), 200


@auth_bp.route("/logout", methods=["GET"])
def logout():
    # A plain GET link, as legacy portals did. A modern app would
    # use POST so another site can't silently sign the user out.
    session.clear()
    return redirect(url_for("auth.login", signed_off=1))


@auth_bp.route("/session-timeout", methods=["GET"])
def session_timeout():
    session.clear()
    return render_template("session_timeout.html")
