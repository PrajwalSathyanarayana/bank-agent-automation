from functools import wraps

from flask import Blueprint, redirect, render_template, request, session, url_for

auth_bp = Blueprint("auth", __name__)

VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"


def login_required(view_func):
    """Redirects to /session-timeout when no valid session exists.

    Deterministic stand-in for real session expiry (D018): the test
    harness can produce this exact condition on demand by clearing
    cookies, instead of waiting out a real timer.
    """
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("auth.session_timeout"))
        return view_func(*args, **kwargs)
    return wrapped


@auth_bp.route("/login", methods=["GET"])
def login():
    return render_template("login.html", error=None)


@auth_bp.route("/login", methods=["POST"])
def login_submit():
    username = request.form.get("username", "")
    password = request.form.get("password", "")

    if username == VALID_USERNAME and password == VALID_PASSWORD:
        session["logged_in"] = True
        session["username"] = username
        return redirect(url_for("member.dashboard"))

    return render_template("login.html", error="Invalid username or password."), 200


@auth_bp.route("/session-timeout", methods=["GET"])
def session_timeout():
    session.clear()
    return render_template("session_timeout.html")
