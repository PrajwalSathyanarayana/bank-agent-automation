import random
import re

from flask import Blueprint, current_app, redirect, render_template, request, session, url_for

from .activity import current_activity
from .auth import login_required

# The one phone format the credit union keeps, as in its member records: (602) 555-0142.
PHONE_FORMAT = re.compile(r"\(\d{3}\) \d{3}-\d{4}")
PHONE_ERROR = "Phone must be in the form (NNN) NNN-NNNN."
# An address with one @, a name before it and a dotted domain after it; no spaces.
EMAIL_FORMAT = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
EMAIL_ERROR = "Email must be a valid address, like name@example.com."

member_bp = Blueprint("member", __name__)


def _get_member(member_id):
    return current_app.config["MEMBER_DATA"]["members"].get(member_id)


def _open_member(member_id):
    session["current_member_id"] = member_id
    current_activity().member_viewed(member_id)


def _restricted_member_count():
    members = current_app.config["MEMBER_DATA"]["members"].values()
    return sum(any(a["status"] == "restricted" for a in m["accounts"]) for m in members)


def _primary_account(member):
    for account in member["accounts"]:
        if account["is_primary"]:
            return account
    return member["accounts"][0]


@member_bp.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    show_popup = random.random() < 0.5
    return render_template(
        "dashboard.html",
        show_popup=show_popup,
        activity=current_activity().snapshot(),
        restricted_members=_restricted_member_count(),
    )


@member_bp.route("/search", methods=["GET"])
@login_required
def search():
    notice = "Select a member before starting a bill payment." if request.args.get("need_member") else None
    return render_template("search.html", notice=notice)


@member_bp.route("/search", methods=["POST"])
@login_required
def search_submit():
    member_id = request.form.get("member_id", "").strip()
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))
    _open_member(member_id)
    return redirect(url_for("member.member_detail", member_id=member_id))


@member_bp.route("/member/<member_id>", methods=["GET"])
@login_required
def member_detail(member_id):
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))
    _open_member(member_id)
    primary_account = _primary_account(member)
    return render_template("member_detail.html", member=member, primary_account=primary_account)


@member_bp.route("/member/<member_id>/edit", methods=["GET"])
@login_required
def member_edit(member_id):
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))
    _open_member(member_id)
    return render_template("member_edit.html", member=member, saved=False)


@member_bp.route("/member/<member_id>/edit", methods=["POST"])
@login_required
def member_edit_submit(member_id):
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))
    _open_member(member_id)

    email = request.form.get("email", member["email"]).strip()
    phone = request.form.get("phone", member["phone"]).strip()
    error = PHONE_ERROR if not PHONE_FORMAT.fullmatch(phone) else (
        EMAIL_ERROR if not EMAIL_FORMAT.fullmatch(email) else None)
    if error is not None:
        # Shown again with what was typed, so it can be corrected; nothing is saved.
        return render_template("member_edit.html", member=member, saved=False, error=error,
                               entered={"email": email, "phone": phone})
    member["email"] = email
    member["phone"] = phone
    return render_template("member_edit.html", member=member, saved=True)


@member_bp.route("/member/<member_id>/accounts", methods=["GET"])
@login_required
def member_accounts(member_id):
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))
    _open_member(member_id)
    return render_template("member_accounts.html", member=member)


@member_bp.route("/member/not-found", methods=["GET"])
@login_required
def not_found():
    return render_template("not_found.html")
