import random

from flask import Blueprint, current_app, redirect, render_template, request, session, url_for

from .auth import login_required

member_bp = Blueprint("member", __name__)


def _get_member(member_id):
    return current_app.config["MEMBER_DATA"]["members"].get(member_id)


def _primary_account(member):
    for account in member["accounts"]:
        if account["is_primary"]:
            return account
    return member["accounts"][0]


@member_bp.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    show_popup = random.random() < 0.5
    return render_template("dashboard.html", show_popup=show_popup)


@member_bp.route("/search", methods=["GET"])
@login_required
def search():
    return render_template("search.html")


@member_bp.route("/search", methods=["POST"])
@login_required
def search_submit():
    member_id = request.form.get("member_id", "").strip()
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))
    session["current_member_id"] = member_id
    return redirect(url_for("member.member_detail", member_id=member_id))


@member_bp.route("/member/<member_id>", methods=["GET"])
@login_required
def member_detail(member_id):
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))
    session["current_member_id"] = member_id
    primary_account = _primary_account(member)
    return render_template("member_detail.html", member=member, primary_account=primary_account)


@member_bp.route("/member/<member_id>/edit", methods=["GET"])
@login_required
def member_edit(member_id):
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))
    session["current_member_id"] = member_id
    return render_template("member_edit.html", member=member, saved=False)


@member_bp.route("/member/<member_id>/edit", methods=["POST"])
@login_required
def member_edit_submit(member_id):
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))

    member["email"] = request.form.get("email", member["email"])
    member["phone"] = request.form.get("phone", member["phone"])
    session["current_member_id"] = member_id
    return render_template("member_edit.html", member=member, saved=True)


@member_bp.route("/member/<member_id>/accounts", methods=["GET"])
@login_required
def member_accounts(member_id):
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))
    session["current_member_id"] = member_id
    return render_template("member_accounts.html", member=member)


@member_bp.route("/member/not-found", methods=["GET"])
@login_required
def not_found():
    return render_template("not_found.html")
