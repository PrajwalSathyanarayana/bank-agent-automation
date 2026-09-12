from flask import Blueprint, current_app, redirect, render_template, request, session, url_for

from .auth import login_required

billpay_bp = Blueprint("billpay", __name__)

DEFAULT_AMOUNT = 50.00


def _get_member(member_id):
    return current_app.config["MEMBER_DATA"]["members"].get(member_id)


def _get_payees():
    return current_app.config["MEMBER_DATA"]["payees"]


def _get_payee(payee_id):
    for payee in _get_payees():
        if payee["payee_id"] == payee_id:
            return payee
    return None


def _primary_checking_account(member):
    for account in member["accounts"]:
        if account["is_primary"]:
            return account
    return member["accounts"][0]


@billpay_bp.route("/billpay", methods=["GET"])
@login_required
def billpay():
    member_id = session.get("current_member_id")
    if member_id is None:
        return redirect(url_for("member.search"))
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))

    return render_template(
        "billpay.html",
        member=member,
        payees=_get_payees(),
        default_amount=DEFAULT_AMOUNT,
        error=None,
    )


@billpay_bp.route("/billpay", methods=["POST"])
@login_required
def billpay_submit():
    member_id = session.get("current_member_id")
    if member_id is None:
        return redirect(url_for("member.search"))
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))

    payee_id = request.form.get("payee_id", "")
    amount_raw = request.form.get("amount", "")
    payees = _get_payees()

    payee = _get_payee(payee_id)
    if payee is None:
        return render_template(
            "billpay.html", member=member, payees=payees,
            default_amount=amount_raw, error="Please select a valid payee.",
        ), 200

    try:
        amount = float(amount_raw)
    except ValueError:
        return render_template(
            "billpay.html", member=member, payees=payees,
            default_amount=amount_raw, error="Amount must be a valid number.",
        ), 200

    if amount <= 0:
        return render_template(
            "billpay.html", member=member, payees=payees,
            default_amount=amount_raw, error="Amount must be greater than zero.",
        ), 200

    checking = _primary_checking_account(member)
    if amount > checking["balance"]:
        return render_template(
            "billpay.html", member=member, payees=payees,
            default_amount=amount_raw, error="Insufficient funds for this payment amount.",
        ), 200

    session["pending_payment"] = {"payee_id": payee_id, "amount": amount}
    return redirect(url_for("billpay.billpay_confirm"))


@billpay_bp.route("/billpay/confirm", methods=["GET"])
@login_required
def billpay_confirm():
    member_id = session.get("current_member_id")
    pending_payment = session.get("pending_payment")
    if member_id is None or pending_payment is None:
        return redirect(url_for("billpay.billpay"))
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))

    payee = _get_payee(pending_payment["payee_id"])
    return render_template(
        "billpay_confirm.html",
        member=member,
        payee=payee,
        amount=pending_payment["amount"],
        confirmed=False,
    )


@billpay_bp.route("/billpay/confirm", methods=["POST"])
@login_required
def billpay_confirm_submit():
    member_id = session.get("current_member_id")
    pending_payment = session.get("pending_payment")
    if member_id is None or pending_payment is None:
        return redirect(url_for("billpay.billpay"))
    member = _get_member(member_id)
    if member is None:
        return redirect(url_for("member.not_found"))

    payee = _get_payee(pending_payment["payee_id"])
    checking = _primary_checking_account(member)
    checking["balance"] -= pending_payment["amount"]

    session.pop("pending_payment", None)

    return render_template(
        "billpay_confirm.html",
        member=member,
        payee=payee,
        amount=pending_payment["amount"],
        confirmed=True,
        new_balance=checking["balance"],
    )
