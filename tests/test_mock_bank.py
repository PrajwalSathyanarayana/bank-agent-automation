import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mock_bank"))
from app import check_member_data, create_app, load_member_data  # noqa: E402


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def logged_in_client(client):
    client.post("/login", data={"username": "admin", "password": "admin123"})
    return client


# --- auth.py ---

def test_login_page_renders(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Staff Portal" in resp.data


def test_login_wrong_credentials_shows_inline_error(client):
    resp = client.post("/login", data={"username": "admin", "password": "wrongpass"})
    assert resp.status_code == 200
    assert b"Invalid username or password" in resp.data


def test_login_correct_credentials_redirects_to_dashboard(client):
    resp = client.post("/login", data={"username": "admin", "password": "admin123"})
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/dashboard"


def test_session_timeout_renders_and_clears_session(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    resp = client.get("/session-timeout")
    assert resp.status_code == 200
    assert b"session has expired" in resp.data
    with client.session_transaction() as sess:
        assert "logged_in" not in sess


def test_protected_route_without_session_redirects_to_session_timeout(client):
    resp = client.get("/dashboard", follow_redirects=True)
    assert b"session has expired" in resp.data


# --- member.py ---

def test_dashboard_renders_when_authenticated(logged_in_client):
    resp = logged_in_client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Staff Dashboard" in resp.data


def test_search_page_renders(logged_in_client):
    resp = logged_in_client.get("/search")
    assert resp.status_code == 200
    assert b"Member Search" in resp.data


def test_search_valid_member_id_redirects_to_detail(logged_in_client):
    resp = logged_in_client.post("/search", data={"member_id": "10234"})
    assert resp.headers["Location"] == "/member/10234"


def test_search_invalid_member_id_redirects_to_not_found(logged_in_client):
    resp = logged_in_client.post("/search", data={"member_id": "99999"})
    assert resp.headers["Location"] == "/member/not-found"


def test_member_detail_renders_for_valid_id(logged_in_client):
    resp = logged_in_client.get("/member/10234")
    assert resp.status_code == 200
    assert b"Laura" in resp.data
    assert b"checking" in resp.data


def test_member_detail_invalid_id_redirects_to_not_found(logged_in_client):
    resp = logged_in_client.get("/member/99999")
    assert resp.headers["Location"] == "/member/not-found"


def test_not_found_page_renders_directly(logged_in_client):
    resp = logged_in_client.get("/member/not-found")
    assert resp.status_code == 200
    assert b"No member found" in resp.data


def test_member_edit_form_prefilled(logged_in_client):
    resp = logged_in_client.get("/member/10234/edit")
    assert resp.status_code == 200
    assert b"laura.whitfield@example.com" in resp.data


def test_member_edit_submit_updates_record(logged_in_client):
    resp = logged_in_client.post(
        "/member/10234/edit",
        data={"email": "laura.w.updated@example.com", "phone": "(602) 555-9999"},
    )
    assert resp.status_code == 200
    assert b"Profile updated successfully" in resp.data
    assert b"laura.w.updated@example.com" in resp.data


def test_member_accounts_lists_checking_and_savings(logged_in_client):
    resp = logged_in_client.get("/member/10234/accounts")
    assert resp.status_code == 200
    assert b"CHK-10234-01" in resp.data
    assert b"SAV-10234-01" in resp.data


def test_member_accounts_omits_savings_when_absent(logged_in_client):
    resp = logged_in_client.get("/member/20567/accounts")
    assert b"CHK-20567-01" in resp.data
    assert b"SAV-20567" not in resp.data


# --- billpay.py ---

def test_billpay_without_member_selected_redirects_to_search(logged_in_client):
    resp = logged_in_client.get("/billpay")
    assert resp.headers["Location"] == "/search?need_member=1"


# --- bill pay redirect notice, boot-scoped sessions, breadcrumb link ---

def test_search_explains_why_when_sent_from_billpay(logged_in_client):
    resp = logged_in_client.get("/search?need_member=1")
    assert b"Select a member before starting a bill payment" in resp.data


def test_search_shows_no_notice_normally(logged_in_client):
    resp = logged_in_client.get("/search")
    assert b"Select a member before starting a bill payment" not in resp.data


def test_session_from_previous_server_boot_is_rejected(app, logged_in_client):
    app.config["BOOT_ID"] = "simulated-restart"
    resp = logged_in_client.get("/dashboard", follow_redirects=True)
    assert b"session has expired" in resp.data


def test_root_shows_homepage_for_session_from_previous_boot(app, logged_in_client):
    app.config["BOOT_ID"] = "simulated-restart"
    resp = logged_in_client.get("/")
    assert resp.status_code == 200
    assert b"Welcome to CoreBank Teller" in resp.data
    assert b"Sign Off" not in resp.data


def test_breadcrumb_home_is_a_link(logged_in_client):
    resp = logged_in_client.get("/dashboard")
    assert b'<a href="/">Home</a>' in resp.data


def test_billpay_form_renders_with_default_amount(logged_in_client):
    logged_in_client.get("/member/20567")
    resp = logged_in_client.get("/billpay")
    assert resp.status_code == 200
    assert b"50.0" in resp.data


def test_billpay_invalid_payee_shows_inline_error(logged_in_client):
    logged_in_client.get("/member/20567")
    resp = logged_in_client.post("/billpay", data={"payee_id": "P999", "amount": "50.00"})
    assert b"valid payee" in resp.data


def test_billpay_non_numeric_amount_shows_inline_error(logged_in_client):
    logged_in_client.get("/member/20567")
    resp = logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "abc"})
    assert b"valid number" in resp.data


def test_billpay_zero_amount_shows_inline_error(logged_in_client):
    logged_in_client.get("/member/20567")
    resp = logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "0"})
    assert b"greater than zero" in resp.data


def test_billpay_insufficient_funds_shows_inline_error(logged_in_client):
    logged_in_client.get("/member/20567")  # balance 512.75
    resp = logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "1000.00"})
    assert b"Insufficient funds" in resp.data


def test_billpay_valid_submission_redirects_to_confirm(logged_in_client):
    logged_in_client.get("/member/20567")
    resp = logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "100.00"})
    assert resp.headers["Location"] == "/billpay/confirm"


def test_billpay_confirm_shows_summary_before_confirming(logged_in_client):
    logged_in_client.get("/member/20567")
    logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "100.00"})
    resp = logged_in_client.get("/billpay/confirm")
    assert resp.status_code == 200
    assert b"Confirm Payment" in resp.data
    assert b"Sunbelt Electric" in resp.data


def test_billpay_confirm_submit_deducts_balance(logged_in_client):
    logged_in_client.get("/member/20567")  # balance 512.75
    logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "100.00"})
    resp = logged_in_client.post("/billpay/confirm")
    assert resp.status_code == 200
    assert b"Payment Submitted Successfully" in resp.data
    assert b"412.75" in resp.data


def test_billpay_confirm_without_pending_payment_redirects_to_billpay(logged_in_client):
    resp = logged_in_client.get("/billpay/confirm")
    assert resp.headers["Location"] == "/billpay"


# --- homepage + sign off ---

def test_root_shows_homepage_with_login_button_when_not_logged_in(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Welcome to CoreBank Teller" in resp.data
    assert b'action="/login"' in resp.data


def test_root_redirects_to_dashboard_when_logged_in(logged_in_client):
    resp = logged_in_client.get("/")
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/dashboard"


def test_logout_clears_session_and_redirects_to_login(logged_in_client):
    resp = logged_in_client.get("/logout")
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/login?signed_off=1"
    resp = logged_in_client.get("/dashboard", follow_redirects=True)
    assert b"session has expired" in resp.data


def test_login_page_shows_signed_off_message(client):
    resp = client.get("/login?signed_off=1")
    assert b"You have been signed off" in resp.data


def test_sign_off_link_hidden_when_not_logged_in(client):
    resp = client.get("/login")
    assert b"Sign Off" not in resp.data


def test_sign_off_link_shown_when_logged_in(logged_in_client):
    resp = logged_in_client.get("/dashboard")
    assert b"Sign Off" in resp.data


# --- inactive membership means restricted accounts ---

def _checking_balance(app, member_id):
    return app.config["MEMBER_DATA"]["members"][member_id]["accounts"][0]["balance"]


def test_member_detail_shows_restriction_notice_for_inactive_member(logged_in_client):
    resp = logged_in_client.get("/member/30891")
    assert b"Membership inactive" in resp.data
    assert b"auto loan charged off" in resp.data


def test_member_detail_has_no_notice_for_active_member(logged_in_client):
    resp = logged_in_client.get("/member/10234")
    assert b"Membership inactive" not in resp.data


def test_inactive_member_accounts_show_restricted(logged_in_client):
    resp = logged_in_client.get("/member/30891/accounts")
    assert resp.data.count(b"<td>restricted</td>") == 2
    assert b"<td>active</td>" not in resp.data


def test_billpay_form_replaced_by_notice_for_restricted_member(logged_in_client):
    logged_in_client.get("/member/30891")
    resp = logged_in_client.get("/billpay")
    assert resp.status_code == 200
    assert b"Bill Pay unavailable" in resp.data
    assert b'name="payee_id"' not in resp.data


def test_billpay_submit_refused_for_restricted_member(app, logged_in_client):
    # A direct POST must not get past the block the form page shows.
    logged_in_client.get("/member/30891")
    resp = logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "50.00"})
    assert b"Bill Pay unavailable" in resp.data
    assert _checking_balance(app, "30891") == 130.00
    with logged_in_client.session_transaction() as sess:
        assert "pending_payment" not in sess


def test_pending_payment_dropped_if_account_restricted_before_confirm(app, logged_in_client):
    logged_in_client.get("/member/10234")
    logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "50.00"})
    app.config["MEMBER_DATA"]["members"]["10234"]["accounts"][0]["status"] = "restricted"
    resp = logged_in_client.post("/billpay/confirm")
    assert resp.headers["Location"] == "/billpay"
    assert _checking_balance(app, "10234") == 2450.32
    with logged_in_client.session_transaction() as sess:
        assert "pending_payment" not in sess


# --- live dashboard metrics ---

def _dashboard(client):
    return client.get("/dashboard").data


def test_dashboard_starts_at_zero_and_counts_restricted_members(logged_in_client):
    page = _dashboard(logged_in_client)
    assert b"<td>Members Looked Up</td><td>0</td>" in page
    assert b"<td>Bill Payments Completed</td><td>0</td>" in page
    assert b"<td>Bill Payments Total</td><td>$0.00</td>" in page
    assert b"<td>Members with Restricted Accounts</td><td>1</td>" in page


def test_dashboard_counts_distinct_members_looked_up(logged_in_client):
    logged_in_client.get("/member/10234")
    logged_in_client.get("/member/10234/accounts")
    logged_in_client.get("/member/20567")
    assert b"<td>Members Looked Up</td><td>2</td>" in _dashboard(logged_in_client)


def test_dashboard_counts_completed_payment_and_total(logged_in_client):
    logged_in_client.get("/member/20567")
    logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "100.00"})
    logged_in_client.post("/billpay/confirm")
    page = _dashboard(logged_in_client)
    assert b"<td>Bill Payments Completed</td><td>1</td>" in page
    assert b"<td>Bill Payments Total</td><td>$100.00</td>" in page


def test_payment_not_counted_until_confirmed(logged_in_client):
    logged_in_client.get("/member/20567")
    logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "100.00"})
    assert b"<td>Bill Payments Completed</td><td>0</td>" in _dashboard(logged_in_client)


def test_dashboard_counts_blocked_bill_pay_attempts(logged_in_client):
    logged_in_client.get("/member/30891")
    logged_in_client.get("/billpay")
    logged_in_client.post("/billpay", data={"payee_id": "P001", "amount": "50.00"})
    page = _dashboard(logged_in_client)
    assert b"<td>Bill Pay Attempts Blocked (Restricted)</td><td>2</td>" in page


def test_seed_data_passes_the_startup_check():
    check_member_data(load_member_data())  # should not raise


def _seed_with(edit):
    data = copy.deepcopy(load_member_data())
    edit(data["members"])
    return data


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda m: m["30891"]["accounts"][0].update(status="active"), "is not restricted"),
        (lambda m: m["30891"].pop("inactive_reason"), "inactive_reason"),
        (lambda m: m["10234"].update(inactive_reason="Dormant"), "inactive_reason"),
        (lambda m: m["10234"]["accounts"][0].update(status="closed"), "unknown account status"),
    ],
    ids=["inactive_with_active_account", "inactive_without_reason",
         "active_with_reason", "unknown_status"],
)
def test_startup_check_rejects_contradictory_data(edit, message):
    with pytest.raises(ValueError, match=message):
        check_member_data(_seed_with(edit))
