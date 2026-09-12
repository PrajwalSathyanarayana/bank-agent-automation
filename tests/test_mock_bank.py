import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mock_bank"))
from app import create_app  # noqa: E402


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
    assert resp.headers["Location"] == "/search"


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


# --- D027: homepage + sign off ---

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
