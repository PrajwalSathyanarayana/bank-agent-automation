import pytest

from src.config.env import env
from src.safety.allowlist import check_domain


async def _sign_in(page) -> None:
    await page.goto("/login")
    await page.fill("input[name='username']", env.mock_bank_username)
    await page.fill("input[name='password']", env.mock_bank_password.get_secret_value())
    await page.click("input[type='submit']")
    await page.wait_for_url("**/dashboard")


# --- test harness: in-process mock bank + shared browser ---

@pytest.mark.anyio
async def test_harness_serves_the_login_page(page):
    response = await page.goto("/login")
    assert response.status == 200
    assert await page.title() == "Sign On - Sunbelt Credit Union"


def test_harness_server_passes_the_allowlist(mock_bank_url):
    check_domain(mock_bank_url)  # should not raise


@pytest.mark.anyio
async def test_each_test_starts_signed_out(page):
    await page.goto("/dashboard")
    assert not page.url.endswith("/dashboard")


@pytest.mark.anyio
async def test_env_credentials_sign_in_to_the_bank(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    assert page.url.endswith("/dashboard")


@pytest.mark.anyio
async def test_dashboard_popup_can_be_forced_on(page, dashboard_popup):
    dashboard_popup(True)
    await _sign_in(page)
    assert await page.locator(".overlay").is_visible()


@pytest.mark.anyio
async def test_dashboard_popup_can_be_forced_off(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    assert await page.locator(".overlay").count() == 0
