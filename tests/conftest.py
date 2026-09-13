import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from playwright.async_api import async_playwright
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mock_bank"))
from app import create_app  # noqa: E402
import blueprints.member as member_routes  # noqa: E402


@pytest.fixture(scope="session")
def anyio_backend():
    # Session scope lets one event loop, and so one browser, serve every test.
    return "asyncio"


@pytest.fixture(scope="session")
def mock_bank_url():
    # Port 0: the OS picks a free port, so a hand-started bank on 5000 is untouched.
    server = make_server("localhost", 0, create_app(), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{server.server_port}"
    server.shutdown()
    thread.join()


@pytest.fixture(scope="session")
async def browser():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser, mock_bank_url):
    # A fresh context per test: no cookies, so every test starts signed out.
    context = await browser.new_context(base_url=mock_bank_url)
    page = await context.new_page()
    yield page
    await context.close()


@pytest.fixture
def dashboard_popup(monkeypatch):
    """Call with True or False before the dashboard loads to force the popup on or off."""
    def force(shown: bool) -> None:
        monkeypatch.setattr(
            member_routes, "random", SimpleNamespace(random=lambda: 0.0 if shown else 1.0)
        )
    return force
