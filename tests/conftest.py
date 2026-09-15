import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from playwright.async_api import async_playwright
from werkzeug.serving import make_server

from src.config.settings import settings
from src.surface.browser import launch_args

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mock_bank"))
from app import create_app  # noqa: E402
import blueprints.member as member_routes  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "browser: drives a real Chromium page against the in-process mock bank")
    config.addinivalue_line("markers", "llm: calls the real model; costs money, run only on purpose")


def pytest_collection_modifyitems(config, items):
    # A test that uses a browser page or the bank server is a browser test; no test has
    # to say so itself.
    for item in items:
        if {"page", "browser", "mock_bank_url", "switched_bank"} & set(item.fixturenames):
            item.add_marker(pytest.mark.browser)


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
async def browser(mock_bank_url):
    async with async_playwright() as playwright:
        # The same start-up flags as the discovery browser: the localhost rule avoids
        # Chromium's IPv6-first delay. URLs still say localhost, so the real allowlist
        # is exercised.
        browser = await playwright.chromium.launch(args=launch_args(mock_bank_url))
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser, mock_bank_url):
    # A fresh context per test: no cookies, so every test starts signed out.
    # Same window and scale as discovery, so tests see what the model sees.
    context = await browser.new_context(
        base_url=mock_bank_url,
        viewport={
            "width": settings.discovery_viewport_width,
            "height": settings.discovery_viewport_height,
        },
        device_scale_factor=settings.discovery_device_scale_factor,
    )
    page = await context.new_page()
    yield page
    await context.close()


@pytest.fixture
def switched_bank():
    """Call with test switches (renamed_menu=True, slow_pages_ms=...) to start a separate
    in-process bank beside the normal one; returns its address."""
    servers = []

    def start(**switches) -> str:
        server = make_server("localhost", 0, create_app(**switches), threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://localhost:{server.server_port}"

    yield start
    for server in servers:
        server.shutdown()


@pytest.fixture
def dashboard_popup(monkeypatch):
    """Call with True or False before the dashboard loads to force the popup on or off."""
    def force(shown: bool) -> None:
        monkeypatch.setattr(
            member_routes, "random", SimpleNamespace(random=lambda: 0.0 if shown else 1.0)
        )
    return force
