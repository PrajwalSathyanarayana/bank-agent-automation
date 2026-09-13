import pytest

from src.config.env import env
from src.discovery.perception import (
    DESCRIPTION_MAX,
    OUTSIDE_MARKER,
    Box,
    ElementFacts,
    Observation,
    PageElement,
    UnknownElement,
    choose_elements,
    describe,
    describe_options,
    element_kind,
)
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


@pytest.mark.anyio
async def test_harness_browser_uses_the_discovery_window(page):
    await page.goto("/login")
    assert page.viewport_size == {"width": 1280, "height": 800}
    assert await page.evaluate("window.devicePixelRatio") == 1


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


# --- perception: element kinds and descriptions (no browser) ---

VISIBLE = Box(10, 10, 100, 20)
BELOW = Box(10, 900, 100, 20)


def _facts(tag, box=VISIBLE, **fields) -> ElementFacts:
    return ElementFacts(tag=tag, box=box, **fields)


@pytest.mark.parametrize(
    "facts, kind",
    [
        (_facts("a"), "link"),
        (_facts("input", input_type="text"), "text box"),
        (_facts("input"), "text box"),
        (_facts("input", input_type="password"), "password box"),
        (_facts("input", input_type="submit"), "button"),
        (_facts("input", input_type="reset"), "button"),
        (_facts("input", input_type="checkbox"), "checkbox"),
        (_facts("input", input_type="date"), "date input"),
        (_facts("select"), "dropdown"),
        (_facts("textarea"), "text area"),
        (_facts("button"), "button"),
        (_facts("div", role="button"), "button"),
        (_facts("span", role="menuitem"), "menu item"),
        (_facts("td"), "clickable cell"),
        (_facts("div"), "clickable area"),
    ],
)
def test_element_kind(facts, kind):
    assert element_kind(facts) == kind


def test_link_is_described_by_its_own_text():
    assert describe(_facts("a", text="Bill Pay")) == 'link "Bill Pay"'


def test_unnamed_input_uses_its_left_label_and_says_so():
    facts = _facts("input", input_type="text", label="Username:", label_source="left label")
    assert describe(facts) == 'text box, left label "Username:", empty'


def test_text_box_shows_its_current_value():
    facts = _facts("input", input_type="text", label="Amount:", label_source="left label", value="50.0")
    assert describe(facts) == 'text box, left label "Amount:", value "50.0"'


@pytest.mark.parametrize(
    "source, wording",
    [("accessible name", "named"), ("label above", "label above"),
     ("placeholder", "placeholder"), ("title", "title")],
)
def test_label_source_is_worded_for_the_model(source, wording):
    facts = _facts("input", input_type="text", label="Member ID", label_source=source)
    assert describe(facts) == f'text box, {wording} "Member ID", empty'


def test_input_without_any_label_says_no_label():
    assert describe(_facts("input", input_type="text")) == "text box, no label, empty"


def test_password_box_never_shows_a_value_even_if_one_arrives():
    facts = _facts("input", input_type="password", label="Password:", label_source="left label",
                   value="hunter2", filled=True)
    description = describe(facts)
    assert description == 'password box, left label "Password:", filled'
    assert "hunter2" not in description


def test_dropdown_shows_selected_option():
    facts = _facts("select", label="Payee:", label_source="left label",
                   selected="Sunbelt Electric Co", options=("Sunbelt Electric Co",))
    assert describe(facts) == 'dropdown, left label "Payee:", selected "Sunbelt Electric Co"'


def test_dropdown_options_are_capped_with_a_count_of_the_rest():
    facts = _facts("select", options=tuple(f"Payee {n}" for n in range(12)))
    line = describe_options(facts)
    assert line.startswith('options: "Payee 0" | "Payee 1"')
    assert '"Payee 9"' in line and '"Payee 10"' not in line
    assert line.endswith("(+2 more)")


def test_page_text_is_quoted_escaped_and_kept_on_one_line():
    # Page text must not be able to close our quote or add lines of its own.
    facts = _facts("input", input_type="text", label='Say "hi"\nnow', label_source="left label")
    description = describe(facts)
    assert '"Say \\"hi\\" now"' in description
    assert "\n" not in description


def test_description_is_capped_even_with_long_page_text():
    facts = _facts("input", input_type="text", label="A" * 200, label_source="left label", value="B" * 200)
    description = describe(facts)
    assert len(description) <= DESCRIPTION_MAX
    assert description.startswith('text box, left label "AAA')
    assert "…" in description


# --- perception: which elements are listed, and the list text ---

def test_visible_elements_come_first_each_group_in_page_order():
    facts = [_facts("a"), _facts("a", box=BELOW), _facts("a")]
    assert choose_elements(facts, 1280, 800, max_elements=10) == ([0, 2, 1], 0)


def test_element_list_is_capped_with_a_count_of_the_rest():
    facts = [_facts("a"), _facts("a", box=BELOW), _facts("a")]
    assert choose_elements(facts, 1280, 800, max_elements=2) == ([0, 2], 1)


def test_box_above_or_beside_the_window_is_outside():
    assert not Box(10, -50, 100, 20).intersects(1280, 800)
    assert not Box(1300, 10, 100, 20).intersects(1280, 800)
    assert Box(10, 790, 100, 20).intersects(1280, 800)  # partly visible counts as visible


def _element(number, facts, in_viewport=True) -> PageElement:
    # handle is a live browser object; these tests never act, so any placeholder will do.
    return PageElement(number=number, facts=facts, description=describe(facts),
                       in_viewport=in_viewport, handle=object())


def _observation(elements, omitted=0) -> Observation:
    return Observation(url="http://localhost/x", title="X", screenshot=b"", marked_screenshot=b"",
                       elements=elements, omitted_count=omitted)


def test_element_list_text_numbers_marks_outside_and_lists_options():
    payee = _facts("select", label="Payee:", label_source="left label",
                   selected="Sunbelt Electric Co", options=("Sunbelt Electric Co", "Desert Valley Water Utility"))
    text = _observation(
        [_element(1, _facts("a", text="Bill Pay")),
         _element(2, payee),
         _element(3, _facts("a", text="Sign Off", box=BELOW), in_viewport=False)],
        omitted=4,
    ).element_list_text()
    assert text.splitlines() == [
        '[1] link "Bill Pay"',
        '[2] dropdown, left label "Payee:", selected "Sunbelt Electric Co"',
        '    options: "Sunbelt Electric Co" | "Desert Valley Water Utility"',
        f'[3] link "Sign Off"{OUTSIDE_MARKER}',
        "+4 more elements not listed",
    ]


def test_unknown_element_number_raises_a_clear_error():
    observation = _observation([_element(1, _facts("a", text="Bill Pay"))])
    assert observation.element(1).number == 1
    with pytest.raises(UnknownElement, match=r"no element \[7\]"):
        observation.element(7)
