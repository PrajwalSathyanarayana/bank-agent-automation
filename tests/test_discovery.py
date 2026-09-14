import asyncio
import dataclasses
import json
import re
from types import SimpleNamespace
import math
import struct
from datetime import datetime, timezone
from io import BytesIO

import pytest
from PIL import Image, ImageFont
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import expect
from pydantic import SecretStr

from src.config.env import configured_credentials, env
from src.config.settings import settings
from src.discovery.browser import (
    ActionFailed,
    BrowserSession,
    action_timeout_ms,
    dismiss_dialogs,
    launch_args,
    number_text,
    placeholder_values,
    select_option,
    type_text,
)
from src.discovery.prompts import (
    ASSERT_FIRST,
    NO_ACTION_REPROMPT,
    STUCK_CATEGORIES,
    SYSTEM_PROMPT,
    goal_message,
    page_blocks,
    progress,
    tool_definitions,
)
from src.discovery.locators import (
    Candidate,
    NoProvenLocator,
    Rejection,
    RunValues,
    Verdict,
    build_candidates,
    derive_locators,
    generate_candidates,
    number_pattern,
    prove,
    scan,
)
from src.discovery import backstop
from src.discovery.agent import DiscoveryRequest, ModelCallFailed, ModelReply, discover, estimated_cost_usd
from src.discovery.artifact_builder import ArtifactContract, UnsignedArtifact, build_and_save, write_artifact
from src.discovery.backstop import (
    AbortCode,
    FieldKind,
    ScanInputs,
    UnclassifiedField,
    abort_error,
    artifact_fields,
    convert,
    find_problems,
    flag_assertions,
    scan_artifact,
)
from src.discovery.recorder import (
    Action,
    AssertionRefused,
    ExtractionRefused,
    Recorder,
    RecordingError,
    TypingRefused,
)
from src.locating.checks import element_wording, shows_phrase
from src.locating.resolver import resolve
from src.observability.logger import RunLogger
from src.safety.classifier import classify
from src.safety.integrity import IntegrityCheckFailed, sign, verify
from src.types.artifact_schema import (
    Artifact,
    ArtifactMetadata,
    CredentialDefinition,
    CredentialKind,
    GlobalAssertion,
    GlobalAssertionType,
    InputParamDefinition,
    KnownOutcome,
    OutcomeSignal,
    OutputParamDefinition,
    OutputType,
    ParamType,
)
from src.types.result_schema import ExecutionStatus, HandoffResolution
from src.types.step_schema import ActionType, CheckpointType, Locator, LocatorType, SafetyTier, Step, StepCheckpoint
from src.discovery.perception import (
    _COLLECTOR_SOURCE,
    _TAG_FONT,
    DESCRIPTION_MAX,
    MARK_COLOURS,
    NO_ELEMENTS,
    OUTSIDE_MARKER,
    TAG_FONT_SIZE,
    TAG_TEXT_COLOUR,
    Box,
    ElementFacts,
    Observation,
    ObservationReleased,
    PageElement,
    UnknownElement,
    _element_indexes,
    _place_tag,
    choose_elements,
    describe,
    describe_options,
    element_kind,
    mark,
    mark_colour,
    observe,
    observing,
)
from src.safety.allowlist import AllowlistViolation, check_domain


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


def test_action_element_also_shows_an_accessible_name_that_differs():
    facts = _facts("a", text="Edit", label="Edit member 10234", label_source="accessible name")
    assert describe(facts) == 'link "Edit", named "Edit member 10234"'


def test_accessible_name_equal_to_the_text_is_shown_once():
    # Equal after collapsing whitespace and ignoring case.
    facts = _facts("button", text="Log  In", label="log in", label_source="accessible name")
    assert describe(facts) == 'button "Log In"'


def test_guessed_label_is_never_added_next_to_visible_text():
    facts = _facts("a", text="Edit", label="Open help", label_source="title")
    assert describe(facts) == 'link "Edit"'


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


def test_empty_list_says_so_in_one_fixed_line():
    assert _observation([]).element_list_text() == NO_ELEMENTS


def test_element_indexes_sort_as_numbers_and_drop_other_keys():
    # Shuffled on purpose: a missing sort could pass by luck if keys arrived in order,
    # and a text sort would put "10" before "2".
    keys = ["10", "length", "2", "0", "11", "1", "3", "9", "4", "8", "5", "7", "6"]
    assert _element_indexes(keys) == list(range(12))


# --- perception: observe() against real and hand-written pages (browser) ---

QUIRKS_DOCTYPE = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">'
# A real 1x1 GIF: a broken or empty image can render at zero size and be dropped as
# invisible before the label rules ever run.
PIXEL = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
FAKE_PASSWORD = "Zq9-not-the-real-password"


async def _observe_html(page, body, body_attrs="", **options) -> Observation:
    # Same doctype as the bank's pages, so hand-written pages also render in quirks mode.
    await page.set_content(f"{QUIRKS_DOCTYPE}<html><body {body_attrs}>{body}</body></html>")
    return await observe(page, **options)


async def _element_with_id(observation, element_id) -> PageElement:
    for element in observation.elements:
        if await element.handle.get_attribute("id") == element_id:
            return element
    raise AssertionError(f"no listed element has id {element_id!r}")


# The collector and Python agree

@pytest.mark.anyio
async def test_collector_keys_match_element_facts_exactly(page):
    await page.set_content(
        f"{QUIRKS_DOCTYPE}<a href='#'>Link</a><input><input type='password'>"
        "<input type='checkbox'><select><option>One</option></select><textarea></textarea>"
    )
    collected = await page.evaluate_handle(_COLLECTOR_SOURCE)
    try:
        raw_facts = await collected.evaluate("result => result.facts")
    finally:
        await collected.dispose()
    fact_keys = {field.name for field in dataclasses.fields(ElementFacts)}
    box_keys = {field.name for field in dataclasses.fields(Box)}
    assert len(raw_facts) == 6
    for raw in raw_facts:
        assert set(raw) == fact_keys
        assert set(raw["box"]) == box_keys


@pytest.mark.anyio
async def test_each_handle_is_the_element_its_facts_describe_past_ten(page):
    links = " ".join(f'<a href="#" id="link-{n}">Link {n}</a>' for n in range(12))
    observation = await _observe_html(page, links)
    assert len(observation.elements) == 12
    for element in observation.elements:
        n = element.facts.text.removeprefix("Link ")
        assert await element.handle.get_attribute("id") == f"link-{n}"


@pytest.mark.anyio
async def test_sign_on_handles_match_their_facts(page):
    await page.goto("/login")
    observation = await observe(page)
    input_types = []
    for element in observation.elements:
        assert await element.handle.evaluate("el => el.tagName.toLowerCase()") == element.facts.tag
        if element.facts.tag == "input":
            assert await element.handle.evaluate("el => el.type") == element.facts.input_type
            input_types.append(element.facts.input_type)
    assert input_types == ["text", "password", "submit"]


# Which elements are candidates

@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param('<a href="#">Target</a>', id="link"),
        pytest.param('<a onclick="go()">Target</a>', id="a with onclick and no href"),
        pytest.param("<input>", id="text box"),
        pytest.param("<button>Target</button>", id="enabled button"),
        pytest.param('<table><tr><td onclick="go()">Target</td></tr></table>', id="td with onclick"),
        pytest.param('<span role="button">Target</span>', id="role button"),
        pytest.param('<span role="link">Target</span>', id="role link"),
        pytest.param('<span role="menuitem">Target</span>', id="role menuitem"),
        pytest.param('<span role="button link">Target</span>', id="first of several roles"),
        pytest.param('<span tabindex="0">Target</span>', id="tabindex 0"),
    ],
)
async def test_candidate_is_listed(page, body):
    observation = await _observe_html(page, body)
    assert len(observation.elements) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body, body_attrs",
    [
        pytest.param("<span>Target</span>", "", id="plain text"),
        pytest.param("<a>Target</a>", "", id="a with no href and no onclick"),
        pytest.param('<span tabindex="-1">Target</span>', "", id="tabindex -1"),
        pytest.param('<span tabindex="abc">Target</span>', "", id="tabindex not a number"),
        pytest.param('<span role="checkbox">Target</span>', "", id="widget role"),
        pytest.param("<span>Target</span>", 'onclick="closeMenus()"', id="onclick on body"),
    ],
)
async def test_non_candidate_is_not_listed(page, body, body_attrs):
    observation = await _observe_html(page, body, body_attrs)
    assert observation.elements == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param("<button disabled>Target</button>", id="button disabled"),
        pytest.param("<fieldset disabled><input></fieldset>", id="input in a disabled fieldset"),
        pytest.param('<span onclick="go()" disabled>Target</span>', id="disabled attribute on a span"),
        pytest.param('<div role="button" aria-disabled="true">Target</div>', id="aria-disabled"),
        pytest.param('<div aria-disabled="true"><button>Target</button></div>',
                     id="inside an aria-disabled container"),
    ],
)
async def test_disabled_element_is_not_listed(page, body):
    observation = await _observe_html(page, body)
    assert observation.elements == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param('<input type="hidden" value="x">', id="hidden input"),
        pytest.param('<div style="display:none"><button>Target</button></div>', id="display none parent"),
        pytest.param('<button style="visibility:hidden">Target</button>', id="visibility hidden"),
        pytest.param('<div style="visibility:hidden"><button>Target</button></div>',
                     id="visibility hidden parent"),
        pytest.param('<div role="button" style="width:0;height:0;overflow:hidden">Target</div>',
                     id="zero size"),
    ],
)
async def test_hidden_element_is_not_listed(page, body):
    observation = await _observe_html(page, body)
    assert observation.elements == []


@pytest.mark.anyio
async def test_plain_text_nav_items_are_not_listed(page, dashboard_popup):
    # The dead nav items are plain spans with no onclick, role or tabindex, so they are
    # never candidates. The disabled filter has its own tests above.
    dashboard_popup(False)
    await _sign_in(page)
    texts = [element.facts.text for element in (await observe(page)).elements]
    assert "Bill Pay" in texts
    for dead in ("Transfers", "Stop Payments", "Reports", "Administration"):
        assert dead not in texts


@pytest.mark.anyio
async def test_both_bill_pay_links_on_member_detail_are_listed(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    bill_pay = [element for element in (await observe(page)).elements if element.facts.text == "Bill Pay"]
    assert len(bill_pay) == 2
    assert bill_pay[0].facts.box != bill_pay[1].facts.box


@pytest.mark.anyio
async def test_off_page_and_scrolled_past_are_judged_in_different_coordinates(page):
    body = ('<a href="#" style="position:absolute;left:-9999px">Skip</a>'
            '<a href="#">Top</a><div style="height:3000px"></div><button>Bottom</button>')
    await page.set_content(f"{QUIRKS_DOCTYPE}<html><body>{body}</body></html>")
    await page.evaluate("window.scrollTo(0, document.scrollingElement.scrollHeight)")
    observation = await observe(page)
    # "Skip" and "Top" both have boxes outside the window, yet get different answers.
    # Whether an element is on the page at all is judged in page coordinates (box plus
    # scroll): "Skip" sits 9999 px left of the page; "Top" is on the page, only scrolled
    # past. Whether it is in the visible area is judged in window coordinates: "Top" is
    # outside it, and the list says so.
    assert observation.element_list_text().splitlines() == [
        '[1] button "Bottom"',
        f'[2] link "Top"{OUTSIDE_MARKER}',
    ]


@pytest.mark.anyio
async def test_clickable_row_and_the_link_inside_it_are_both_listed(page):
    body = '<table><tr onclick="openAccount()"><td>Checking</td><td><a href="#">Edit</a></td></tr></table>'
    observation = await _observe_html(page, body)
    assert [element.description for element in observation.elements] == [
        'clickable area "Checking Edit"',
        'link "Edit"',
    ]


@pytest.mark.anyio
async def test_elements_inside_an_iframe_are_out_of_scope(page):
    observation = await _observe_html(page, '<iframe srcdoc="<button>Inside</button>"></iframe>')
    assert await page.frame_locator("iframe").locator("button").count() == 1  # it is there
    assert observation.elements == []  # but deliberately not collected


@pytest.mark.anyio
async def test_elements_inside_a_shadow_root_are_out_of_scope(page):
    body = '<div><template shadowrootmode="open"><button>Inside</button></template></div>'
    observation = await _observe_html(page, body)
    assert await page.locator("button").count() == 1  # Playwright finds it through the shadow root
    assert observation.elements == []  # but deliberately not collected


# Text and labels

@pytest.mark.anyio
@pytest.mark.parametrize(
    "body, fact",
    [
        pytest.param('<span id="a">Billing</span><span id="b">street</span>'
                     '<input id="t" aria-labelledby="a missing b">', "label",
                     id="aria-labelledby with two ids and a missing one"),
        pytest.param('<span id="h" style="display:none">Close dialog</span>'
                     '<button id="t" aria-labelledby="h">x</button>', "label",
                     id="aria-labelledby to a hidden label"),
        pytest.param('<input id="t" aria-label="Member ID">', "label", id="aria-label"),
        pytest.param('<label for="t">Member ID</label><input id="t">', "label", id="label for"),
        pytest.param('<label>Amount <input id="t" value="50.00"></label>', "label", id="wrapping label"),
        pytest.param('<a href="#" id="t"><span aria-label="Close">×</span></a>', "text",
                     id="child aria-label"),
        pytest.param(f'<a href="#" id="t"><img src="{PIXEL}" width="16" height="16" alt="Print"></a>',
                     "text", id="image alt in a link"),
        pytest.param('<style>.req::before { content: "* "; }</style>'
                     '<label class="req" for="t">Email</label><input id="t">', "label",
                     id="CSS before text"),
        pytest.param('<label for="t">Pay <select><option>weekly</option><option selected>monthly</option>'
                     '</select> from</label><input id="t">', "label", id="label containing a dropdown"),
    ],
)
async def test_names_match_playwrights_accessible_name(page, body, fact):
    element = await _element_with_id(await _observe_html(page, body), "t")
    ours = getattr(element.facts, fact)
    assert ours
    if fact == "label":
        assert element.facts.label_source == "accessible name"
    await expect(page.locator("#t")).to_have_accessible_name(ours)


@pytest.mark.anyio
async def test_password_inside_a_label_is_never_read(page):
    # Deliberately differs from Playwright, whose name here would include the password.
    body = '<label for="t">User <input type="password" value="hunter2"> name</label><input id="t">'
    element = await _element_with_id(await _observe_html(page, body), "t")
    assert element.facts.label == "User name"


@pytest.mark.anyio
async def test_visible_text_inside_a_hidden_parent_is_read(page):
    # Deliberately differs from Playwright ("A"): "B" is on screen, and Chromium's own
    # accessibility tree reads "AB" too.
    body = '<a href="#" id="t">A<span style="visibility:hidden">X<b style="visibility:visible">B</b></span></a>'
    element = await _element_with_id(await _observe_html(page, body), "t")
    assert element.facts.text == "AB"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body, description",
    [
        pytest.param('<table><tr><td>Member ID:</td><td></td><td><input id="t"></td></tr></table>',
                     'text box, left label "Member ID:", empty', id="left label past an empty spacer cell"),
        pytest.param('<table><tr><th>First</th><th>Last</th></tr>'
                     '<tr><td><input></td><td><input id="t"></td></tr></table>',
                     'text box, label above "Last", empty', id="label above from a header row"),
        pytest.param('<table><tr><td><input value="Jane"></td><td><input id="t"></td></tr></table>',
                     "text box, no label, empty", id="stops at a cell holding another field"),
        pytest.param('<table><tr><td colspan="2">Address</td></tr>'
                     '<tr><td><input></td><td><input id="t"></td></tr></table>',
                     'text box, label above "Address", empty', id="cell above spanning two columns"),
        pytest.param('<input id="t" placeholder="Search members">',
                     'text box, placeholder "Search members", empty', id="placeholder"),
        pytest.param('<input id="t" title="Member ID">', 'text box, title "Member ID", empty', id="title"),
    ],
)
async def test_label_rules_for_form_fields(page, body, description):
    element = await _element_with_id(await _observe_html(page, body), "t")
    assert element.description == description


@pytest.mark.anyio
async def test_image_link_in_a_data_row_gets_no_label_from_the_row(page):
    body = (f'<table><tr><td>Checking</td><td>$1,240.00</td><td><a href="#" id="t">'
            f'<img src="{PIXEL}" width="16" height="16"></a></td></tr></table>')
    element = await _element_with_id(await _observe_html(page, body), "t")
    assert element.description == "link, no label"


@pytest.mark.anyio
async def test_sign_on_page_is_labelled_by_its_left_cells(page):
    await page.goto("/login")
    assert (await observe(page)).element_list_text().splitlines() == [
        '[1] link "Home"',
        '[2] text box, left label "Username:", empty',
        '[3] password box, left label "Password:", empty',
        '[4] button "Log In"',
    ]
    # The inputs have no accessible name, so the left-cell rule is what names them.
    for name in ("username", "password"):
        await expect(page.locator(f"input[name='{name}']")).to_have_accessible_name("")


@pytest.mark.anyio
async def test_page_text_with_quotes_and_accents_is_quoted_safely(page):
    body = '<table><tr><td>Payee "Café" Ñame:</td><td><input></td></tr></table>'
    observation = await _observe_html(page, body)
    assert observation.elements[0].description == 'text box, left label "Payee \\"Café\\" Ñame:", empty'


# Facts values

@pytest.mark.anyio
async def test_typed_password_appears_nowhere_in_the_observation(page):
    await page.goto("/login")
    await page.fill("input[name='password']", FAKE_PASSWORD)
    observation = await observe(page)
    password_box = next(element for element in observation.elements if element.facts.input_type == "password")
    assert password_box.facts.filled and password_box.facts.value is None
    assert password_box.description == 'password box, left label "Password:", filled'
    # A dataclass repr leaking into a log line is the realistic failure, so check that too.
    everything = json.dumps([dataclasses.asdict(element.facts) for element in observation.elements])
    everything += repr(observation) + observation.element_list_text()
    assert FAKE_PASSWORD not in everything


@pytest.mark.anyio
async def test_facts_show_current_values_and_button_defaults(page):
    body = (
        "<table>"
        '<tr><td>Payee:</td><td><select><option>Sunbelt Electric Co</option>'
        '<option selected label="Desert Water">Desert Valley Water Utility</option></select></td></tr>'
        '<tr><td>Amount:</td><td><input value="50.00"></td></tr>'
        '<tr><td>Recurring:</td><td><input type="checkbox" checked></td></tr>'
        '<tr><td></td><td><input type="submit"> <input type="reset"></td></tr>'
        "</table>"
    )
    assert (await _observe_html(page, body)).element_list_text().splitlines() == [
        '[1] dropdown, left label "Payee:", selected "Desert Water"',
        '    options: "Sunbelt Electric Co" | "Desert Water"',
        '[2] text box, left label "Amount:", value "50.00"',
        '[3] checkbox, left label "Recurring:", checked',
        '[4] button "Submit"',
        '[5] button "Reset"',
    ]


# observe() as a whole

async def _assert_boxes_line_up_with_the_screenshot(observation) -> None:
    width, height = settings.discovery_viewport_width, settings.discovery_viewport_height
    # A PNG stores its width and height at bytes 16-24; equal to the window proves scale 1.
    assert struct.unpack(">II", observation.screenshot[16:24]) == (width, height)
    checked = 0
    for element in observation.elements:
        box = element.facts.box
        x, y = box.x + box.width / 2, box.y + box.height / 2
        if not (0 <= x < width and 0 <= y < height):
            continue
        # The hit can be a child (the text inside a link), so containment, not identity.
        inside = await element.handle.evaluate(
            "(el, [x, y]) => { const hit = document.elementFromPoint(x, y);"
            " return hit !== null && el.contains(hit); }",
            [x, y],
        )
        assert inside, element.description
        checked += 1
    assert checked > 0


@pytest.mark.anyio
async def test_boxes_line_up_with_the_screenshot_on_sign_on(page):
    await page.goto("/login")
    await _assert_boxes_line_up_with_the_screenshot(await observe(page))


@pytest.mark.anyio
async def test_boxes_line_up_with_the_screenshot_on_the_dashboard(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await _assert_boxes_line_up_with_the_screenshot(await observe(page))


@pytest.mark.anyio
async def test_cap_keeps_visible_elements_first_and_counts_the_rest(page):
    # "Far" comes first in the page but is below the fold, so it is the one cut.
    body = ('<a href="#" style="position:absolute;top:2000px">Far</a>'
            '<a href="#">One</a> <a href="#">Two</a>')
    observation = await _observe_html(page, body, max_elements=2)
    assert len(observation.elements) == 2
    for element in observation.elements:
        assert await element.handle.inner_text() == element.facts.text
    assert observation.omitted_count == 1
    assert observation.element_list_text().splitlines() == [
        '[1] link "One"',
        '[2] link "Two"',
        "+1 more elements not listed",
    ]


@pytest.mark.anyio
async def test_popup_close_button_and_the_page_behind_it_are_listed(page, dashboard_popup):
    dashboard_popup(True)
    await _sign_in(page)
    texts = [element.facts.text for element in (await observe(page)).elements]
    assert "Close" in texts
    assert "Member Search" in texts  # behind the popup, still listed


@pytest.mark.anyio
async def test_page_with_nothing_to_act_on_says_so(page):
    observation = await _observe_html(page, "<p>Scheduled maintenance in progress.</p>")
    assert observation.elements == [] and observation.omitted_count == 0
    assert observation.element_list_text() == NO_ELEMENTS


@pytest.mark.anyio
async def test_same_page_observed_twice_is_numbered_the_same(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    first = await observe(page)
    second = await observe(page)
    assert first.element_list_text() == second.element_list_text()
    assert [element.facts for element in first.elements] == [element.facts for element in second.elements]


@pytest.mark.anyio
async def test_observation_records_the_address_and_title(page):
    await page.goto("/login")
    observation = await observe(page)
    assert observation.url == page.url and observation.url.endswith("/login")
    assert observation.title == "Sign On - Sunbelt Credit Union"


# --- perception: the marked screenshot ---

WHITE = (255, 255, 255)
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 800
WIDE_BOX = Box(100, 100, 200, 40)  # drawn at pixels (100, 100) to (299, 139)


def _blank_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), WHITE).save(buffer, format="PNG")
    return buffer.getvalue()


def _image(png: bytes) -> Image.Image:
    return Image.open(BytesIO(png)).convert("RGB")


def _marked(number, box, in_viewport=True) -> PageElement:
    return _element(number, _facts("a", box=box, text="x"), in_viewport=in_viewport)


def _contrast(first, second) -> float:
    # The accessibility standard's contrast ratio between two colours.
    def luminance(rgb):
        def channel(value):
            value /= 255
            return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        red, green, blue = (channel(value) for value in rgb)
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_marked_copy_is_a_new_png_of_the_same_size():
    clean = _blank_png()
    marked = mark(clean, [_marked(1, WIDE_BOX)])
    assert marked != clean
    with Image.open(BytesIO(marked)) as image:
        assert image.format == "PNG"
        assert image.size == (IMAGE_WIDTH, IMAGE_HEIGHT)


def test_marking_the_same_input_twice_gives_the_same_image():
    elements = [_marked(1, WIDE_BOX), _marked(2, Box(400, 100, 80, 20))]
    assert mark(_blank_png(), elements) == mark(_blank_png(), elements)


def test_outline_frames_the_element_without_filling_it():
    image = _image(mark(_blank_png(), [_marked(1, WIDE_BOX)]))
    assert image.getpixel((100, 130)) == mark_colour(1)  # left edge, below the tag
    assert image.getpixel((200, 139)) == mark_colour(1)  # bottom edge
    assert image.getpixel((200, 120)) == WHITE  # centre


def _tag_pixels(image, label, area) -> list:
    left, top, right, bottom = _place_tag(label, area, [], IMAGE_WIDTH, IMAGE_HEIGHT)
    return list(image.crop((left, top, right + 1, bottom + 1)).get_flattened_data())


def test_tag_holds_the_element_colour_and_white_digits():
    image = _image(mark(_blank_png(), [_marked(1, WIDE_BOX)]))
    pixels = _tag_pixels(image, "1", (100, 100, 299, 139))
    assert mark_colour(1) in pixels
    # Digits are anti-aliased, so look for pixels that are nearly white.
    assert max(min(pixel) for pixel in pixels) >= 230


def test_element_outside_the_visible_area_leaves_the_image_unchanged():
    image = _image(mark(_blank_png(), [_marked(1, BELOW, in_viewport=False)]))
    assert image.getcolors() == [(IMAGE_WIDTH * IMAGE_HEIGHT, WHITE)]


def test_outlines_never_cross_a_number():
    # Element 2's left edge (x = 105) runs straight through element 1's tag.
    elements = [_marked(1, WIDE_BOX), _marked(2, Box(105, 50, 100, 100))]
    image = _image(mark(_blank_png(), elements))
    assert mark_colour(2) not in _tag_pixels(image, "1", (100, 100, 299, 139))
    assert image.getpixel((105, 130)) == mark_colour(2)  # the outline is there outside the tag


def test_mark_colours_cycle_and_neighbours_differ():
    assert mark_colour(len(MARK_COLOURS) + 1) == mark_colour(1)
    assert all(mark_colour(n) != mark_colour(n + 1) for n in range(1, 30))


@pytest.mark.parametrize("colour", MARK_COLOURS)
def test_white_digits_are_readable_on_every_mark_colour(colour):
    assert _contrast(colour, TAG_TEXT_COLOUR) >= 4.5


def test_tag_moves_right_past_a_tag_in_the_way():
    first = _place_tag("1", (100, 100, 299, 139), [], IMAGE_WIDTH, IMAGE_HEIGHT)
    second = _place_tag("2", (100, 100, 299, 139), [first], IMAGE_WIDTH, IMAGE_HEIGHT)
    assert second[0] == first[2] + 1
    assert second[1] == first[1]


def test_tag_with_no_room_to_move_right_stays_at_its_corner():
    in_the_way = (1250, 100, 1279, 112)
    tag = _place_tag("1", (1270, 100, 1279, 139), [in_the_way], IMAGE_WIDTH, IMAGE_HEIGHT)
    assert tag[2] == IMAGE_WIDTH - 1  # as far right as the image allows, not beyond it
    assert tag[1] == 100


def test_element_cut_off_at_the_top_gets_its_tag_on_the_visible_part():
    image = _image(mark(_blank_png(), [_marked(1, Box(100, -10, 200, 40))]))
    assert image.getpixel((100, 0)) == mark_colour(1)


def test_tag_of_an_element_cut_off_at_the_bottom_stays_inside_the_image():
    tag = _place_tag("1", (100, 795, 299, IMAGE_HEIGHT - 1), [], IMAGE_WIDTH, IMAGE_HEIGHT)
    assert tag[3] == IMAGE_HEIGHT - 1
    assert tag[1] < 795


def test_tag_font_is_pillows_scalable_font():
    assert isinstance(_TAG_FONT, ImageFont.FreeTypeFont)
    assert _TAG_FONT.size == TAG_FONT_SIZE


def _bottom_edge_pixel(image, box):
    # The middle of the element's bottom edge, which the outline always covers.
    return image.getpixel((int(box.x + box.width / 2), math.ceil(box.y + box.height) - 1))


@pytest.mark.anyio
async def test_marked_copy_outlines_the_password_box_and_the_clean_copy_does_not(page):
    await page.goto("/login")
    observation = await observe(page)
    password_box = next(element for element in observation.elements if element.facts.input_type == "password")
    colour = mark_colour(password_box.number)
    assert _bottom_edge_pixel(_image(observation.marked_screenshot), password_box.facts.box) == colour
    assert _bottom_edge_pixel(_image(observation.screenshot), password_box.facts.box) != colour


@pytest.mark.anyio
async def test_both_bill_pay_links_are_outlined_in_their_own_colours(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    observation = await observe(page)
    image = _image(observation.marked_screenshot)
    bill_pay = [element for element in observation.elements if element.facts.text == "Bill Pay"]
    assert len(bill_pay) == 2
    for element in bill_pay:
        assert _bottom_edge_pixel(image, element.facts.box) == mark_colour(element.number)
    assert mark_colour(bill_pay[0].number) != mark_colour(bill_pay[1].number)


# --- perception: releasing element references ---

class _TurnFailed(Exception):
    pass


@pytest.mark.anyio
async def test_observation_is_released_when_the_block_ends(page):
    await page.goto("/login")
    async with observing(page) as observation:
        assert observation.element(1).number == 1
    with pytest.raises(ObservationReleased, match="replaced; use the latest one"):
        observation.element(1)


@pytest.mark.anyio
async def test_observation_is_released_even_when_the_turn_fails(page):
    await page.goto("/login")
    with pytest.raises(_TurnFailed):  # the turn's own error comes out unchanged
        async with observing(page) as observation:
            raise _TurnFailed()
    with pytest.raises(ObservationReleased):
        observation.element(1)


@pytest.mark.anyio
async def test_released_references_are_let_go_in_the_browser(page):
    # Not just a flag: the browser-side reference is gone, so acting on it fails.
    await page.goto("/login")
    async with observing(page) as observation:
        username_box = observation.element(2).handle
    with pytest.raises(PlaywrightError):
        await username_box.fill("x", timeout=2000)


@pytest.mark.anyio
async def test_releasing_twice_is_harmless(page):
    await page.goto("/login")
    observation = await observe(page)
    await observation.release()
    await observation.release()


@pytest.mark.anyio
async def test_releasing_after_the_page_has_moved_on_is_harmless(page):
    # Pins Playwright's behaviour: releasing raises nothing once the page has changed,
    # so a release at the end of a turn that navigated can never hide an error.
    await page.goto("/login")
    observation = await observe(page)
    await page.goto("/")
    await observation.release()
    with pytest.raises(ObservationReleased):
        observation.element(1)


# --- locators: generating candidates ---

def _parts(tag="a", input_type="", name=None, value=None, href=None, text="", scopes=()):
    # Raw parts as the in-page script returns them, with one full-page position path.
    return {
        "tag": tag, "type": input_type, "name": name, "value": value, "href": href, "text": text,
        "scopes": [{"selector": selector, "token": token} for selector, token in scopes],
        "positions": [{"path": f"html > body:nth-of-type(1) > {tag}:nth-of-type(1)", "token": None}],
    }


def _kinds(candidates) -> list[str]:
    return [candidate.kind for candidate in candidates]


@pytest.mark.anyio
async def test_form_field_candidates_come_in_the_legacy_aware_order(page):
    await page.goto("/login")
    observation = await observe(page)
    username_box = next(element for element in observation.elements if element.facts.label == "Username:")
    kinds = _kinds(await generate_candidates(username_box))
    assert kinds[:2] == ["name", "label"]
    assert set(kinds[2:]) == {"scoped", "position"}
    first_position = kinds.index("position")
    assert all(kind == "position" for kind in kinds[first_position:])


def test_link_text_comes_before_its_address():
    facts = _facts("a", text="View All Accounts")
    parts = _parts(href="/member/10234/accounts", text="View All Accounts")
    assert _kinds(build_candidates(facts, parts)) == ["text", "address", "position"]


def test_button_value_is_its_text_and_a_selector_only_when_scoped():
    facts = _facts("input", input_type="submit", text="Log In")
    parts = _parts(tag="input", input_type="submit", value="Log In", scopes=[("table.form", "form")])
    assert [(candidate.kind, candidate.value) for candidate in build_candidates(facts, parts)] == [
        ("text", "Log In"),
        ("scoped", 'table.form input[type="submit"][value="Log In"]'),
        ("position", "html > body:nth-of-type(1) > input:nth-of-type(1)"),
    ]


@pytest.mark.parametrize(
    "href",
    [pytest.param("#", id="hash"), pytest.param("javascript:void(0)", id="javascript"),
     pytest.param("   ", id="blank")],
)
def test_link_that_goes_nowhere_gets_no_address_candidate(href):
    candidates = build_candidates(_facts("a", text="Go"), _parts(href=href, text="Go"))
    assert "address" not in _kinds(candidates)


@pytest.mark.parametrize(
    "label, expected",
    [
        pytest.param('Payee "A":', "='Payee \"A\":']", id="double quote inside: single-quoted"),
        pytest.param("Payee's name:", "=\"Payee's name:\"]", id="single quote inside: double-quoted"),
        pytest.param("It's \"x\":", None, id="both quotes: no label candidate"),
    ],
)
def test_label_xpath_quotes_what_it_can_and_skips_the_rest(label, expected):
    facts = _facts("input", input_type="text", label=label, label_source="left label")
    candidates = build_candidates(facts, _parts(tag="input", input_type="text"))
    labels = [candidate.value for candidate in candidates if candidate.kind == "label"]
    if expected is None:
        assert labels == []
    else:
        assert len(labels) == 1 and expected in labels[0]


@pytest.mark.parametrize(
    "source, listed",
    [pytest.param("accessible name", True, id="real accessible name"),
     pytest.param("left label", False, id="left label"),
     pytest.param("placeholder", False, id="placeholder")],
)
def test_accessible_name_candidate_only_for_a_real_accessible_name(source, listed):
    facts = _facts("input", input_type="text", label="Member ID", label_source=source)
    candidates = build_candidates(facts, _parts(tag="input", input_type="text"))
    assert ("accessible name" in _kinds(candidates)) is listed


@pytest.mark.anyio
async def test_position_paths_start_at_the_nearest_unique_container(page):
    body = ('<div class="row"><a href="#">One</a></div>'
            '<div class="row"><div class="box"><a id="t" href="#">Two</a></div></div>')
    element = await _element_with_id(await _observe_html(page, body), "t")
    paths = [candidate.value for candidate in await generate_candidates(element) if candidate.kind == "position"]
    assert paths[0] == "div.box > a:nth-of-type(1)"
    assert paths[-1].startswith("html > ")
    assert not any(path.startswith("div.row") for path in paths)  # on the page twice, so never an anchor


# --- locators: the data scan ---

LOCATOR_RUN = RunValues(
    text_inputs={"member_id": "10234", "payee_name": "Sunbelt Electric Co"},
    number_inputs={"amount": 50.0},
    username="admin",
    secrets={"bank_password": SecretStr(FAKE_PASSWORD)},
)


def _text_candidate(text) -> Candidate:
    return Candidate("text", LocatorType.TEXT_CONTENT, text, data=(text,))


def _address_candidate(address) -> Candidate:
    return Candidate("address", LocatorType.CSS, "", address=address)


@pytest.mark.parametrize(
    "address, stored",
    [
        pytest.param("/member/10234/accounts", 'a[href="/member/{member_id}/accounts"]',
                     id="whole segment becomes a placeholder"),
        pytest.param("/member/view?id=10234", None, id="query parameter is discarded"),
        pytest.param("/member/10234x/accounts", None, id="part of a segment is discarded"),
    ],
)
def test_input_in_an_address_becomes_a_placeholder_only_as_a_whole_segment(address, stored):
    assert scan(_address_candidate(address), LOCATOR_RUN).stored_value == stored


def test_segment_equal_to_two_inputs_is_discarded():
    run = RunValues(text_inputs={"member_id": "10234", "account_id": "10234"})
    outcome = scan(_address_candidate("/member/10234/accounts"), run)
    assert outcome.stored_value is None and "two inputs" in outcome.reason


def test_text_input_anywhere_in_any_case_discards_the_candidate():
    outcome = scan(_text_candidate("Pay SUNBELT ELECTRIC CO"), LOCATOR_RUN)
    assert outcome.stored_value is None and "payee_name" in outcome.reason


@pytest.mark.parametrize(
    "number, text, matches",
    [
        pytest.param(50.0, "Pay $50.00", True, id="50 as $50.00"),
        pytest.param(50.0, "page 50", True, id="50 as 50"),
        pytest.param(50.0, "50.0 due", True, id="50 as 50.0"),
        pytest.param(1240.5, "Balance 1,240.50", True, id="thousands separator"),
        pytest.param(50.0, "Top 150", False, id="not inside 150"),
        pytest.param(50.0, "50.75", False, id="not inside 50.75"),
        pytest.param(50.75, "50.8", False, id="never rounded"),
    ],
)
def test_number_is_matched_in_its_common_forms_as_a_whole_number(number, text, matches):
    assert bool(number_pattern(number).search(text)) is matches


@pytest.mark.parametrize(
    "candidate, kept",
    [
        pytest.param(_address_candidate("/teller/Admin/profile"), False, id="a whole word in any case"),
        pytest.param(_text_candidate("Administration"), True, id="not inside a longer word"),
    ],
)
def test_username_is_matched_as_a_whole_word(candidate, kept):
    assert (scan(candidate, LOCATOR_RUN).stored_value is not None) is kept


def test_secret_discards_the_candidate_and_is_named_never_shown():
    outcome = scan(_text_candidate(f"token {FAKE_PASSWORD}"), LOCATOR_RUN)
    assert outcome.stored_value is None
    assert outcome.secret == "bank_password"
    assert FAKE_PASSWORD not in repr(outcome)


def test_selector_structure_is_never_scanned():
    # The 50 in nth-of-type(50) is not an amount of 50.
    candidate = Candidate("position", LocatorType.CSS, "div.actions > a:nth-of-type(50)", data=("actions",))
    assert scan(candidate, LOCATOR_RUN).stored_value == "div.actions > a:nth-of-type(50)"


@pytest.mark.parametrize(
    "candidate, stored",
    [
        pytest.param(_text_candidate("Braces {x}"), "Braces {{x}}", id="text"),
        pytest.param(_address_candidate("/p/{x}/10234"), 'a[href="/p/{{x}}/{member_id}"]',
                     id="address beside a placeholder"),
    ],
)
def test_literal_braces_are_doubled(candidate, stored):
    assert scan(candidate, LOCATOR_RUN).stored_value == stored


# --- locators: the proof ---

PROOF_PAGE = ('<div class="nav"><a id="t" href="/billpay">Bill Pay</a></div>'
              '<div class="actions"><a id="other" href="/billpay">Bill Pay</a></div>')


@pytest.mark.anyio
@pytest.mark.parametrize(
    "locator, verdict",
    [
        pytest.param(Locator(type=LocatorType.CSS, value='div.nav a[href="/billpay"]', priority=0),
                     Verdict.PROVEN, id="proven"),
        pytest.param(Locator(type=LocatorType.TEXT_CONTENT, value="Bill Pay", priority=0),
                     Verdict.SEVERAL, id="several"),
        pytest.param(Locator(type=LocatorType.CSS, value='a[href="/nowhere"]', priority=0),
                     Verdict.NO_MATCH, id="no match"),
        pytest.param(Locator(type=LocatorType.CSS, value="div.actions a", priority=0),
                     Verdict.OTHER_ELEMENT, id="other element"),
        pytest.param(Locator(type=LocatorType.CSS, value='a[href="/{member_id}"]', priority=0),
                     Verdict.UNFILLABLE, id="unfillable"),
    ],
)
async def test_proof_verdicts(page, locator, verdict):
    await page.set_content(f"{QUIRKS_DOCTYPE}<html><body>{PROOF_PAGE}</body></html>")
    target = await page.query_selector("#t")
    assert await prove(page, target, locator, {}) is verdict


@pytest.mark.anyio
async def test_hidden_duplicate_counts_because_replay_would_see_it(page):
    await page.set_content(f'{QUIRKS_DOCTYPE}<html><body><a id="t" href="/x">Go</a>'
                           '<a href="/x" style="display:none">Go</a></body></html>')
    target = await page.query_selector("#t")
    locator = Locator(type=LocatorType.CSS, value='a[href="/x"]', priority=0)
    assert await prove(page, target, locator, {}) is Verdict.SEVERAL


# --- locators: the whole pipeline ---

def _bank_run(member_id="10234") -> RunValues:
    return RunValues(
        text_inputs={"member_id": member_id, "payee_name": "Sunbelt Electric Co"},
        number_inputs={"amount": 50.0},
        username=env.mock_bank_username,
        secrets={"bank_password": env.mock_bank_password},
    )


@pytest.mark.anyio
async def test_username_box_gets_name_label_and_position(page):
    await page.goto("/login")
    observation = await observe(page)
    username_box = next(element for element in observation.elements if element.facts.label == "Username:")
    derived = await derive_locators(page, username_box, _bank_run())
    assert derived.kinds == ["name", "label", "position"]
    assert [locator.priority for locator in derived.locators] == [0, 1, 2]
    assert derived.locators[0].value == 'input[name="username"]'
    assert not derived.weak


@pytest.mark.anyio
async def test_each_bill_pay_link_gets_three_locators_that_never_find_the_other(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    observation = await observe(page)
    first, second = [element for element in observation.elements if element.facts.text == "Bill Pay"]
    run = _bank_run()
    for element, other in ((first, second), (second, first)):
        derived = await derive_locators(page, element, run)
        assert len(derived.locators) == 3
        assert derived.kinds[0] == "scoped"
        assert Rejection("text", Verdict.SEVERAL.value) in derived.rejected
        assert Rejection("address", Verdict.SEVERAL.value) in derived.rejected
        for locator in derived.locators:
            assert await prove(page, other.handle, locator, run.text_inputs) is Verdict.OTHER_ELEMENT


@pytest.mark.anyio
async def test_second_pass_fills_the_third_slot_with_a_set_aside_variant(page):
    await page.goto("/login")
    observation = await observe(page)
    log_in = next(element for element in observation.elements if element.facts.text == "Log In")
    derived = await derive_locators(page, log_in, _bank_run())
    assert derived.kinds == ["text", "position", "scoped"]
    assert derived.locators[2].value == 'table.form input[type="submit"][value="Log In"]'


@pytest.mark.anyio
async def test_only_a_position_locator_is_flagged_weak(page):
    body = (f'<table><tr><td>Checking</td><td>$1,240.00</td><td><a href="#" id="t">'
            f'<img src="{PIXEL}" width="16" height="16"></a></td></tr></table>')
    element = await _element_with_id(await _observe_html(page, body), "t")
    derived = await derive_locators(page, element, _bank_run())
    assert derived.kinds == ["position"]
    assert derived.weak


@pytest.mark.anyio
async def test_secret_on_the_page_is_reported_by_name_and_never_shown(page):
    run = RunValues(secrets={"bank_password": SecretStr(FAKE_PASSWORD)})
    body = f'<a href="/reset?token={FAKE_PASSWORD}" id="t">Reset</a>'
    element = await _element_with_id(await _observe_html(page, body), "t")
    derived = await derive_locators(page, element, run)
    assert derived.secrets_found == ["bank_password"]
    assert derived.kinds[0] == "text"  # the element is still recorded, without the secret
    assert FAKE_PASSWORD not in repr(derived)


@pytest.mark.anyio
async def test_stored_locators_find_the_same_link_for_another_member(page, dashboard_popup):
    # The end-to-end check that no member data is baked into a saved locator.
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    observation = await observe(page)
    view_all = next(element for element in observation.elements if element.facts.text == "View All Accounts")
    derived = await derive_locators(page, view_all, _bank_run("10234"))
    assert 'a[href="/member/{member_id}/accounts"]' in [locator.value for locator in derived.locators]
    await page.goto("/member/40412")
    for locator in derived.locators:
        found = resolve(page, locator, {"member_id": "40412"})
        hrefs = await found.evaluate_all("elements => elements.map(element => element.getAttribute('href'))")
        assert hrefs == ["/member/40412/accounts"], locator.value


@pytest.mark.anyio
async def test_element_gone_before_deriving_raises(page):
    element = await _element_with_id(await _observe_html(page, '<a id="t" href="/x">Go</a>'), "t")
    await page.evaluate("document.getElementById('t').remove()")
    with pytest.raises(NoProvenLocator):
        await derive_locators(page, element, RunValues())


@pytest.mark.anyio
async def test_same_page_derives_the_same_locators_twice(page):
    await page.goto("/login")
    observation = await observe(page)
    username_box = next(element for element in observation.elements if element.facts.label == "Username:")
    assert await derive_locators(page, username_box, _bank_run()) == await derive_locators(
        page, username_box, _bank_run()
    )


# --- safety tier on the real confirm page (regression) ---

@pytest.mark.anyio
@pytest.mark.parametrize(
    "description, tier",
    [pytest.param('button "Confirm Payment"', SafetyTier.IRREVERSIBLE, id="confirm payment"),
     pytest.param('link "Cancel"', SafetyTier.SAFE, id="cancel")],
)
async def test_confirm_payment_is_irreversible_whatever_the_model_calls_it(page, dashboard_popup, description, tier):
    # The page repeats "Confirm Payment" in its panel title, so the button's text locator
    # is rejected; with a neutral reason, only the button's own wording names it.
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    await page.goto("/billpay")
    await page.click("input[type='submit']")
    await page.wait_for_url("**/billpay/confirm")
    observation = await observe(page)
    element = next(element for element in observation.elements if element.description == description)
    derived = await derive_locators(page, element, _bank_run())
    step = Step(sequence_index=1, action=ActionType.CLICK, description="Submit it", locators=derived.locators)
    wording = await element_wording(element.handle)
    assert classify(step, page.url, element_wording=wording) == tier


# --- recorder ---

@pytest.fixture
def run_logger(tmp_path, monkeypatch) -> RunLogger:
    # Each test logs to its own temporary folder, never the project's evidence folder.
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    return RunLogger("DISCOVERY", capability="recorder_test")


@pytest.fixture
def recorder(run_logger) -> Recorder:
    return Recorder(run_logger)


def _log_lines(logger) -> list[dict]:
    return [json.loads(line) for line in logger.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _checks(step) -> list[tuple]:
    return [(checkpoint.type, checkpoint.expected_value) for checkpoint in step.checkpoints]


async def _start(recorder, page, path="/login", run=None):
    await page.goto(path)
    return await recorder.commit(recorder.draft_start(page.url), page, run or _bank_run(), derived=None)


async def _start_on_html(recorder, page, body):
    await page.set_content(f"{QUIRKS_DOCTYPE}<html><body>{body}</body></html>")
    return await recorder.commit(recorder.draft_start(page.url), page, _bank_run(), derived=None)


async def _draft_on(recorder, page, description_start, action):
    observation = await observe(page)
    element = next(element for element in observation.elements if element.description.startswith(description_start))
    run = _bank_run()
    derived = await derive_locators(page, element, run)
    step = await recorder.draft_step(action, element.handle, derived, page.url, run=run)
    return element, derived, step


async def _open_confirm_page(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    await page.goto("/billpay")
    await page.click("input[type='submit']")
    await page.wait_for_url("**/billpay/confirm")


PAYEE_SELECT = ('<table><tr><td>Payee:</td><td><select name="payee_id">'
                '<option value="P001">Sunbelt Electric Co</option><option value="P002">Desert Water</option>'
                '</select></td></tr></table>')


# Drafting steps

@pytest.mark.anyio
async def test_start_step_is_a_safe_navigate_with_no_locators(page, recorder):
    await page.goto("/login")
    step = recorder.draft_start(page.url)
    assert (step.sequence_index, step.action, step.locators, step.safety_tier) == (
        0, ActionType.NAVIGATE, [], SafetyTier.SAFE)


@pytest.mark.anyio
async def test_start_step_can_only_be_the_first(page, recorder):
    await _start(recorder, page)
    with pytest.raises(RecordingError, match="only be the first step"):
        recorder.draft_start(page.url)


@pytest.mark.anyio
async def test_action_before_the_start_step_is_refused(page, recorder):
    await page.goto("/login")
    with pytest.raises(RecordingError, match="start step must be recorded"):
        await _draft_on(recorder, page, 'button "Log In"', Action(ActionType.CLICK, "Sign in"))


@pytest.mark.anyio
async def test_action_the_model_cannot_take_is_refused(page, recorder):
    await _start(recorder, page)
    with pytest.raises(RecordingError, match="not one of the model's recorded actions"):
        await _draft_on(recorder, page, 'button "Log In"', Action(ActionType.NAVIGATE, "Open a page"))


@pytest.mark.anyio
async def test_typed_text_is_stored_as_written(page, recorder):
    await _start(recorder, page)
    _, _, step = await _draft_on(recorder, page, 'text box, left label "Username:"',
                                 Action(ActionType.TYPE, "Enter the username", "{credential:bank_username}"))
    assert step.input_value == "{credential:bank_username}"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "label, option_value",
    [
        pytest.param("Desert Water", "P002", id="fixed choice keeps its hidden value"),
        pytest.param("{payee_name}", None, id="choice from an input keeps none"),
        pytest.param("Not a payee", None, id="label not in the dropdown keeps none"),
    ],
)
async def test_dropdown_keeps_a_hidden_value_only_for_a_fixed_choice(page, recorder, label, option_value):
    await _start_on_html(recorder, page, PAYEE_SELECT)
    _, _, step = await _draft_on(recorder, page, "dropdown", Action(ActionType.SELECT, "Choose the payee", label))
    assert (step.input_value, step.option_value) == (label, option_value)


@pytest.mark.anyio
async def test_extract_keeps_its_output_name_and_no_typed_value(page, recorder):
    await _start_on_html(recorder, page, PAYEE_SELECT)
    _, _, step = await _draft_on(recorder, page, "dropdown",
                                 Action(ActionType.EXTRACT_TEXT, "Read the payee", output_key="payee_shown"))
    assert (step.action, step.output_key, step.input_value) == (ActionType.EXTRACT_TEXT, "payee_shown", None)


@pytest.mark.anyio
async def test_empty_reason_is_recorded_as_missing_never_invented(page, recorder):
    await _start_on_html(recorder, page, '<a href="/x">Go</a>')
    _, _, step = await _draft_on(recorder, page, 'link "Go"', Action(ActionType.CLICK, "   "))
    assert step.description == "click (the model gave no reason)"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "description, tier",
    [pytest.param('button "Confirm Payment"', SafetyTier.IRREVERSIBLE, id="confirm payment"),
     pytest.param('link "Cancel"', SafetyTier.SAFE, id="cancel")],
)
async def test_recorder_marks_confirm_payment_irreversible_whatever_the_reason_says(
    page, recorder, dashboard_popup, description, tier
):
    await _open_confirm_page(page, dashboard_popup)
    await recorder.commit(recorder.draft_start(page.url), page, _bank_run(), derived=None)
    _, _, step = await _draft_on(recorder, page, description, Action(ActionType.CLICK, "Submit it"))
    assert step.safety_tier == tier


# Committing steps

@pytest.mark.anyio
async def test_start_step_checks_the_page_it_opened(page, recorder):
    committed = await _start(recorder, page)
    assert _checks(committed.step) == [
        (CheckpointType.PAGE_PATH, "/login"),
        (CheckpointType.PAGE_TITLE, "Sign On - Sunbelt Credit Union"),
    ]


@pytest.mark.anyio
async def test_click_checks_where_the_page_landed(page, recorder, dashboard_popup):
    dashboard_popup(False)
    await _start(recorder, page)
    await page.fill("input[name='username']", env.mock_bank_username)
    await page.fill("input[name='password']", env.mock_bank_password.get_secret_value())
    element, derived, step = await _draft_on(recorder, page, 'button "Log In"', Action(ActionType.CLICK, "Sign in"))
    await element.handle.click()
    await page.wait_for_url("**/dashboard")
    committed = await recorder.commit(step, page, _bank_run(), derived=derived)
    assert _checks(committed.step) == [
        (CheckpointType.PAGE_PATH, "/dashboard"),
        (CheckpointType.PAGE_TITLE, "Dashboard - Sunbelt Credit Union"),
    ]


@pytest.mark.anyio
async def test_member_in_the_landing_path_becomes_a_placeholder(page, recorder, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await _start(recorder, page, "/member/10234")
    element, derived, step = await _draft_on(recorder, page, 'link "View All Accounts"',
                                             Action(ActionType.CLICK, "Open the accounts"))
    await element.handle.click()
    await page.wait_for_url("**/member/10234/accounts")
    committed = await recorder.commit(step, page, _bank_run(), derived=derived)
    assert (CheckpointType.PAGE_PATH, "/member/{member_id}/accounts") in _checks(committed.step)


@pytest.mark.anyio
async def test_title_carrying_an_input_is_left_out(page, recorder):
    await page.set_content(f"{QUIRKS_DOCTYPE}<html><head><title>Member 10234</title></head>"
                           "<body><p>Details</p></body></html>")
    assert await page.title() == "Member 10234"  # the title is really there to be checked
    committed = await recorder.commit(recorder.draft_start(page.url), page, _bank_run(), derived=None)
    assert _checks(committed.step) == []


@pytest.mark.anyio
async def test_path_carrying_an_input_inside_a_segment_is_left_out(page, recorder):
    # /member-10234 can't become a placeholder (not a whole segment), so the check goes.
    committed = await _start(recorder, page, "/member-10234")
    assert _checks(committed.step) == [(CheckpointType.PAGE_TITLE, "404 Not Found")]


@pytest.mark.anyio
async def test_secret_in_the_address_is_left_out_and_warned_about(page, recorder, run_logger):
    run = RunValues(secrets={"bank_password": SecretStr(FAKE_PASSWORD)})
    committed = await _start(recorder, page, f"/reset/{FAKE_PASSWORD}", run=run)
    assert all(kind != CheckpointType.PAGE_PATH for kind, _ in _checks(committed.step))
    assert committed.secrets_found == ["bank_password"]
    warning = next(line for line in _log_lines(run_logger) if line["event_type"] == "SECRET_ON_PAGE")
    assert (warning["secrets"], warning["found_in"]) == (["bank_password"], "page address or title")
    assert FAKE_PASSWORD not in run_logger.log_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_next_step_check_goes_on_the_previous_step_never_the_last(page, recorder):
    await _start_on_html(recorder, page, '<a href="#one">One</a> <a href="#two">Two</a>')
    for name in ("One", "Two"):
        _, derived, step = await _draft_on(recorder, page, f'link "{name}"', Action(ActionType.CLICK, f"Open {name}"))
        await recorder.commit(step, page, _bank_run(), derived=derived)
    next_checks = [
        sum(checkpoint.type == CheckpointType.NEXT_STEP_TARGET for checkpoint in step.checkpoints)
        for step in recorder.steps
    ]
    assert next_checks == [1, 1, 0]


@pytest.mark.anyio
async def test_checking_step_gets_no_page_checks(page, recorder):
    await _start(recorder, page)
    drafted = await recorder.draft_assertion("authorized personnel only", "Check the notice", page, _bank_run())
    committed = await recorder.commit(drafted.step, page, _bank_run(), derived=drafted.derived)
    assert _checks(committed.step) == []


@pytest.mark.anyio
async def test_irreversible_step_is_recorded_unclicked_with_no_checks(page, recorder, run_logger):
    await _start(recorder, page)
    _, derived, step = await _draft_on(recorder, page, 'button "Log In"', Action(ActionType.CLICK, "Sign in"))
    committed = await recorder.commit(step, page, _bank_run(), derived=derived, acted=False)
    assert committed.step.checkpoints == []
    assert _log_lines(run_logger)[-1]["acted"] is False


@pytest.mark.anyio
async def test_step_committed_out_of_order_is_refused(page, recorder):
    await _start(recorder, page)
    _, derived, first = await _draft_on(recorder, page, 'button "Log In"', Action(ActionType.CLICK, "Sign in"))
    _, _, stale = await _draft_on(recorder, page, 'button "Log In"', Action(ActionType.CLICK, "Sign in again"))
    await recorder.commit(first, page, _bank_run(), derived=derived)
    with pytest.raises(RecordingError, match="out of order"):
        await recorder.commit(stale, page, _bank_run(), derived=derived)


@pytest.mark.anyio
async def test_commit_requires_the_derived_locators_argument(page, recorder):
    # Keyword-only and required, so the run log always gets the locator details.
    await page.goto("/login")
    with pytest.raises(TypeError):
        await recorder.commit(recorder.draft_start(page.url), page, _bank_run())


# Typing into password boxes

@pytest.mark.anyio
async def test_secret_into_the_real_password_box_is_drafted(page, recorder):
    await _start(recorder, page)
    _, _, step = await _draft_on(recorder, page, "password box",
                                 Action(ActionType.TYPE, "Enter the password", "{credential:bank_password}"))
    assert (step.action, step.input_value) == (ActionType.TYPE, "{credential:bank_password}")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "field, value, message",
    [
        pytest.param("password box", "not-our-password", "a password box only takes a secret reference",
                     id="literal into password box"),
        pytest.param("password box", "{credential:bank_username}", "a password box only takes a secret reference",
                     id="username into password box"),
        pytest.param("password box", "{member_id}", "a password box only takes a secret reference",
                     id="input into password box"),
        pytest.param('text box, left label "Username:"', "{credential:bank_password}",
                     "is a secret and can only be typed into a password box", id="secret into username box"),
    ],
)
async def test_wrong_typing_on_the_real_sign_on_page_is_refused_before_a_key_is_pressed(
    page, recorder, run_logger, field, value, message
):
    await _start(recorder, page)
    with pytest.raises(TypingRefused, match=message):
        await _draft_on(recorder, page, field, Action(ActionType.TYPE, "Type it", value))
    assert await page.input_value('input[name="username"]') == ""
    assert await page.input_value('input[name="password"]') == ""
    assert len(recorder.steps) == 1
    assert len(_log_lines(run_logger)) == 1


@pytest.mark.anyio
async def test_our_real_password_typed_as_a_literal_is_refused_and_never_shown(page, recorder, run_logger):
    # The logger scrubs only the configured secrets, so a fake one proves the value never
    # reached the log at all rather than being scrubbed on the way.
    run = RunValues(secrets={"bank_password": SecretStr(FAKE_PASSWORD)})
    await _start(recorder, page, run=run)
    observation = await observe(page)
    box = next(element for element in observation.elements if element.description.startswith("password box"))
    derived = await derive_locators(page, box, run)
    with pytest.raises(TypingRefused) as refused:
        await recorder.draft_step(Action(ActionType.TYPE, "Enter the password", FAKE_PASSWORD),
                                  box.handle, derived, page.url, run=run)
    assert FAKE_PASSWORD not in str(refused.value)
    assert FAKE_PASSWORD not in run_logger.log_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_clicking_a_password_box_is_not_a_typing_check(page, recorder):
    await _start(recorder, page)
    _, _, step = await _draft_on(recorder, page, "password box", Action(ActionType.CLICK, "Focus the password box"))
    assert step.action == ActionType.CLICK


def test_draft_step_requires_this_runs_values(recorder):
    # A required keyword, so no caller can skip the check by leaving out the secrets.
    with pytest.raises(TypeError, match="run"):
        recorder.draft_step(Action(ActionType.TYPE, "Type it", "x"), None, None, "about:blank")


# Assertions by text

@pytest.mark.anyio
async def test_unique_phrase_becomes_a_checking_step(page, recorder):
    await _start(recorder, page)
    drafted = await recorder.draft_assertion("authorized personnel only", "Check the notice", page, _bank_run())
    step = drafted.step
    assert (step.action, step.input_value, step.sequence_index) == (
        ActionType.ASSERT_TEXT, "authorized personnel only", 1)
    assert step.locators
    for locator in step.locators:
        found = resolve(page, locator, {})
        assert await found.count() == 1
        assert await shows_phrase(await found.element_handle(), "authorized personnel only")


REFUSAL_PAGE = ('<div>Payment submitted</div><div>Payment submitted</div>'
                '<div style="display:none">Receipt ready</div>')


@pytest.mark.anyio
@pytest.mark.parametrize(
    "phrase, message",
    [
        pytest.param("   ", "needs the text", id="empty"),
        pytest.param("Nothing like this", "no visible element shows", id="not shown"),
        pytest.param("Payment submitted", "shown by 2 elements", id="shown twice"),
        pytest.param("Receipt ready", "no visible element shows", id="only hidden"),
    ],
)
async def test_assertion_is_refused_with_a_reason_for_the_model(page, recorder, phrase, message):
    await _start_on_html(recorder, page, REFUSAL_PAGE)
    with pytest.raises(AssertionRefused, match=message):
        await recorder.draft_assertion(phrase, "Check", page, _bank_run())


@pytest.mark.anyio
async def test_confirm_payment_phrase_is_refused_on_the_real_confirm_page(page, recorder, dashboard_popup):
    # The panel title and the button both show it; the prompt must ask for a phrase shown once.
    await _open_confirm_page(page, dashboard_popup)
    await recorder.commit(recorder.draft_start(page.url), page, _bank_run(), derived=None)
    with pytest.raises(AssertionRefused, match="shown by 2 elements"):
        await recorder.draft_assertion("Confirm Payment", "Check the confirm page", page, _bank_run())


# The run log and the whole recording

@pytest.mark.anyio
async def test_each_commit_writes_one_step_recorded_line(page, recorder, run_logger):
    await _start(recorder, page)
    _, derived, step = await _draft_on(recorder, page, 'text box, left label "Username:"',
                                       Action(ActionType.TYPE, "Enter the username", "{credential:bank_username}"))
    await recorder.commit(step, page, _bank_run(), derived=derived)
    lines = _log_lines(run_logger)
    assert [line["event_type"] for line in lines] == ["STEP_RECORDED", "STEP_RECORDED"]
    recorded = lines[1]
    assert [locator["kind"] for locator in recorded["locators"]] == derived.kinds
    assert recorded["rejected"] == [{"kind": r.kind, "reason": r.reason} for r in derived.rejected]
    assert (recorded["weak"], recorded["next_step_check_added_to"]) == (False, 0)
    assert recorded["input_value"] == "{credential:bank_username}"


@pytest.mark.anyio
async def test_a_whole_recording_on_the_bank_is_a_valid_artifact(page, recorder, run_logger, dashboard_popup):
    dashboard_popup(False)
    run = _bank_run()
    await _start(recorder, page)

    async def record(description, action, perform=None, lands_on=None, acted=True):
        element, derived, step = await _draft_on(recorder, page, description, action)
        if acted:
            await perform(element.handle)
            if lands_on:
                await page.wait_for_url(lands_on)
        await recorder.commit(step, page, run, derived=derived, acted=acted)

    await record('text box, left label "Username:"',
                 Action(ActionType.TYPE, "Enter the username", "{credential:bank_username}"),
                 lambda handle: handle.fill(env.mock_bank_username))
    await record("password box", Action(ActionType.TYPE, "Enter the password", "{credential:bank_password}"),
                 lambda handle: handle.fill(env.mock_bank_password.get_secret_value()))
    await record('button "Log In"', Action(ActionType.CLICK, "Sign in"),
                 lambda handle: handle.click(), "**/dashboard")
    await record('link "Member Search"', Action(ActionType.CLICK, "Open member search"),
                 lambda handle: handle.click(), "**/search")
    await record('text box, left label "Member ID:"', Action(ActionType.TYPE, "Enter the member", "{member_id}"),
                 lambda handle: handle.fill("10234"))
    await record('button "Search"', Action(ActionType.CLICK, "Search"),
                 lambda handle: handle.click(), "**/member/10234")
    await record('link "Bill Pay"', Action(ActionType.CLICK, "Open Bill Pay"),
                 lambda handle: handle.click(), "**/billpay")
    await record("dropdown", Action(ActionType.SELECT, "Choose the payee", "{payee_name}"),
                 lambda handle: handle.select_option(label="Sunbelt Electric Co"))
    await record('text box, left label "Amount:"', Action(ActionType.TYPE, "Enter the amount", "{amount}"),
                 lambda handle: handle.fill("50.00"))
    await record('button "Continue"', Action(ActionType.CLICK, "Continue"),
                 lambda handle: handle.click(), "**/billpay/confirm")
    drafted = await recorder.draft_assertion("Amount:", "Check the confirmation", page, run)
    await recorder.commit(drafted.step, page, run, derived=drafted.derived)
    await record('button "Confirm Payment"', Action(ActionType.CLICK, "Submit it"), acted=False)

    now = datetime.now(timezone.utc)
    artifact = Artifact(
        metadata=ArtifactMetadata(
            capability="member_servicing_and_bill_pay",
            description="For member {member_id}, pay {amount} to {payee_name}.",
            version="1.0.0",
            target_url=f"{env.mock_bank_base_url}/login",
            created_timestamp=now,
            last_updated_timestamp=now,
        ),
        input_parameters=[
            InputParamDefinition(key="member_id", type=ParamType.STRING, description="Member ID"),
            InputParamDefinition(key="amount", type=ParamType.NUMBER, description="Amount to pay"),
            InputParamDefinition(key="payee_name", type=ParamType.STRING, description="Payee"),
        ],
        credentials=[
            CredentialDefinition(key="bank_username", kind=CredentialKind.CONFIG, description="Teller username"),
            CredentialDefinition(key="bank_password", kind=CredentialKind.SECRET, description="Teller password"),
        ],
        steps=recorder.steps,
    )
    assert len(artifact.steps) == 13
    assert artifact.steps[-1].safety_tier == SafetyTier.IRREVERSIBLE
    assert artifact.steps[-1].checkpoints == []
    assert env.mock_bank_password.get_secret_value() not in run_logger.log_path.read_text(encoding="utf-8")


# --- backstop: the save-time scan over the whole artifact ---

def _bs_locators(*values, kind=LocatorType.CSS) -> list[Locator]:
    return [Locator(type=kind, value=value, priority=number) for number, value in enumerate(values)]


def _bs_step(index, action, description, value=None, *, locators=None, checkpoints=(), option_value=None) -> Step:
    if locators is None:
        locators = [] if action == ActionType.NAVIGATE else _bs_locators(f"#s{index}")
    return Step(sequence_index=index, action=action, description=description, locators=locators,
                input_value=value, option_value=option_value, checkpoints=list(checkpoints))


def _bs_check(kind, value, target=None) -> StepCheckpoint:
    return StepCheckpoint(type=kind, expected_value=value, target_locator=target, timeout_ms=10000)


def _bs_artifact(*steps, example_value=None, global_assertions=()) -> Artifact:
    # Step 0 opens the start page; the steps given follow it, numbered from 1.
    now = datetime.now(timezone.utc)
    return Artifact(
        metadata=ArtifactMetadata(capability="member_servicing_and_bill_pay",
                                  description="For member {member_id}, pay {amount} to {payee_name}.",
                                  version="1.0.0", target_url="http://localhost:5000/login",
                                  created_timestamp=now, last_updated_timestamp=now),
        input_parameters=[
            InputParamDefinition(key="member_id", type=ParamType.STRING, description="Member ID",
                                 example_value=example_value),
            InputParamDefinition(key="amount", type=ParamType.NUMBER, description="Amount to pay"),
            InputParamDefinition(key="payee_name", type=ParamType.STRING, description="Payee"),
        ],
        credentials=[
            CredentialDefinition(key="bank_username", kind=CredentialKind.CONFIG, description="Teller username"),
            CredentialDefinition(key="bank_password", kind=CredentialKind.SECRET, description="Teller password"),
        ],
        steps=[_bs_step(0, ActionType.NAVIGATE, "Open the start page"), *steps],
        global_assertions=list(global_assertions),
    )


def _bs_inputs(extracted=None, **extra_text) -> ScanInputs:
    # The username and the secret are set here, not read from .env, so every case is explicit.
    run = RunValues(text_inputs={"member_id": "10234", "payee_name": "Sunbelt Electric Co", **extra_text},
                    number_inputs={"amount": 50.0}, username="admin",
                    secrets={"bank_password": SecretStr(FAKE_PASSWORD)})
    return ScanInputs(run, username_key="bank_username", extracted=extracted or {})


def _bs_problems(*steps, **extra_text):
    inputs = _bs_inputs(**extra_text)
    return find_problems(convert(_bs_artifact(*steps), inputs), inputs)


# Which fields the scan reads

def test_every_text_field_is_read_with_the_treatment_for_who_wrote_it():
    artifact = _bs_artifact(
        _bs_step(1, ActionType.TYPE, "Enter the username", "{credential:bank_username}"),
        _bs_step(2, ActionType.SELECT, "Pay from checking", "Checking", option_value="CHK"),
        _bs_step(3, ActionType.ASSERT_TEXT, "Check the panel", "Payment Details"),
        _bs_step(4, ActionType.CLICK, "Open the member", checkpoints=[
            _bs_check(CheckpointType.PAGE_TITLE, "Member Detail"),
            _bs_check(CheckpointType.URL_CONTAINS, "/member"),
            _bs_check(CheckpointType.TEXT_MATCH, "Active", _bs_locators("td.status")[0])]),
        example_value="10234",
        global_assertions=[GlobalAssertion(type=GlobalAssertionType.SUCCESS_BANNER_TEXT, value="Payment ready")],
    )
    kinds = {"/".join(map(str, field.path)): field.kind for field in artifact_fields(artifact)}
    assert {
        "metadata/description": FieldKind.CONTRACT,
        "input_parameters/0/example_value": FieldKind.CONTRACT,
        "steps/0/description": FieldKind.DESCRIPTION,
        "steps/1/input_value": FieldKind.TYPED,
        "steps/2/input_value": FieldKind.TYPED,
        "steps/2/option_value": FieldKind.RECORDED,
        "steps/3/input_value": FieldKind.CHECKED,
        "steps/4/locators/0/value": FieldKind.RECORDED,
        "steps/4/checkpoints/0/expected_value": FieldKind.RECORDED,
        "steps/4/checkpoints/1/expected_value": FieldKind.RECORDED,
        "steps/4/checkpoints/2/target_locator/value": FieldKind.RECORDED,
        "steps/4/checkpoints/2/expected_value": FieldKind.CHECKED,
        "global_assertions/0/value": FieldKind.CHECKED,
    }.items() <= kinds.items()


def test_a_text_field_with_no_treatment_stops_the_scan(monkeypatch):
    # Stands in for a field added to the schema later and never given a treatment.
    monkeypatch.setattr(backstop, "SKIPPED_FIELDS", backstop.SKIPPED_FIELDS - {("steps", "*", "step_id")})
    with pytest.raises(UnclassifiedField, match=r"steps/\*/step_id"):
        artifact_fields(_bs_artifact())


def test_an_input_value_on_a_click_has_no_treatment():
    with pytest.raises(UnclassifiedField, match="click"):
        artifact_fields(_bs_artifact(_bs_step(1, ActionType.CLICK, "Go", "hello")))


def test_known_outcomes_and_currencies_are_read_as_the_engineers_contract():
    # model_copy skips validation: only which fields exist matters here, not whether a
    # step reads the output.
    artifact = _bs_artifact().model_copy(update={
        "output_definitions": [OutputParamDefinition(key="balance", type=OutputType.MONEY,
                                                     description="Checking balance", currency="USD")],
        "known_outcomes": [
            KnownOutcome(code="MEMBER_NOT_FOUND", description="No member has this ID",
                         signal=OutcomeSignal.PAGE_TEXT, text="No member found with that ID."),
            KnownOutcome(code="PAYEE_NOT_FOUND", description="The payee isn't in the list",
                         signal=OutcomeSignal.NO_SUCH_OPTION, input_key="payee_name"),
        ],
    })
    kinds = {"/".join(map(str, field.path)): field.kind for field in artifact_fields(artifact)}
    assert {
        "output_definitions/0/description": FieldKind.CONTRACT,
        "known_outcomes/0/description": FieldKind.CONTRACT,
        "known_outcomes/0/text": FieldKind.CONTRACT,
        "known_outcomes/1/description": FieldKind.CONTRACT,
    }.items() <= kinds.items()
    assert not [path for path in kinds if path.endswith(("/code", "/signal", "/input_key", "/currency"))]


def test_the_allowed_pages_are_read_as_the_engineers_contract():
    inputs = _bs_inputs()
    data = _bs_artifact().model_dump(mode="json")
    data["allowed_paths"] = ["/login", f"/{FAKE_PASSWORD}"]
    artifact = Artifact.model_validate(data)
    kinds = {"/".join(map(str, field.path)): field.kind for field in artifact_fields(artifact)}
    assert (kinds["allowed_paths/0"], kinds["allowed_paths/1"]) == (FieldKind.CONTRACT, FieldKind.CONTRACT)
    [finding] = find_problems(convert(artifact, inputs), inputs)
    assert finding.code == AbortCode.SECRET_LITERAL
    assert "allowed page 1" in finding.message
    assert FAKE_PASSWORD not in finding.message


def test_a_secret_in_a_known_outcome_stops_the_save_and_names_the_outcome():
    inputs = _bs_inputs()
    data = _bs_artifact().model_dump(mode="json")
    data["known_outcomes"] = [{"code": "LOCKED", "description": "The account is locked", "signal": "page_text",
                               "text": f"Password {FAKE_PASSWORD} is locked"}]
    [finding] = find_problems(convert(Artifact.model_validate(data), inputs), inputs)
    assert finding.code == AbortCode.SECRET_LITERAL
    assert "known outcome LOCKED text" in finding.message
    assert FAKE_PASSWORD not in finding.message


# Converting this run's values

@pytest.mark.parametrize(
    "action, typed, stored",
    [
        pytest.param(ActionType.TYPE, "ADMIN", "{credential:bank_username}", id="username in any case"),
        pytest.param(ActionType.TYPE, "10234", "{member_id}", id="text input"),
        pytest.param(ActionType.TYPE, "50.00", "{amount}", id="amount with two decimals"),
        pytest.param(ActionType.TYPE, "50", "{amount}", id="amount as a whole number"),
        pytest.param(ActionType.SELECT, "sunbelt electric co", "{payee_name}", id="payee label in any case"),
        pytest.param(ActionType.TYPE, "Member 10234", "Member 10234", id="input inside other text stays"),
        pytest.param(ActionType.TYPE, "$50.00", "$50.00", id="amount with a currency sign stays"),
        pytest.param(ActionType.TYPE, "Checking", "Checking", id="literal matching nothing stays"),
        pytest.param(ActionType.SELECT, "admin", "admin", id="username is never a dropdown choice"),
    ],
)
def test_a_typed_value_is_converted_only_when_the_whole_value_is_one_input(action, typed, stored):
    converted = convert(_bs_artifact(_bs_step(1, action, "Type it", typed)), _bs_inputs())
    assert converted.artifact.steps[1].input_value == stored


def test_a_dropdown_whose_choice_became_an_input_drops_its_hidden_value():
    artifact = _bs_artifact(_bs_step(1, ActionType.SELECT, "Pick", "Sunbelt Electric Co", option_value="P001"))
    converted = convert(artifact, _bs_inputs())
    assert (converted.artifact.steps[1].input_value, converted.artifact.steps[1].option_value) == ("{payee_name}", None)
    assert converted.option_values_dropped == [1]


def test_whole_words_are_replaced_in_checked_text_and_descriptions():
    artifact = _bs_artifact(
        _bs_step(1, ActionType.ASSERT_TEXT, "Check the amount", "Amount: $50.00"),
        _bs_step(2, ActionType.CLICK, "Logged in as admin; pay member 10234 $50 to SUNBELT ELECTRIC CO. "
                                      "Administration menu; 150 and 50.75 are other numbers."),
    )
    converted = convert(artifact, _bs_inputs())
    assert converted.artifact.steps[1].input_value == "Amount: ${amount}"
    assert converted.artifact.steps[2].description == (
        "Logged in as (teller username); pay member {member_id} ${amount} to {payee_name}. "
        "Administration menu; 150 and 50.75 are other numbers.")
    assert [c.replaced_with for c in converted.conversions if c.step_index == 2] == [
        "(teller username)", "{member_id}", "{amount}", "{payee_name}"]


def test_placeholders_recorded_fields_and_the_contract_are_never_rewritten():
    artifact = _bs_artifact(
        _bs_step(1, ActionType.CLICK, "Already {member_id}", checkpoints=[
            _bs_check(CheckpointType.PAGE_TITLE, "Member 10234")]),
        example_value="10234",
    )
    original = artifact.model_dump()
    converted = convert(artifact, _bs_inputs())
    assert converted.artifact.model_dump() == original
    assert converted.conversions == []
    assert artifact.model_dump() == original


def test_a_value_equal_to_two_inputs_is_left_as_it_was():
    artifact = _bs_artifact(_bs_step(1, ActionType.TYPE, "Search for member 10234", "10234"))
    converted = convert(artifact, _bs_inputs(account_id="10234"))
    assert (converted.artifact.steps[1].input_value, converted.artifact.steps[1].description) == (
        "10234", "Search for member 10234")
    assert [a.candidates for a in converted.ambiguities] == [("{account_id}", "{member_id}")] * 2


# What stops the save

@pytest.mark.parametrize(
    "step, extra_text, code, words",
    [
        pytest.param(_bs_step(1, ActionType.TYPE, "Type it", FAKE_PASSWORD), {},
                     AbortCode.SECRET_LITERAL, "the secret bank_password", id="secret"),
        pytest.param(_bs_step(1, ActionType.TYPE, "Type it", "Member 10234"), {},
                     AbortCode.EMBEDDED_INPUT_LITERAL, "the input member_id inside other text", id="embedded input"),
        pytest.param(_bs_step(1, ActionType.ASSERT_TEXT, "Check", "Teller: admin"), {},
                     AbortCode.USERNAME_LITERAL, "teller username", id="username in an assertion"),
        pytest.param(_bs_step(1, ActionType.ASSERT_TEXT, "Check", "Call (602) 555-0142"), {},
                     AbortCode.SENSITIVE_LITERAL, "phone number", id="phone in an assertion"),
        pytest.param(_bs_step(1, ActionType.CLICK, "Email laura.whitfield@example.com"), {},
                     AbortCode.SENSITIVE_LITERAL, "email address", id="email in a description"),
        pytest.param(_bs_step(1, ActionType.CLICK, "Pay", locators=_bs_locators("Pay $50.00", kind=LocatorType.TEXT_CONTENT)),
                     {}, AbortCode.RECORDED_FIELD_LITERAL, "locator 1: carries the value of the input amount",
                     id="amount in a locator"),
        pytest.param(_bs_step(1, ActionType.CLICK, "Go", checkpoints=[_bs_check(CheckpointType.PAGE_PATH, "/member/10234")]),
                     {}, AbortCode.RECORDED_FIELD_LITERAL, "page path check: carries the value of the input member_id",
                     id="member in a page path"),
        pytest.param(_bs_step(1, ActionType.TYPE, "Type it", "10234"), {"account_id": "10234"},
                     AbortCode.AMBIGUOUS_LITERAL, "{account_id}, {member_id}", id="equal to two inputs"),
    ],
)
def test_each_problem_stops_the_save_with_its_own_code(step, extra_text, code, words):
    findings = _bs_problems(step, **extra_text)
    assert [finding.code for finding in findings] == [code]
    assert words in findings[0].message
    assert FAKE_PASSWORD not in findings[0].message


@pytest.mark.parametrize(
    "example_value, codes",
    [pytest.param(FAKE_PASSWORD, [AbortCode.SECRET_LITERAL], id="a secret stops it"),
     pytest.param("10234", [], id="an input value is the engineer's choice")],
)
def test_the_contract_is_checked_for_secrets_only(example_value, codes):
    inputs = _bs_inputs()
    findings = find_problems(convert(_bs_artifact(example_value=example_value), inputs), inputs)
    assert [finding.code for finding in findings] == codes


@pytest.mark.anyio
@pytest.mark.parametrize("amount", [1.0, 2.0, 3.0])
async def test_position_numbers_our_generator_writes_are_never_read_as_an_amount(page, amount):
    # Every locator the recorder would store for the sign-on page's elements, position
    # paths and the label XPath's [1] included, passes the second look with a small amount.
    await page.goto("/login")
    observation = await observe(page)
    run = RunValues(number_inputs={"amount": amount})
    stored = []
    for element in observation.elements:
        for candidate in await generate_candidates(element):
            outcome = scan(candidate, run)
            if outcome.stored_value is not None:
                stored.append((candidate.locator_type, outcome.stored_value))
    assert any(":nth-of-type(" in value for _, value in stored)
    assert any("[1]" in value for _, value in stored)
    locators = [Locator(type=kind, value=value, priority=number) for number, (kind, value) in enumerate(stored)]
    artifact = _bs_artifact(_bs_step(1, ActionType.CLICK, "Sign in", locators=locators))
    inputs = ScanInputs(run, username_key="bank_username", extracted={})
    assert find_problems(convert(artifact, inputs), inputs) == []


def test_the_result_reports_the_worst_finding_and_counts_the_rest():
    findings = _bs_problems(_bs_step(1, ActionType.TYPE, "Type it", "Member 10234"),
                            _bs_step(2, ActionType.TYPE, "Type it", FAKE_PASSWORD))
    error = abort_error(findings)
    assert error.code == "SECRET_LITERAL"
    assert error.message.startswith("step 2 typed value:")
    assert error.message.endswith("; 1 more finding in the run log")


def test_on_a_tie_the_earliest_finding_is_reported():
    findings = _bs_problems(_bs_step(1, ActionType.ASSERT_TEXT, "Check", "Teller: admin"),
                            _bs_step(2, ActionType.TYPE, "Type it", "Member 10234"))
    assert abort_error(findings).message.startswith("step 1 assertion:")


def test_abort_error_needs_a_finding():
    with pytest.raises(ValueError):
        abort_error([])


# Assertions that may hold only for this member, and the report

def test_an_assertion_holding_a_value_this_run_read_or_typed_is_flagged_not_stopped():
    artifact = _bs_artifact(
        _bs_step(1, ActionType.SELECT, "Pay from checking", "Checking"),
        _bs_step(2, ActionType.ASSERT_TEXT, "Check the balance", "Balance: $2,450.32"),
        _bs_step(3, ActionType.ASSERT_TEXT, "Check the account", "Checking account"),
        _bs_step(4, ActionType.ASSERT_TEXT, "Check the panel", "Payment Details"),
    )
    inputs = _bs_inputs(extracted={"checking_balance": "$2,450.32"})
    flagged = flag_assertions(convert(artifact, inputs).artifact, inputs)
    assert [(flag.step_index, flag.reason) for flag in flagged] == [
        (2, "contains the value read into checking_balance"),
        (3, "contains the value typed at step 1"),
    ]
    assert find_problems(convert(artifact, inputs), inputs) == []


def test_a_clean_artifact_passes_with_its_report():
    artifact = _bs_artifact(
        _bs_step(1, ActionType.TYPE, "Enter member 10234", "10234"),
        _bs_step(2, ActionType.SELECT, "Pay from checking", "Checking"),
        _bs_step(3, ActionType.ASSERT_TEXT, "Check the amount", "Amount: $50.00"),
    )
    result = scan_artifact(artifact, _bs_inputs())
    assert result.error is None
    assert result.artifact.steps[1].input_value == "{member_id}"
    fields = result.report.log_fields()
    assert fields["outcome"] == "passed"
    assert [c["replaced_with"] for c in fields["conversions"]] == ["{member_id}", "{member_id}", "{amount}"]
    assert fields["literals_kept"] == [{"step": 2, "field": "steps/2/input_value", "value": "Checking"}]
    assert fields["assertions_recorded"] == [{"step": 3, "field": "steps/3/input_value", "text": "Amount: ${amount}"}]
    assert fields["findings"] == []


def test_a_stopped_save_logs_every_finding_and_never_a_value_that_stopped_it(run_logger):
    artifact = _bs_artifact(
        _bs_step(1, ActionType.TYPE, "Type it", "Member 10234"),
        _bs_step(2, ActionType.TYPE, "Type it", FAKE_PASSWORD),
        _bs_step(3, ActionType.SELECT, "Pay from checking", "Checking"),
    )
    result = scan_artifact(artifact, _bs_inputs())
    assert result.artifact is None
    assert result.error.code == "SECRET_LITERAL"
    run_logger.backstop_scan(**result.report.log_fields())
    line = _log_lines(run_logger)[-1]
    assert (line["event_type"], line["outcome"]) == ("BACKSTOP_SCAN", "SECRET_LITERAL")
    assert [finding["code"] for finding in line["findings"]] == ["EMBEDDED_INPUT_LITERAL", "SECRET_LITERAL"]
    assert line["literals_kept"] == [{"step": 3, "field": "steps/3/input_value", "value": "Checking"}]
    logged = run_logger.log_path.read_text(encoding="utf-8")
    assert FAKE_PASSWORD not in logged
    assert "Member 10234" not in logged


# --- artifact builder: validate, scan, sign, write ---

@pytest.fixture
def storage(tmp_path, monkeypatch):
    # Each test saves into its own temporary store, never the project's artifacts folder.
    folder = tmp_path / "artifacts"
    monkeypatch.setattr(settings, "artifact_storage_dir", folder)
    return folder


def _ab_contract(outputs=()) -> ArtifactContract:
    # The same contract the backstop tests use, taken from their artifact.
    template = _bs_artifact()
    return ArtifactContract(
        capability=template.metadata.capability,
        description=template.metadata.description,
        target_url=template.metadata.target_url,
        input_parameters=template.input_parameters,
        output_definitions=list(outputs),
        credentials=template.credentials,
    )


def _ab_start() -> Step:
    return _bs_step(0, ActionType.NAVIGATE, "Open the start page",
                    checkpoints=[_bs_check(CheckpointType.PAGE_PATH, "/login")])


def test_a_clean_recording_is_saved_signed_and_logged(storage, run_logger):
    steps = [_ab_start(),
             _bs_step(1, ActionType.TYPE, "Enter member 10234", "10234"),
             _bs_step(2, ActionType.ASSERT_TEXT, "Check the amount", "Amount: $50.00")]
    result = build_and_save(_ab_contract(), steps, _bs_inputs(), run_logger)
    assert result.error is None
    metadata = result.artifact.metadata
    assert result.path == storage / "member_servicing_and_bill_pay" / f"{metadata.artifact_id}_v1.0.0.json"
    saved = Artifact.model_validate_json(result.path.read_text(encoding="utf-8"))
    verify(saved, env.artifact_signing_key)
    assert (saved.steps[1].input_value, saved.steps[2].input_value) == ("{member_id}", "Amount: ${amount}")
    lines = _log_lines(run_logger)
    assert [line["event_type"] for line in lines] == ["BACKSTOP_SCAN", "ARTIFACT_SAVED"]
    assert lines[1]["sha256_hash"] == metadata.integrity_hash
    assert [file.name for file in result.path.parent.iterdir()] == [result.path.name]


def test_an_invalid_recording_is_not_saved_and_its_message_shows_no_value(storage, run_logger):
    # A declared output no step reads, and a secret typed as a literal that must not be echoed.
    steps = [_ab_start(), _bs_step(1, ActionType.TYPE, "Type it", FAKE_PASSWORD)]
    outputs = [OutputParamDefinition(key="checking_balance", type=OutputType.STRING, description="Balance")]
    result = build_and_save(_ab_contract(outputs), steps, _bs_inputs(), run_logger)
    assert result.error.code == "ARTIFACT_INVALID"
    assert "checking_balance" in result.error.message
    assert FAKE_PASSWORD not in result.error.message
    assert not storage.exists()
    assert _log_lines(run_logger) == []


BILL_PAY_OUTCOMES = [
    KnownOutcome(code="MEMBER_NOT_FOUND", description="No member has this ID",
                 signal=OutcomeSignal.PAGE_TEXT, text="No member found with that ID."),
    KnownOutcome(code="PAYEE_NOT_FOUND", description="The payee isn't in the list",
                 signal=OutcomeSignal.NO_SUCH_OPTION, input_key="payee_name"),
]


def test_the_contracts_known_outcomes_are_saved_and_signed(storage, run_logger):
    contract = dataclasses.replace(_ab_contract(), known_outcomes=BILL_PAY_OUTCOMES)
    steps = [_ab_start(), _bs_step(1, ActionType.TYPE, "Enter member 10234", "10234")]
    result = build_and_save(contract, steps, _bs_inputs(), run_logger)
    assert result.error is None
    saved = Artifact.model_validate_json(result.path.read_text(encoding="utf-8"))
    assert saved.known_outcomes == BILL_PAY_OUTCOMES
    verify(saved, env.artifact_signing_key)
    # The outcomes are part of what the signature covers: dropping one is detected.
    with pytest.raises(IntegrityCheckFailed):
        verify(saved.model_copy(update={"known_outcomes": BILL_PAY_OUTCOMES[1:]}), env.artifact_signing_key)


def test_a_known_outcome_naming_an_undeclared_input_is_not_saved(storage, run_logger):
    stray = KnownOutcome(code="PAYEE_NOT_FOUND", description="The payee isn't in the list",
                         signal=OutcomeSignal.NO_SUCH_OPTION, input_key="payee")
    contract = dataclasses.replace(_ab_contract(), known_outcomes=[stray])
    result = build_and_save(contract, [_ab_start()], _bs_inputs(), run_logger)
    assert result.error.code == "ARTIFACT_INVALID"
    assert "payee is not a declared input" in result.error.message
    assert not storage.exists()


def test_the_contracts_allowed_pages_are_saved_and_signed(storage, run_logger):
    contract = dataclasses.replace(_ab_contract(), allowed_paths=["/login", "/member/*"])
    result = build_and_save(contract, [_ab_start()], _bs_inputs(), run_logger)
    assert result.error is None
    saved = Artifact.model_validate_json(result.path.read_text(encoding="utf-8"))
    assert saved.allowed_paths == ["/login", "/member/*"]
    verify(saved, env.artifact_signing_key)


def test_a_contract_whose_start_page_is_not_allowed_is_not_saved(storage, run_logger):
    contract = dataclasses.replace(_ab_contract(), allowed_paths=["/member/*"])
    result = build_and_save(contract, [_ab_start()], _bs_inputs(), run_logger)
    assert result.error.code == "ARTIFACT_INVALID"
    assert "start page" in result.error.message
    assert not storage.exists()


def _ab_recording(description="Enter member 10234", extra_step=False, outcomes=()):
    # A contract and its recording; the defaults make the same recording every time.
    steps = [_ab_start(), _bs_step(1, ActionType.TYPE, description, "10234")]
    if extra_step:
        steps.append(_bs_step(2, ActionType.ASSERT_TEXT, "Check the amount", "Amount: $50.00"))
    return dataclasses.replace(_ab_contract(), known_outcomes=list(outcomes)), steps


def test_a_rediscovery_with_nothing_changed_writes_no_new_version(storage, run_logger):
    first = build_and_save(*_ab_recording(), _bs_inputs(), run_logger)
    again = build_and_save(*_ab_recording(), _bs_inputs(), run_logger)
    assert (again.path, again.artifact.metadata.version) == (first.path, "1.0.0")
    assert list(first.path.parent.iterdir()) == [first.path]
    last = _log_lines(run_logger)[-1]
    assert (last["event_type"], last["artifact_id"], last["version"]) == (
        "ARTIFACT_UNCHANGED", first.artifact.metadata.artifact_id, "1.0.0")


@pytest.mark.parametrize(
    "changes, expected",
    [
        pytest.param({"description": "Type member 10234"}, "1.0.1", id="wording only: patch"),
        pytest.param({"extra_step": True}, "1.1.0", id="an extra step: minor"),
        pytest.param({"outcomes": BILL_PAY_OUTCOMES}, "2.0.0", id="a new known outcome: major"),
    ],
)
def test_a_rediscovery_gets_the_next_version_for_what_changed(storage, run_logger, changes, expected):
    first = build_and_save(*_ab_recording(), _bs_inputs(), run_logger)
    again = build_and_save(*_ab_recording(**changes), _bs_inputs(), run_logger)
    assert again.artifact.metadata.version == expected
    assert again.path.name == f"{again.artifact.metadata.artifact_id}_v{expected}.json"
    verify(Artifact.model_validate_json(again.path.read_text(encoding="utf-8")), env.artifact_signing_key)
    assert sorted(path.name for path in again.path.parent.iterdir()) == sorted([first.path.name, again.path.name])


def test_a_latest_version_that_fails_its_signature_is_numbered_past_with_a_major_bump(storage, run_logger):
    first = build_and_save(*_ab_recording(), _bs_inputs(), run_logger)
    data = json.loads(first.path.read_text(encoding="utf-8"))
    data["metadata"]["description"] = "Changed by hand."
    first.path.write_text(json.dumps(data), encoding="utf-8")
    # The same recording again: it isn't compared with a file that can't be trusted.
    again = build_and_save(*_ab_recording(), _bs_inputs(), run_logger)
    assert again.artifact.metadata.version == "2.0.0"


def test_a_recording_with_no_check_at_all_is_not_saved(storage, run_logger):
    steps = [_bs_step(0, ActionType.NAVIGATE, "Open the start page"), _bs_step(1, ActionType.CLICK, "Go")]
    result = build_and_save(_ab_contract(), steps, _bs_inputs(), run_logger)
    assert (result.error.code, result.artifact) == ("ARTIFACT_INVALID", None)
    assert "no checkpoint" in result.error.message
    assert not storage.exists()


def test_a_backstop_finding_stops_the_save_after_logging_the_report(storage, run_logger):
    steps = [_ab_start(), _bs_step(1, ActionType.TYPE, "Type it", FAKE_PASSWORD)]
    result = build_and_save(_ab_contract(), steps, _bs_inputs(), run_logger)
    assert (result.error.code, result.path) == ("SECRET_LITERAL", None)
    assert not storage.exists()
    assert [line["event_type"] for line in _log_lines(run_logger)] == ["BACKSTOP_SCAN"]


def test_an_unsigned_artifact_is_never_written(storage):
    with pytest.raises(UnsignedArtifact):
        write_artifact(_bs_artifact())
    assert not storage.exists()


def test_a_capability_that_is_not_a_simple_name_is_never_used_as_a_folder(storage):
    artifact = _bs_artifact()
    escaping = artifact.model_copy(update={"metadata": artifact.metadata.model_copy(update={"capability": "../escape"})})
    with pytest.raises(ValueError, match="simple lowercase name"):
        write_artifact(sign(escaping, env.artifact_signing_key))
    assert not storage.exists()


# --- browser: the session and the functions that act on the page ---

@pytest.mark.parametrize(
    "base_url, args",
    [pytest.param("http://localhost:5000", ["--host-resolver-rules=MAP localhost 127.0.0.1"], id="localhost"),
     pytest.param("http://127.0.0.1:5000", [], id="an IP address"),
     pytest.param("https://portal.examplebank.com", [], id="a real bank")],
)
def test_the_localhost_rule_applies_only_to_a_localhost_bank(base_url, args):
    assert launch_args(base_url) == args


@pytest.mark.parametrize("number, typed", [(50.0, "50"), (512.75, "512.75"), (1240.5, "1240.5"), (0.1, "0.1")])
def test_a_number_is_typed_in_its_plain_shortest_form(number, typed):
    assert number_text(number) == typed


@pytest.mark.parametrize("time_left, cap", [(120_000, 30_000), (5_000, 5_000), (0, 1)])
def test_an_action_gets_thirty_seconds_or_what_is_left(time_left, cap):
    assert action_timeout_ms(time_left) == cap


def test_placeholder_values_name_credentials_as_artifacts_do_and_keep_secrets_wrapped():
    values = placeholder_values({"member_id": "10234"}, {"amount": 50.0}, configured_credentials())
    assert (values["member_id"], values["amount"]) == ("10234", "50")
    assert values["credential:bank_username"] == env.mock_bank_username
    assert isinstance(values["credential:bank_password"], SecretStr)


BROWSER_VALUES = placeholder_values({"member_id": "10234", "payee_name": "Sunbelt Electric Co"}, {"amount": 50.0},
                                    {"bank_password": SecretStr(FAKE_PASSWORD)})


@pytest.mark.anyio
async def test_typing_fills_placeholders_and_the_secret_only_at_the_keystroke(page):
    await page.set_content('<input id="m" type="text"><input id="p" type="password">')
    await type_text(await page.query_selector("#m"), "{member_id}", BROWSER_VALUES, timeout_ms=5_000)
    await type_text(await page.query_selector("#p"), "{credential:bank_password}", BROWSER_VALUES, timeout_ms=5_000)
    assert await page.input_value("#m") == "10234"
    assert await page.input_value("#p") == FAKE_PASSWORD


@pytest.mark.anyio
async def test_a_failed_keystroke_never_carries_the_value(page):
    await page.set_content('<input id="p" type="password">')
    box = await page.query_selector("#p")
    await page.evaluate("document.getElementById('p').remove()")
    with pytest.raises(ActionFailed) as failure:
        await type_text(box, "{credential:bank_password}", BROWSER_VALUES, timeout_ms=1_000)
    assert FAKE_PASSWORD not in str(failure.value)
    assert (failure.value.__cause__, failure.value.__context__) == (None, None)


@pytest.mark.anyio
async def test_an_option_is_chosen_by_its_label_with_the_input_filled(page):
    await page.set_content(f"{QUIRKS_DOCTYPE}<html><body>{PAYEE_SELECT}</body></html>")
    await select_option(await page.query_selector("select"), "{payee_name}", BROWSER_VALUES, timeout_ms=5_000)
    assert await page.eval_on_selector("select", "select => select.value") == "P001"


@pytest.mark.anyio
async def test_a_dialog_nobody_expected_is_dismissed_and_noted(page, run_logger):
    notes = dismiss_dialogs(page, run_logger)
    # A dialog left open stalls the page, so a regression must fail here, not hang.
    assert await asyncio.wait_for(page.evaluate("confirm('Submit this payment?')"), timeout=10) is False
    assert notes == ["confirm: Submit this payment?"]
    line = _log_lines(run_logger)[-1]
    assert (line["event_type"], line["dialog_type"], line["dialog_message"]) == (
        "DIALOG_DISMISSED", "confirm", "Submit this payment?")


@pytest.mark.anyio
async def test_a_dialog_is_accepted_only_while_the_system_expects_one(page, run_logger):
    expecting = {"now": False}
    notes = dismiss_dialogs(page, run_logger, accept_now=lambda: expecting["now"])
    assert await asyncio.wait_for(page.evaluate("confirm('Leave this page?')"), timeout=10) is False
    expecting["now"] = True
    assert await asyncio.wait_for(page.evaluate("confirm('Submit this payment?')"), timeout=10) is True
    assert notes == ["confirm: Leave this page?", "confirm (accepted): Submit this payment?"]
    assert [line["event_type"] for line in _log_lines(run_logger)] == ["DIALOG_DISMISSED", "DIALOG_ACCEPTED"]


@pytest.mark.anyio
async def test_the_session_opens_an_allowed_start_page_and_refuses_another(mock_bank_url, run_logger):
    async with BrowserSession(run_logger) as session:
        await session.open(f"{mock_bank_url}/login", timeout_ms=10_000)
        assert session.page.url.endswith("/login")
        assert session.page.viewport_size == {"width": settings.discovery_viewport_width,
                                              "height": settings.discovery_viewport_height}
        with pytest.raises(AllowlistViolation):
            await session.open("https://example.com/", timeout_ms=1_000)
        assert session.page.url.endswith("/login")


# --- prompts: what the model reads ---

@pytest.mark.parametrize("word", ["bill pay", "payee", "member", "dashboard", "confirm payment", "localhost",
                                  "popup", "teller", "admin"])
def test_the_system_prompt_says_nothing_about_this_bank(word):
    # The same prompt must work on any app, so discovery is real and nothing is scripted.
    assert word not in SYSTEM_PROMPT.lower()


def test_every_tool_has_a_strict_schema_and_every_action_needs_a_reason():
    tools = {tool["name"]: tool for tool in tool_definitions(["checking_balance"])}
    assert set(tools) == {"click", "type_text", "select_option", "extract_text", "dismiss_overlay",
                          "assert_visible", "mark_goal_complete", "report_stuck"}
    for tool in tools.values():
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert schema["required"] == list(schema["properties"])
    for name in ("click", "type_text", "select_option", "extract_text", "dismiss_overlay", "assert_visible"):
        assert "reason" in tools[name]["input_schema"]["required"]
    assert "element" not in tools["assert_visible"]["input_schema"]["properties"]
    assert "element" not in tools["extract_text"]["input_schema"]["properties"]
    assert "label" in tools["extract_text"]["input_schema"]["required"]
    assert tools["extract_text"]["input_schema"]["properties"]["output_key"]["enum"] == ["checking_balance"]
    assert tools["report_stuck"]["input_schema"]["properties"]["category"]["enum"] == list(STUCK_CATEGORIES)


def test_extract_text_is_offered_only_when_outputs_are_declared():
    assert "extract_text" not in {tool["name"] for tool in tool_definitions([])}


def test_the_goal_message_gives_placeholders_and_values_but_credentials_by_name_only():
    contract = _ab_contract([OutputParamDefinition(key="checking_balance", type=OutputType.STRING,
                                                   description="Checking balance before paying")])
    text = goal_message("For member 10234, pay 50 to Sunbelt Electric Co.", contract.input_parameters,
                        {"member_id": "10234", "amount": 50.0, "payee_name": "Sunbelt Electric Co"},
                        contract.credentials, contract.output_definitions)
    assert text.startswith("Goal: For member 10234, pay 50 to Sunbelt Electric Co.")
    assert '{member_id} = "10234"' in text and '{amount} = "50"' in text
    assert "{credential:bank_password} (Teller password; secret: password boxes only)" in text
    assert "checking_balance (Checking balance before paying; text)" in text
    assert env.mock_bank_username not in text
    assert env.mock_bank_password.get_secret_value() not in text


def test_the_goal_message_says_what_kind_of_value_each_output_is():
    outputs = [OutputParamDefinition(key="balance", type=OutputType.MONEY, description="Balance", currency="USD"),
               OutputParamDefinition(key="count", type=OutputType.NUMBER, description="Count")]
    text = goal_message("Read them.", [], {}, [], outputs)
    assert "- balance (Balance; an amount in USD)" in text
    assert "- count (Count; a number)" in text


def test_progress_and_page_blocks_are_packaged_for_the_model():
    assert progress(3, 40, 12 * 60_000) == "Step 3 of 40, about 12 minutes left."
    assert progress(40, 40, 50_000) == "Step 40 of 40, about 1 minute left."
    image, elements = page_blocks(b"\x89PNG fake", "[1] link \"Home\"")
    assert (image["type"], image["source"]["media_type"]) == ("image", "image/png")
    assert elements == {"type": "text", "text": "Elements you can act on:\n[1] link \"Home\""}


# --- agent: the discovery loop, driven by a scripted model (no API calls) ---

class ScriptedModel:
    """Plays a fixed list of replies. An element is named by the start of its description
    in the list the loop just sent, so a script survives renumbering."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.received = []

    async def reply(self, messages, tools, *, timeout_s):
        self.received.append(messages[-1])
        entry = self.replies.pop(0)
        if isinstance(entry, Exception):
            raise entry
        if entry == "text":
            return ModelReply("end_turn", [SimpleNamespace(type="text", text="Let me look at this page first.")],
                              SCRIPTED_USAGE)
        if entry == "refusal":
            return ModelReply("refusal", [], SCRIPTED_USAGE)
        name, args = entry
        args = dict(args)
        if isinstance(args.get("element"), str):
            args["element"] = _element_number(messages[-1], args["element"])
        call = SimpleNamespace(type="tool_use", id=f"call_{len(self.received)}", name=name, input=args)
        return ModelReply("tool_use", [call], SCRIPTED_USAGE)


SCRIPTED_USAGE = {"input_tokens": 1_200, "cache_write_tokens": 300, "cache_read_tokens": 0, "output_tokens": 80}


def _texts(message) -> list[str]:
    texts = []
    for block in message["content"]:
        if block["type"] == "text":
            texts.append(block["text"])
        elif block["type"] == "tool_result":
            texts.append(block["content"])
    return texts


def _element_number(message, description_start) -> int:
    for text in _texts(message):
        for line in text.splitlines():
            found = re.match(r"\[(\d+)\] (.*)", line)
            if found and found.group(2).startswith(description_start):
                return int(found.group(1))
    raise AssertionError(f"no element starting with {description_start!r} in the list sent to the model")


SIGN_IN = [
    ("type_text", {"element": 'text box, left label "Username:"', "text": "{credential:bank_username}",
                   "reason": "Enter the username"}),
    ("type_text", {"element": "password box", "text": "{credential:bank_password}", "reason": "Enter the password"}),
    ("click", {"element": 'button "Log In"', "reason": "Sign in"}),
]
NOTICE_CHECK = ("assert_visible", {"expected_text": "authorized personnel only", "reason": "Check the sign-on notice"})
STOP = ("report_stuck", {"category": "NO_PROGRESS", "detail": "Stopping the test here"})
TO_CONFIRM_PAGE = [
    *SIGN_IN,
    ("click", {"element": 'link "Member Search"', "reason": "Open member search"}),
    ("type_text", {"element": 'text box, left label "Member ID:"', "text": "{member_id}", "reason": "Enter the member"}),
    ("click", {"element": 'button "Search"', "reason": "Search"}),
    ("click", {"element": 'link "Bill Pay"', "reason": "Open Bill Pay"}),
    ("select_option", {"element": "dropdown", "option_label": "{payee_name}", "reason": "Choose the payee"}),
    ("type_text", {"element": 'text box, left label "Amount:"', "text": "{amount}", "reason": "Enter the amount"}),
    ("click", {"element": 'button "Continue"', "reason": "Continue to confirmation"}),
    ("assert_visible", {"expected_text": "Amount:", "reason": "Check the confirmation page"}),
]
CONFIRM = ("click", {"element": 'button "Confirm Payment"', "reason": "Submit it"})
BALANCE_OUTPUT = OutputParamDefinition(key="checking_balance_before", type=OutputType.STRING,
                                       description="Checking balance before paying")


@pytest.fixture
def discovery(mock_bank_url, storage, run_logger):
    # The real loop, browser and mock bank; only the model is scripted.
    base = dataclasses.replace(_ab_contract(), target_url=f"{mock_bank_url}/login")
    values = {"member_id": "10234", "amount": 50.0, "payee_name": "Sunbelt Electric Co"}

    async def run(*replies, outputs=(), sandbox=False, **options):
        # The environment is set explicitly, so the .env setting never changes a test.
        request = DiscoveryRequest(dataclasses.replace(base, output_definitions=list(outputs)), values)
        model = ScriptedModel(*replies)
        return await discover(request, model, run_logger, sandbox=sandbox, **options), model

    return run


@pytest.mark.anyio
async def test_discovery_records_the_flow_and_stops_before_the_irreversible_step(
    discovery, dashboard_popup, storage, run_logger
):
    dashboard_popup(False)
    result, model = await discovery(*TO_CONFIRM_PAGE, CONFIRM)
    assert result.status == ExecutionStatus.HUMAN_ESCALATED, result.error
    handoff = result.handoff_events[0]
    assert (handoff.trigger_reason, handoff.resolution) == ("IRREVERSIBLE_STEP", HandoffResolution.ABORTED)
    assert model.replies == []
    [saved] = list((storage / "member_servicing_and_bill_pay").iterdir())
    artifact = Artifact.model_validate_json(saved.read_text(encoding="utf-8"))
    verify(artifact, env.artifact_signing_key)
    assert len(artifact.steps) == 13
    assert artifact.steps[-1].safety_tier == SafetyTier.IRREVERSIBLE
    assert [step.input_value for step in artifact.steps if step.action == ActionType.TYPE] == [
        "{credential:bank_username}", "{credential:bank_password}", "{member_id}", "{amount}"]
    events = [line["event_type"] for line in _log_lines(run_logger)]
    assert events[0] == "EXECUTION_STARTED" and events[-2:] == ["EXECUTION_ENDED", "SUMMARY_METRICS"]
    assert env.mock_bank_password.get_secret_value() not in run_logger.log_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_mark_goal_complete_needs_a_passing_assertion_first(discovery, storage):
    done = ("mark_goal_complete", {"summary": "Checked the sign-on page"})
    result, model = await discovery(done, NOTICE_CHECK, done)
    assert result.status == ExecutionStatus.SUCCESS, result.error
    answer = model.received[1]["content"][0]
    assert (answer["is_error"], answer["content"]) == (True, ASSERT_FIRST)
    assert len(list((storage / "member_servicing_and_bill_pay").iterdir())) == 1


@pytest.mark.anyio
async def test_the_step_limit_ends_the_run_without_an_artifact(discovery, storage, monkeypatch):
    monkeypatch.setattr(settings, "discovery_max_steps", 2)
    result, _ = await discovery(NOTICE_CHECK, NOTICE_CHECK)
    assert (result.status, result.error.code) == (ExecutionStatus.HARD_ABORT, "MAX_STEPS")
    assert not storage.exists()


@pytest.mark.anyio
async def test_two_replies_without_an_action_end_the_run_as_stuck(discovery):
    result, model = await discovery("text", "text")
    assert (result.status, result.error.code) == (ExecutionStatus.HARD_ABORT, "STUCK_NO_PROGRESS")
    assert _texts(model.received[1])[0] == NO_ACTION_REPROMPT


@pytest.mark.anyio
async def test_a_refusal_ends_the_run_at_once(discovery):
    result, _ = await discovery("refusal")
    assert (result.status, result.error.code) == (ExecutionStatus.HARD_ABORT, "MODEL_REFUSED")


@pytest.mark.anyio
async def test_report_stuck_ends_the_run_with_its_category(discovery):
    result, _ = await discovery(("report_stuck", {"category": "ERROR_SHOWN", "detail": "The page shows an error"}))
    assert (result.status, result.error.code) == (ExecutionStatus.HARD_ABORT, "STUCK_ERROR_SHOWN")
    assert "The page shows an error" in result.error.message


@pytest.mark.anyio
async def test_a_failed_model_call_is_retried_once(discovery):
    result, model = await discovery(ModelCallFailed("APITimeoutError"), STOP)
    assert result.error.code == "STUCK_NO_PROGRESS"
    assert len(model.received) == 2


@pytest.mark.anyio
async def test_two_failed_model_calls_end_the_run_as_a_technical_failure(discovery):
    result, _ = await discovery(ModelCallFailed("APITimeoutError"), ModelCallFailed("APITimeoutError"))
    assert (result.status, result.error.code) == (ExecutionStatus.TECHNICAL_FAIL, "MODEL_UNAVAILABLE")


@pytest.mark.anyio
async def test_an_overlay_is_closed_but_never_recorded(discovery, dashboard_popup, run_logger):
    dashboard_popup(True)
    await discovery(*SIGN_IN,
                    ("dismiss_overlay", {"element": 'button "Close"', "reason": "Close the notice covering the page"}),
                    STOP)
    events = [line["event_type"] for line in _log_lines(run_logger)]
    assert events.count("OVERLAY_DISMISSED") == 1
    assert events.count("STEP_RECORDED") == 4  # the start page and the three sign-in steps


@pytest.mark.anyio
async def test_a_refused_action_is_told_to_the_model_and_the_run_goes_on(discovery):
    result, model = await discovery(
        ("type_text", {"element": "password box", "text": "guess123", "reason": "Enter the password"}), STOP)
    answer = model.received[1]["content"][0]
    assert answer["is_error"] is True
    assert "password box only takes a secret reference" in answer["content"]
    assert result.error.code == "STUCK_NO_PROGRESS"


@pytest.mark.anyio
async def test_a_tool_result_holds_only_text_and_the_new_page_follows_it(discovery, run_logger):
    # The second real run ended in HTTP 400 at the first error result, which carried the
    # screenshot inside it; the page now travels beside the result, and the refusal is logged.
    await discovery(("type_text", {"element": "password box", "text": "guess123", "reason": "Enter the password"}), STOP)
    lines = _log_lines(run_logger)
    [refused] = [line for line in lines if line["event_type"] == "ACTION_REFUSED"]
    assert (refused["turn"], refused["tool"]) == (1, "type_text")
    decisions = [(line["turn"], line["tool"]) for line in lines if line["event_type"] == "MODEL_ACTION"]
    assert decisions == [(1, "type_text"), (2, "report_stuck")]


@pytest.mark.anyio
async def test_the_page_travels_beside_the_tool_result_not_inside_it(discovery):
    _, model = await discovery(NOTICE_CHECK, STOP)
    message = model.received[1]["content"]
    assert message[0]["type"] == "tool_result" and isinstance(message[0]["content"], str)
    assert [block["type"] for block in message[1:]] == ["image", "text", "text"]


def test_the_estimated_cost_uses_list_prices_with_cheap_cache_reads():
    nothing = dict.fromkeys(["input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens"], 0)
    assert estimated_cost_usd("claude-opus-5", {**nothing, "input_tokens": 1_000_000}) == 5.0
    assert estimated_cost_usd("claude-opus-5", {**nothing, "cache_write_tokens": 1_000_000}) == 6.25
    assert estimated_cost_usd("claude-opus-5", {**nothing, "cache_read_tokens": 1_000_000}) == 0.5
    assert estimated_cost_usd("claude-opus-5", {**nothing, "output_tokens": 1_000_000}) == 25.0
    assert estimated_cost_usd("an-unpriced-model", {**nothing, "input_tokens": 1_000_000}) is None


@pytest.mark.anyio
async def test_each_model_call_logs_its_tokens_and_the_run_logs_the_totals(discovery, run_logger):
    await discovery(NOTICE_CHECK, STOP)
    lines = _log_lines(run_logger)
    per_call = [line for line in lines if line["event_type"] == "MODEL_USAGE"]
    [total] = [line for line in lines if line["event_type"] == "RUN_USAGE"]
    assert [line["turn"] for line in per_call] == [1, 2]
    assert (total["input_tokens"], total["output_tokens"]) == (2 * 1_200, 2 * 80)


EXTRACT_PAGE = ('<table><tr><td>Name:</td><td>Laura Whitfield</td></tr>'
                '<tr><td>Primary Account Balance:</td><td>$2,450.32</td></tr></table>')


@pytest.mark.anyio
async def test_a_value_is_read_by_its_label_and_its_locator_holds_for_another_record(page, recorder):
    await _start_on_html(recorder, page, EXTRACT_PAGE)
    drafted = await recorder.draft_extraction("Primary Account Balance:", "checking_balance_before",
                                              "Read the balance", page, _bank_run())
    assert drafted.value == "$2,450.32"
    assert (drafted.step.action, drafted.step.output_key) == (ActionType.EXTRACT_TEXT, "checking_balance_before")
    first = drafted.step.locators[0]
    assert (first.type, drafted.derived.kinds[0]) == (LocatorType.XPATH, "label")
    assert all("2,450.32" not in locator.value for locator in drafted.step.locators)
    # Another member's page: the label-anchored locator reads that member's value.
    await page.set_content(f"{QUIRKS_DOCTYPE}<html><body>{EXTRACT_PAGE.replace('$2,450.32', '$15,200.45')}</body></html>")
    assert await resolve(page, first, {}).inner_text() == "$15,200.45"


REFUSED_READING_PAGE = ('<table><tr><td>Last Login:</td></tr></table><p>Notes:</p>'
                        '<div>Status:</div><div>Status:</div>')


@pytest.mark.anyio
@pytest.mark.parametrize(
    "label, message",
    [
        pytest.param("  ", "quote the label", id="empty"),
        pytest.param("Missing", "no visible element shows", id="not shown"),
        pytest.param("Status:", "shown by 2 elements", id="shown twice"),
        pytest.param("Last Login:", "no table cell follows", id="nothing after it"),
        pytest.param("Notes:", "no table cell follows", id="not in a table"),
    ],
)
async def test_a_reading_is_refused_with_a_reason_for_the_model(page, recorder, label, message):
    await _start_on_html(recorder, page, REFUSED_READING_PAGE)
    with pytest.raises(ExtractionRefused, match=message):
        await recorder.draft_extraction(label, "checking_balance_before", "Read it", page, _bank_run())


@pytest.mark.anyio
async def test_a_declared_value_is_read_by_its_label_and_returned(discovery, dashboard_popup, storage):
    dashboard_popup(False)
    reading = ("extract_text", {"label": "Primary Account Balance:", "output_key": "checking_balance_before",
                                "reason": "Read the balance before paying"})
    # After the search the member's page is open: read there, then go on to pay.
    result, _ = await discovery(*TO_CONFIRM_PAGE[:6], reading, *TO_CONFIRM_PAGE[6:], CONFIRM,
                                outputs=[BALANCE_OUTPUT])
    assert result.status == ExecutionStatus.HUMAN_ESCALATED, result.error
    assert result.terminal_outputs == {"checking_balance_before": "$2450.32"}
    [saved] = list((storage / "member_servicing_and_bill_pay").iterdir())
    artifact = Artifact.model_validate_json(saved.read_text(encoding="utf-8"))
    [step] = [step for step in artifact.steps if step.action == ActionType.EXTRACT_TEXT]
    assert step.output_key == "checking_balance_before"
    assert "Primary Account Balance:" in step.locators[0].value
    assert all("2450.32" not in locator.value for locator in step.locators)


MONEY_BALANCE_OUTPUT = OutputParamDefinition(key="checking_balance_before", type=OutputType.MONEY, currency="USD",
                                             description="Checking balance before paying")
READ_BALANCE = ("extract_text", {"label": "Primary Account Balance:", "output_key": "checking_balance_before",
                                 "reason": "Read the balance before paying"})


@pytest.mark.anyio
async def test_a_money_value_is_returned_exactly_as_decimal_text(discovery, dashboard_popup):
    dashboard_popup(False)
    result, _ = await discovery(*TO_CONFIRM_PAGE[:6], READ_BALANCE, *TO_CONFIRM_PAGE[6:], CONFIRM,
                                outputs=[MONEY_BALANCE_OUTPUT])
    assert result.status == ExecutionStatus.HUMAN_ESCALATED, result.error
    assert result.terminal_outputs == {"checking_balance_before": "2450.32"}


@pytest.mark.anyio
async def test_a_value_of_the_wrong_kind_is_refused_and_not_recorded(discovery, dashboard_popup, storage):
    dashboard_popup(False)
    wrong = ("extract_text", {"label": "Primary Account Type:", "output_key": "checking_balance_before",
                              "reason": "Read the balance"})
    result, model = await discovery(*TO_CONFIRM_PAGE[:6], wrong, READ_BALANCE, *TO_CONFIRM_PAGE[6:], CONFIRM,
                                    outputs=[MONEY_BALANCE_OUTPUT])
    # The wrong reading is the seventh reply; the loop's answer to it opens the eighth message.
    answer = model.received[7]["content"][0]
    assert answer["is_error"] is True
    assert answer["content"] == ('Refused: checking_balance_before must be an amount in USD, but "checking" is not '
                                 "a USD amount; read it by the label right before that value.")
    assert result.terminal_outputs == {"checking_balance_before": "2450.32"}
    [saved] = list((storage / "member_servicing_and_bill_pay").iterdir())
    artifact = Artifact.model_validate_json(saved.read_text(encoding="utf-8"))
    assert [step.action for step in artifact.steps].count(ActionType.EXTRACT_TEXT) == 1


@pytest.mark.anyio
async def test_a_number_value_is_returned_as_a_number(discovery, dashboard_popup):
    dashboard_popup(False)
    count = OutputParamDefinition(key="restricted_members", type=OutputType.NUMBER,
                                  description="Members with restricted accounts")
    reading = ("extract_text", {"label": "Members with Restricted Accounts", "output_key": "restricted_members",
                                "reason": "Read the count"})
    check = ("assert_visible", {"expected_text": "Figures since system start", "reason": "Check the dashboard"})
    done = ("mark_goal_complete", {"summary": "Read the count"})
    result, _ = await discovery(*SIGN_IN, reading, check, done, outputs=[count])
    assert result.status == ExecutionStatus.SUCCESS, result.error
    assert result.terminal_outputs == {"restricted_members": 1}
    assert type(result.terminal_outputs["restricted_members"]) is int


@pytest.mark.anyio
async def test_the_goal_cant_be_marked_complete_until_every_declared_value_is_read(discovery):
    done = ("mark_goal_complete", {"summary": "Done"})
    result, model = await discovery(NOTICE_CHECK, done, STOP, outputs=[BALANCE_OUTPUT])
    answer = model.received[2]["content"][0]
    assert answer["is_error"] is True
    assert "checking_balance_before" in answer["content"]
    assert result.error.code == "STUCK_NO_PROGRESS"


@pytest.mark.anyio
async def test_the_irreversible_stop_waits_until_every_declared_value_is_read(
    discovery, dashboard_popup, storage, run_logger
):
    # Stopping at the final submission would lose the run, so the model is sent back first.
    dashboard_popup(False)
    result, model = await discovery(*TO_CONFIRM_PAGE, CONFIRM, STOP, outputs=[BALANCE_OUTPUT])
    answer = model.received[-1]["content"][0]
    assert "checking_balance_before" in answer["content"]
    assert result.error.code == "STUCK_NO_PROGRESS"
    assert not storage.exists()
    tiers = [line["safety_tier"] for line in _log_lines(run_logger) if line["event_type"] == "STEP_RECORDED"]
    assert "IRREVERSIBLE" not in tiers


@pytest.mark.anyio
async def test_a_lower_step_limit_can_be_set_for_one_run(discovery):
    result, _ = await discovery(NOTICE_CHECK, NOTICE_CHECK, max_steps=1)
    assert result.error.code == "MAX_STEPS"
    assert "(1)" in result.error.message


class _BrowserReached(Exception):
    pass


def _browser_trap(monkeypatch):
    # Stands in for the browser: reaching it raises, so a test sees whether the run got
    # that far, with no browser started and nothing sent over the network.
    def trap(*args, **kwargs):
        raise _BrowserReached()

    monkeypatch.setattr("src.discovery.agent.BrowserSession", trap)


REMOTE_BANK = "https://bank.example.com/login"
RUN_VALUES = {"member_id": "10234", "amount": 50.0, "payee_name": "Sunbelt Electric Co"}


@pytest.mark.anyio
async def test_a_sandbox_run_is_refused_when_the_bank_is_not_on_this_machine(monkeypatch, storage, run_logger):
    _browser_trap(monkeypatch)
    model = ScriptedModel()
    request = DiscoveryRequest(dataclasses.replace(_ab_contract(), target_url=REMOTE_BANK), RUN_VALUES)
    result = await discover(request, model, run_logger, sandbox=True)
    assert result.status == ExecutionStatus.HARD_ABORT
    assert result.error.code == "SANDBOX_NOT_LOCAL"
    assert "bank.example.com" in result.error.message
    assert model.received == []
    assert not storage.exists()
    events = [line["event_type"] for line in _log_lines(run_logger)]
    assert events == ["EXECUTION_STARTED", "RUN_USAGE", "EXECUTION_ENDED", "SUMMARY_METRICS"]


@pytest.mark.anyio
async def test_a_production_run_is_not_held_to_this_machine(monkeypatch, run_logger):
    _browser_trap(monkeypatch)
    request = DiscoveryRequest(dataclasses.replace(_ab_contract(), target_url=REMOTE_BANK), RUN_VALUES)
    with pytest.raises(_BrowserReached):
        await discover(request, ScriptedModel(), run_logger, sandbox=False)


# Kept last in the file: this test really pays in the shared test bank, which changes the
# member's balance for anything that runs after it in the same session.
NEW_BALANCE_OUTPUT = OutputParamDefinition(key="new_checking_balance", type=OutputType.STRING,
                                           description="Checking balance after paying")


@pytest.mark.anyio
async def test_in_a_sandbox_discovery_performs_the_irreversible_step_and_learns_what_follows(
    discovery, dashboard_popup, storage, run_logger
):
    dashboard_popup(False)
    before = ("extract_text", {"label": "Primary Account Balance:", "output_key": "checking_balance_before",
                               "reason": "Read the balance before paying"})
    after = ("extract_text", {"label": "New Checking Balance:", "output_key": "new_checking_balance",
                              "reason": "Read the balance after paying"})
    paid = ("assert_visible", {"expected_text": "Payment Submitted Successfully", "reason": "Check the payment went through"})
    done = ("mark_goal_complete", {"summary": "Paid the bill and read the balances"})
    result, _ = await discovery(*TO_CONFIRM_PAGE[:6], before, *TO_CONFIRM_PAGE[6:], CONFIRM, paid, after, done,
                                outputs=[BALANCE_OUTPUT, NEW_BALANCE_OUTPUT], sandbox=True)
    assert result.status == ExecutionStatus.SUCCESS, result.error
    outputs = result.terminal_outputs
    assert float(outputs["new_checking_balance"].lstrip("$")) == float(outputs["checking_balance_before"].lstrip("$")) - 50
    [saved] = list((storage / "member_servicing_and_bill_pay").iterdir())
    artifact = Artifact.model_validate_json(saved.read_text(encoding="utf-8"))
    irreversible = next(step for step in artifact.steps if step.safety_tier == SafetyTier.IRREVERSIBLE)
    assert irreversible.checkpoints  # performed, so what it led to is checked too
    assert [step.action for step in artifact.steps[irreversible.sequence_index + 1:]] == [
        ActionType.ASSERT_TEXT, ActionType.EXTRACT_TEXT]
    events = [line["event_type"] for line in _log_lines(run_logger)]
    assert "IRREVERSIBLE_EXECUTED" in events and "DIALOG_ACCEPTED" in events
