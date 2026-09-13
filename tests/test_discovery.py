import dataclasses
import json
import struct

import pytest
from playwright.async_api import expect

from src.config.env import env
from src.config.settings import settings
from src.discovery.perception import (
    _COLLECTOR_SOURCE,
    DESCRIPTION_MAX,
    NO_ELEMENTS,
    OUTSIDE_MARKER,
    Box,
    ElementFacts,
    Observation,
    PageElement,
    UnknownElement,
    _element_indexes,
    choose_elements,
    describe,
    describe_options,
    element_kind,
    observe,
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
