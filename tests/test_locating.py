from decimal import Decimal

import pytest
from playwright.async_api import Error as PlaywrightError

from src.locating.checks import element_wording, find_phrase, is_password_box, phrase_matches, shows_phrase
from src.locating.resolver import UnfillableLocator, css_string, fill, resolve
from src.locating.values import UnreadableValue, read_money, read_number, read_output
from src.types.artifact_schema import OutputParamDefinition, OutputType
from src.types.step_schema import Locator, LocatorType

QUIRKS_DOCTYPE = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">'


async def _set_page(page, body) -> None:
    # Same doctype as the bank's pages, so hand-written pages render in quirks mode too.
    await page.set_content(f"{QUIRKS_DOCTYPE}<html><body>{body}</body></html>")


def _locator(kind, value) -> Locator:
    return Locator(type=kind, value=value, priority=0)


async def _ids(found) -> list[str]:
    return await found.evaluate_all("elements => elements.map(element => element.id)")


# --- what each locator type means on a page ---

@pytest.mark.anyio
async def test_css_locator_is_never_run_as_xpath(page):
    # Without the explicit engine, Playwright would read "//a" as XPath and find the link.
    await _set_page(page, '<a id="t" href="#">Link</a>')
    assert await _ids(resolve(page, _locator(LocatorType.XPATH, "//a"), {})) == ["t"]
    with pytest.raises(PlaywrightError):
        await resolve(page, _locator(LocatorType.CSS, "//a"), {}).count()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param('<table><tr><td id="cell"><a id="t" href="#">Bill Pay</a></td></tr></table>', ["t"],
                     id="the link, not its same-text cell"),
        pytest.param('<a id="t" href="#">Bill&nbsp;Pay</a>', ["t"], id="nbsp counts as a space"),
        pytest.param('<input id="t" type="submit" value="Bill Pay">', ["t"], id="button by its value"),
        pytest.param('<a id="t" href="#">Bill Pay Now</a>', [], id="exact, not contains"),
    ],
)
async def test_text_content_is_an_exact_text_lookup(page, body, expected):
    await _set_page(page, body)
    assert await _ids(resolve(page, _locator(LocatorType.TEXT_CONTENT, "Bill Pay"), {})) == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param('<input id="t" aria-label="Member ID">', id="aria-label"),
        pytest.param('<label for="t">Member ID</label><input id="t">', id="label for"),
        pytest.param('<span id="l">Member ID</span><input id="t" aria-labelledby="l">', id="aria-labelledby"),
        pytest.param('<label>Member ID <input id="t"></label>', id="wrapping label"),
    ],
)
async def test_aria_label_is_the_accessible_name_not_only_the_attribute(page, body):
    await _set_page(page, body)
    assert await _ids(resolve(page, _locator(LocatorType.ARIA_LABEL, "Member ID"), {})) == ["t"]


# --- filling placeholders ---

def test_placeholder_is_filled_and_doubled_braces_become_single():
    locator = _locator(LocatorType.CSS, 'a[href="/member/{member_id}/accounts"][title="{{x}}"]')
    assert fill(locator, {"member_id": "10234"}) == 'a[href="/member/10234/accounts"][title="{x}"]'


def test_css_value_is_escaped_for_its_quoted_string():
    value = 'a"b\\c\nd'
    locator = _locator(LocatorType.CSS, 'a[href="/member/{member_id}"]')
    assert fill(locator, {"member_id": value}) == f'a[href="/member/{css_string(value)}"]'


def test_text_and_label_values_are_filled_without_escaping():
    # Playwright receives these as plain strings, not selectors.
    locator = _locator(LocatorType.TEXT_CONTENT, "Pay {payee_name}")
    assert fill(locator, {"payee_name": 'Say "hi"'}) == 'Pay Say "hi"'


@pytest.mark.parametrize(
    "locator, values, message",
    [
        pytest.param(_locator(LocatorType.XPATH, '//a[@href="/member/{member_id}"]'), {"member_id": 'ab"c'},
                     "double quote", id="xpath value with a double quote"),
        pytest.param(_locator(LocatorType.CSS, 'a[href="/member/{member_id}"]'), {},
                     "no value for {member_id}", id="missing value"),
        pytest.param(_locator(LocatorType.CSS, "a/{member_id}"), {"member_id": "ab9"},
                     "double-quoted string", id="placeholder outside a quoted string"),
    ],
)
def test_filling_refuses_rather_than_guess(locator, values, message):
    with pytest.raises(UnfillableLocator, match=message) as refused:
        fill(locator, values)
    for value in values.values():
        assert value not in str(refused.value)


@pytest.mark.anyio
async def test_an_injection_shaped_value_matches_nothing(page):
    await _set_page(page, '<a id="one" href="/member/1/accounts">One</a><a id="two" href="/member/2/accounts">Two</a>')
    injection = '"], a[href*="'
    # Unescaped, the value would close the string and add a selector matching every link.
    assert await page.locator('css=a[href="/member/"], a[href*="/accounts"]').count() == 2
    locator = _locator(LocatorType.CSS, 'a[href="/member/{member_id}/accounts"]')
    assert await _ids(resolve(page, locator, {"member_id": injection})) == []


@pytest.mark.parametrize(
    "value, escaped",
    [
        pytest.param('a"b', 'a\\"b', id="quote"),
        pytest.param("a\\b", "a\\\\b", id="backslash"),
        pytest.param("a\nb", "a\\a b", id="line break as a hex escape"),
        pytest.param("a\x7fb", "a\\7f b", id="delete character as a hex escape"),
    ],
)
def test_css_string_escaping(value, escaped):
    assert css_string(value) == escaped


# --- what an element says, for the safety classifier ---

@pytest.mark.anyio
async def test_element_wording_reads_text_button_value_and_labels(page):
    await _set_page(page, '<input id="b" type="submit" value="Confirm Payment" title="Pay now">'
                          '<a id="l" href="#" aria-label="Close dialog">x</a>')
    assert await element_wording(await page.query_selector("#b")) == ["Confirm Payment", "Pay now"]
    assert await element_wording(await page.query_selector("#l")) == ["x", "Close dialog"]


@pytest.mark.anyio
async def test_element_wording_never_reads_a_password_or_typed_value(page):
    await _set_page(page, '<input id="p" type="password"><input id="t" type="text">')
    await page.fill("#p", "Zq9-not-the-real-password")
    await page.fill("#t", "10234")
    assert await element_wording(await page.query_selector("#p")) == []
    assert await element_wording(await page.query_selector("#t")) == []


# --- what kind of field typing goes into ---

@pytest.mark.anyio
@pytest.mark.parametrize(
    "field, expected",
    [
        pytest.param('<input id="f" type="password">', True, id="password"),
        pytest.param('<input id="f" type="PASSWORD">', True, id="type in capitals"),
        pytest.param('<input id="f" type="text">', False, id="text box"),
        pytest.param('<input id="f" type="text" style="-webkit-text-security: disc">', False,
                     id="masked only by styling"),
        pytest.param('<input id="f">', False, id="no type"),
        pytest.param('<div id="f" type="password">x</div>', False, id="div with a type attribute"),
        pytest.param('<input id="f" type="submit" value="Log In">', False, id="submit button"),
    ],
)
async def test_is_password_box_reads_the_element_type(page, field, expected):
    await _set_page(page, field)
    assert await is_password_box(await page.query_selector("#f")) is expected


@pytest.mark.anyio
async def test_a_password_box_switched_to_text_is_no_longer_one(page):
    # A "show password" toggle: the live type decides, not the one the page loaded with.
    await _set_page(page, '<input id="f" type="password">')
    field = await page.query_selector("#f")
    await page.evaluate("document.getElementById('f').type = 'text'")
    assert not await is_password_box(field)


# --- assertion text: one rule for discovery and replay ---

@pytest.mark.parametrize(
    "text, phrase, matches",
    [
        pytest.param("Payment submitted - Ref 88121", "Payment submitted", True, id="data after the phrase"),
        pytest.param("Ref 88121: payment submitted", "Payment submitted", True, id="data before, any case"),
        pytest.param("Payment of $50.00 submitted", "Payment submitted", False, id="data in the middle"),
        pytest.param("Payment", "Pay", False, id="not inside a longer word"),
        pytest.param("Payment   submitted", "payment submitted", True, id="nbsp and runs of spaces"),
        pytest.param("Amount: $50.00", "Amount:", True, id="phrase ending in punctuation"),
        pytest.param("Total Amount:", "Amount:", True, id="at the end of the text"),
        pytest.param("Anything at all", "   ", False, id="empty phrase"),
    ],
)
def test_phrase_matches_whole_words_in_any_case(text, phrase, matches):
    assert phrase_matches(text, phrase) is matches


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body, shown",
    [
        pytest.param('<div id="t">Payment submitted</div>', True, id="visible"),
        pytest.param('<input id="t" type="submit" value="Payment submitted">', True, id="button by its value"),
        pytest.param('<div id="t" style="display:none">Payment submitted</div>', False, id="display none"),
        pytest.param('<div id="t" style="visibility:hidden">Payment submitted</div>', False, id="visibility hidden"),
        pytest.param('<div id="t" style="width:0;height:0;overflow:hidden">Payment submitted</div>', False,
                     id="zero size"),
    ],
)
async def test_shows_phrase_needs_a_visible_element(page, body, shown):
    # A hidden element's visible-text property falls back to its hidden text, so this
    # must fail on visibility, not on the text.
    await _set_page(page, body)
    assert await shows_phrase(await page.query_selector("#t"), "payment submitted") is shown


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param('<div id="outer"><div id="inner"><b>Payment</b> submitted</div></div>', ["inner"],
                     id="innermost element only"),
        pytest.param('<div id="hidden" style="display:none">Payment submitted</div>'
                     '<div id="shown">Payment submitted</div>', ["shown"], id="hidden copy ignored"),
        pytest.param('<div id="one">Payment submitted</div><div id="two">Payment submitted</div>',
                     ["one", "two"], id="both visible copies"),
    ],
)
async def test_find_phrase_returns_the_innermost_visible_matches(page, body, expected):
    await _set_page(page, body)
    found = await find_phrase(page, "Payment submitted")
    assert [await handle.get_attribute("id") for handle in found] == expected


# --- reading a value as the type an output declares ---

@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("$2450.32", "2450.32", id="as the bank shows it"),
        pytest.param("$2,450.32", "2450.32", id="thousands comma"),
        pytest.param("$1,234,567.89", "1234567.89", id="several groups"),
        pytest.param("2450.32", "2450.32", id="no symbol"),
        pytest.param("  $50.00 ", "50.00", id="spaces around"),
        pytest.param("\xa0$50.00", "50.00", id="non-breaking space"),
        pytest.param("$ 50.00", "50.00", id="space after the symbol"),
        pytest.param("$50", "50.00", id="whole dollars"),
        pytest.param("$50.5", "50.50", id="one decimal place"),
        pytest.param("USD 50.00", "50.00", id="code before"),
        pytest.param("50.00 USD", "50.00", id="code after"),
        pytest.param("-$50.00", "-50.00", id="minus before the symbol"),
        pytest.param("$-50.00", "-50.00", id="minus after the symbol"),
        pytest.param("($50.00)", "-50.00", id="brackets"),
        pytest.param("-$0.00", "0.00", id="zero is never negative"),
    ],
)
def test_money_is_read_exactly_to_the_cent(text, expected):
    assert f"{read_money(text, 'USD'):f}" == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("$", id="symbol only"),
        pytest.param("Primary Account Balance: $2450.32", id="label read with the value"),
        pytest.param("$2,45.32", id="broken thousands group"),
        pytest.param("50,00", id="decimal comma"),
        pytest.param("$2450.321", id="finer than a cent"),
        pytest.param("€50.00", id="another currency's symbol"),
        pytest.param("--$50.00", id="two minus signs"),
        pytest.param("($-50.00)", id="brackets and a minus"),
        pytest.param("$50.00)", id="unbalanced bracket"),
        pytest.param("1.2.3", id="two decimal points"),
        pytest.param("$٥٠.00", id="non-ASCII digits"),
    ],
)
def test_anything_else_is_refused_as_money_not_guessed(text):
    with pytest.raises(UnreadableValue, match="is not a USD amount"):
        read_money(text, "USD")


def test_money_adds_up_exactly():
    # Why money is never a float: 0.10 + 0.20 is not 0.30 in floating point.
    assert read_money("$0.10", "USD") + read_money("$0.20", "USD") == Decimal("0.30")


def test_only_currencies_with_a_reading_rule_are_read():
    with pytest.raises(UnreadableValue, match="no rule for reading EUR amounts"):
        read_money("50.00", "EUR")


@pytest.mark.parametrize(
    "text, expected",
    [pytest.param("3", 3, id="whole"), pytest.param("1,204", 1204, id="thousands comma"),
     pytest.param("-7", -7, id="negative"), pytest.param(" 12.5 ", 12.5, id="decimal")],
)
def test_a_number_stays_whole_unless_it_has_a_decimal_point(text, expected):
    value = read_number(text)
    assert value == expected and type(value) is type(expected)


@pytest.mark.parametrize("text", ["", "-", "$3", "3 items", "1,20"])
def test_anything_else_is_refused_as_a_number(text):
    with pytest.raises(UnreadableValue, match="is not a number"):
        read_number(text)


@pytest.mark.parametrize(
    "definition, text, expected",
    [
        pytest.param(OutputParamDefinition(key="balance", type=OutputType.MONEY, description="Balance",
                                           currency="USD"), "$2,450.32", "2450.32", id="money as decimal text"),
        pytest.param(OutputParamDefinition(key="count", type=OutputType.NUMBER, description="Count"),
                     "1,204", 1204, id="number"),
        pytest.param(OutputParamDefinition(key="kind", type=OutputType.STRING, description="Account type"),
                     "Checking", "Checking", id="text as read"),
    ],
)
def test_an_output_is_returned_as_its_declared_type(definition, text, expected):
    assert read_output(text, definition) == expected


def test_a_refusal_quotes_the_page_text_but_never_a_whole_page():
    with pytest.raises(UnreadableValue) as refused:
        read_money("x" * 500, "USD")
    assert len(str(refused.value)) < 120
