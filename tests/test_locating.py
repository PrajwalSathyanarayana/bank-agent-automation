import pytest
from playwright.async_api import Error as PlaywrightError

from src.locating.checks import element_wording
from src.locating.resolver import UnfillableLocator, css_string, fill, resolve
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
