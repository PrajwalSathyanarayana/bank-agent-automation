"""Reading a value the page shows as the type an output declares.

One reader for both modes: discovery checks a value when the model reads it, replay
converts the value it reads, and replay's payment check reads an amount the same way.
Money is read exactly, as a Decimal, and never passes through floating point. A value
that doesn't fit the rules is refused, never guessed.
"""
import re
from decimal import Decimal
from typing import Union

from src.types.artifact_schema import OutputParamDefinition, OutputType

# The currencies we know how to read: how many decimal places a cent needs, and the symbol
# a page may show before the amount. The ISO code may appear before or after it as well.
_MINOR_UNITS = {"USD": 2}
_SYMBOLS = {"USD": "$"}

# Digits with thousands commas in proper groups of three, then an optional fraction.
# ASCII digits only: a page's decimal comma ("50,00") is refused rather than read as fifty.
_DIGITS = re.compile(r"(?P<whole>[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.(?P<fraction>[0-9]+))?")
_SHOWN_MAX = 80


class UnreadableValue(ValueError):
    """The page's text isn't a value of the declared type. The message quotes the text,
    worded for the model and the run log."""


def read_output(text: str, definition: OutputParamDefinition) -> Union[str, int, float]:
    """The value to return for this output: text as read, a number, or money as plain
    decimal text ("2450.32") in the output's currency."""
    if definition.type == OutputType.MONEY:
        # ":f" writes plain digits, never an exponent.
        return f"{read_money(text, definition.currency or ''):f}"
    if definition.type == OutputType.NUMBER:
        return read_number(text)
    return text


def read_money(text: str, currency: str) -> Decimal:
    """An amount of money shown on the page, exact to the cent.

    Accepts "$2,450.32", "2450.32", "USD 50.00", "50 USD", "-$50.00", "$-50.00" and the
    accountant's "($50.00)" for a negative amount. Fewer decimal places than a cent needs
    are filled in ("$50" is 50.00); more are refused, since that is not an amount.
    """
    if currency not in _MINOR_UNITS:
        raise UnreadableValue(f"there is no rule for reading {currency or 'unnamed'} amounts")
    refused = UnreadableValue(f'"{_shown(text)}" is not a {currency} amount')
    value = text.strip()
    negative = False
    if value.startswith("(") and value.endswith(")"):
        negative, value = True, value[1:-1].strip()
    if value.startswith(currency):
        value = value[len(currency):].strip()
    elif value.endswith(currency):
        value = value[:-len(currency)].strip()
    for part in ("-", _SYMBOLS[currency], "-"):
        # A minus sign may stand before or after the symbol, but only once.
        if part == "-" and value.startswith("-"):
            if negative:
                raise refused
            negative, value = True, value[1:].strip()
        elif part != "-" and value.startswith(part):
            value = value[len(part):].strip()
    found = _DIGITS.fullmatch(value)
    if found is None or len(found["fraction"] or "") > _MINOR_UNITS[currency]:
        raise refused
    amount = Decimal(found["whole"].replace(",", "") + "." + (found["fraction"] or "0"))
    amount = amount.quantize(Decimal(1).scaleb(-_MINOR_UNITS[currency]))
    # A zero is never negative: "-$0.00" is 0.00.
    return -amount if negative and amount else amount


def read_number(text: str) -> Union[int, float]:
    """A plain number shown on the page: "3", "1,204", "-7", "12.5". A whole number stays
    an int; anything with a decimal point is a float. No currency symbols."""
    value = text.strip()
    negative = value.startswith("-")
    found = _DIGITS.fullmatch(value[1:].strip() if negative else value)
    if found is None:
        raise UnreadableValue(f'"{_shown(text)}" is not a number')
    digits = ("-" if negative else "") + found["whole"].replace(",", "")
    if found["fraction"] is None:
        return int(digits)
    return float(f"{digits}.{found['fraction']}")


def _shown(text: str) -> str:
    # Enough of the page's text to recognise it, never a whole page.
    flat = " ".join(text.split())
    return flat if len(flat) <= _SHOWN_MAX else flat[:_SHOWN_MAX - 1] + "…"
