"""What may be typed where: a secret goes only into a password box, and a password box
takes nothing but a secret.

One rule for discovery and replay. It sees placeholders, never real values: a secret is
resolved from configuration at the keystroke, after this check has passed.
"""
from collections.abc import Collection
from typing import Optional

from src.types.placeholders import CREDENTIAL_PREFIX, iter_placeholders


def typing_refusal(value: str, *, into_password_box: bool, secret_names: Collection[str]) -> Optional[str]:
    """Why this value can't be typed into this field, worded for the model; None if it can.

    secret_names are the keys of the secret-kind credentials (bank_password). A
    config-kind credential such as the username is not a secret: like an input, it may
    be typed into an ordinary text box, never into a password box.

    A password box takes exactly one secret placeholder with nothing around it, so no
    literal, guess or page-supplied password is ever typed into one or stored. A secret
    placeholder anywhere else would put the password on screen, in the model's next
    screenshot. Reasons name secrets, never their values.
    """
    placeholders = list(iter_placeholders(value))
    secrets_typed = [key for key in (_secret_key(name, secret_names) for name, _, _ in placeholders) if key]

    if into_password_box:
        if len(placeholders) == 1 and secrets_typed and (placeholders[0][1], placeholders[0][2]) == (0, len(value)):
            return None
        if not secret_names:
            return "a password box only takes a secret reference, and this run has none"
        examples = " or ".join(f"{{{CREDENTIAL_PREFIX}:{name}}}" for name in sorted(secret_names))
        return f"a password box only takes a secret reference, exactly {examples}, with nothing else"

    if secrets_typed:
        named = ", ".join(f"{{{CREDENTIAL_PREFIX}:{key}}}" for key in dict.fromkeys(secrets_typed))
        return f"{named} is a secret and can only be typed into a password box"
    return None


def _secret_key(placeholder: str, secret_names: Collection[str]) -> Optional[str]:
    # {credential:bank_password} names the secret bank_password; any other placeholder
    # (an input, a config credential) names none.
    prefix, _, key = placeholder.partition(":")
    return key if prefix == CREDENTIAL_PREFIX and key in secret_names else None
