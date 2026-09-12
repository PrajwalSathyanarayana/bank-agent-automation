from urllib.parse import urlparse

import pytest

from src.config.env import env
from src.safety.allowlist import (
    ALLOWLIST,
    AllowlistConfig,
    AllowlistViolation,
    check_action_type,
    check_domain,
    enforce_safety,
)
from src.safety.classifier import SafetyEscalation, classify, verify_tier
from src.safety.redactor import REDACTED, redact_dict, redact_text
from src.types.step_schema import ActionType, Locator, LocatorType, SafetyTier, Step

MOCK_BANK_HOSTNAME = urlparse(env.mock_bank_base_url).hostname


def test_default_allowlist_derives_domain_from_mock_bank_base_url():
    # Derived from env, not hardcoded, so this test verifies the derivation
    # logic rather than pinning a specific .env value.
    assert ALLOWLIST.allowed_domains == [MOCK_BANK_HOSTNAME]


def test_default_allowlist_permits_all_action_types():
    assert set(ALLOWLIST.allowed_action_types) == set(ActionType)


def test_check_domain_passes_for_allowed_domain():
    check_domain(f"{env.mock_bank_base_url}/dashboard")  # should not raise


def test_check_domain_raises_for_disallowed_domain():
    with pytest.raises(AllowlistViolation):
        check_domain("http://evil-external-site.com/phishing")


def test_check_action_type_passes_for_permitted_action():
    check_action_type(ActionType.CLICK)  # should not raise


def test_check_action_type_raises_when_restricted_by_custom_config():
    restrictive_config = AllowlistConfig(
        allowed_domains=[MOCK_BANK_HOSTNAME],
        allowed_action_types=[ActionType.CLICK, ActionType.TYPE],
    )
    with pytest.raises(AllowlistViolation):
        check_action_type(ActionType.NAVIGATE, config=restrictive_config)


def test_enforce_safety_raises_on_domain_violation_even_if_action_allowed():
    with pytest.raises(AllowlistViolation):
        enforce_safety("http://evil-external-site.com/phishing", ActionType.CLICK)


def test_enforce_safety_raises_on_action_violation_even_if_domain_allowed():
    restrictive_config = AllowlistConfig(
        allowed_domains=[MOCK_BANK_HOSTNAME],
        allowed_action_types=[ActionType.CLICK],
    )
    with pytest.raises(AllowlistViolation):
        enforce_safety(f"{env.mock_bank_base_url}/billpay", ActionType.NAVIGATE, config=restrictive_config)


def test_enforce_safety_passes_when_both_checks_satisfied():
    enforce_safety(f"{env.mock_bank_base_url}/dashboard", ActionType.CLICK)  # should not raise


# --- classifier.py ---

def _step(description: str, locator_text: str | None = None, safety_tier: SafetyTier = SafetyTier.SAFE) -> Step:
    locators = [Locator(type=LocatorType.CSS, value="#target", priority=0)]
    if locator_text is not None:
        locators.append(Locator(type=LocatorType.TEXT_CONTENT, value=locator_text, priority=1))
    return Step(
        sequence_index=0,
        action=ActionType.CLICK,
        description=description,
        locators=locators,
        safety_tier=safety_tier,
    )


# classify() only ever inspects urlparse(url).path — never the domain — so
# these use bare paths (no fake domain) and an arbitrary placeholder member
# ID ("ANY-ID") to make clear there's no dependency on real fixture data
# from members.json. Any ID would produce identical results.

def test_classify_read_only_step_is_safe():
    step = _step("Read member detail")
    assert classify(step, "/member/ANY-ID") == SafetyTier.SAFE


def test_classify_member_edit_is_risky():
    step = _step("Submit profile update")
    assert classify(step, "/member/ANY-ID/edit") == SafetyTier.RISKY


def test_classify_billpay_form_submission_is_risky():
    step = _step("Submit payment form")
    assert classify(step, "/billpay") == SafetyTier.RISKY


def test_classify_confirm_payment_click_is_irreversible():
    step = _step("Click confirm payment button", locator_text="Confirm Payment")
    assert classify(step, "/billpay/confirm") == SafetyTier.IRREVERSIBLE


def test_classify_cancel_click_on_confirm_page_is_not_irreversible():
    step = _step("Click cancel link", locator_text="Cancel")
    assert classify(step, "/billpay/confirm") == SafetyTier.SAFE


def test_verify_tier_raises_when_artifact_underdeclares_risk():
    step = _step("Click confirm payment button", locator_text="Confirm Payment", safety_tier=SafetyTier.SAFE)
    with pytest.raises(SafetyEscalation):
        verify_tier(step, "/billpay/confirm")


def test_verify_tier_passes_when_declared_tier_matches():
    step = _step("Click confirm payment button", locator_text="Confirm Payment", safety_tier=SafetyTier.IRREVERSIBLE)
    assert verify_tier(step, "/billpay/confirm") == SafetyTier.IRREVERSIBLE


def test_verify_tier_passes_when_declared_tier_is_overcautious():
    step = _step("Read member detail", safety_tier=SafetyTier.RISKY)
    assert verify_tier(step, "/member/ANY-ID") == SafetyTier.SAFE


# --- redactor.py ---

def test_redact_dict_does_not_mutate_input():
    original = {"password": "hunter2", "member_id": "ANY-ID"}
    snapshot = dict(original)
    redact_dict(original)
    assert original == snapshot


def test_redact_dict_redacts_known_credential_keys():
    data = {"password": "hunter2", "api_key": "sk-abc123", "member_id": "ANY-ID"}
    result = redact_dict(data)
    assert result["password"] == REDACTED
    assert result["api_key"] == REDACTED
    assert result["member_id"] == "ANY-ID"  # not sensitive, left untouched


def test_redact_dict_leaves_non_sensitive_fields_untouched():
    data = {"account_id": "CHK-ANY-01", "account_type": "checking", "balance": 100.0}
    result = redact_dict(data)
    assert result == data


def test_redact_dict_collapses_nested_address_to_single_placeholder():
    data = {
        "first_name": "Jane",
        "address": {"street": "1 Any St", "city": "Anytown", "state": "AZ", "zip": "00000"},
    }
    result = redact_dict(data)
    assert result["first_name"] == REDACTED
    assert result["address"] == REDACTED  # whole nested structure collapsed, not per sub-field


def test_redact_dict_recurses_into_list_of_dicts_without_over_redacting():
    data = {
        "accounts": [
            {"account_id": "CHK-ANY-01", "account_type": "checking", "balance": 100.0},
            {"account_id": "SAV-ANY-01", "account_type": "savings", "balance": 500.0},
        ]
    }
    result = redact_dict(data)
    assert result == data  # no sensitive keys inside account records


def test_redact_dict_case_insensitive_key_matching():
    data = {"Email": "jane.doe@example.com"}
    result = redact_dict(data)
    assert result["Email"] == REDACTED


def test_redact_text_scrubs_email():
    assert redact_text("contact jane.doe@example.com for details") == f"contact {REDACTED} for details"


def test_redact_text_scrubs_phone_number():
    assert redact_text("call (602) 555-0142 now") == f"call {REDACTED} now"


def test_redact_text_scrubs_ssn_pattern():
    assert redact_text("SSN on file: 123-45-6789") == "SSN on file: " + REDACTED


def test_redact_dict_applies_regex_backstop_to_non_sensitive_key():
    # "notes" is not in SENSITIVE_KEYS, but the value itself leaks an
    # email — the regex backstop should still catch it (D023's hybrid).
    data = {"notes": "please follow up with jane.doe@example.com"}
    result = redact_dict(data)
    assert REDACTED in result["notes"]
    assert "jane.doe@example.com" not in result["notes"]
