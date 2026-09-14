import json
from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest
from pydantic import SecretStr

from src.config.env import env
from src.safety.allowlist import (
    ALLOWLIST,
    AllowlistConfig,
    AllowlistViolation,
    check_action_type,
    check_domain,
    check_route,
    enforce_safety,
)
from src.safety.classifier import SafetyEscalation, classify, verify_tier
from src.safety.integrity import (
    IntegrityCheckFailed,
    canonical_bytes,
    compute_signature,
    sign,
    verify,
)
from src.safety.redactor import REDACTED, redact_dict, redact_text, scrub_known_values, sensitive_patterns_in
from src.safety.sandbox import sandbox_refusal
from src.safety.secret_typing import typing_refusal
from src.types.artifact_schema import (
    Artifact,
    ArtifactMetadata,
    CredentialDefinition,
    CredentialKind,
    InputParamDefinition,
    ParamType,
)
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


PAGES = ["/login", "/member/*", "/billpay"]


@pytest.mark.parametrize(
    "path, allowed",
    [
        pytest.param("/member/10234", True, id="a listed pattern"),
        pytest.param("/billpay?payee=P001#top", True, id="query and fragment ignored"),
        pytest.param("/member/10234/edit", False, id="a page not listed"),
        pytest.param("", False, id="no path is the root, which isn't listed"),
    ],
)
def test_check_route_allows_only_the_capabilitys_pages(path, allowed):
    url = f"{env.mock_bank_base_url}{path}"
    if allowed:
        check_route(url, PAGES)  # should not raise
    else:
        with pytest.raises(AllowlistViolation, match="not one this capability may visit"):
            check_route(url, PAGES)


def test_an_empty_page_list_allows_any_page_on_the_host():
    check_route(f"{env.mock_bank_base_url}/member/10234/edit", [])  # should not raise


def test_enforce_safety_raises_on_a_page_not_allowed_even_on_the_right_domain():
    with pytest.raises(AllowlistViolation, match="'/member/10234/edit'"):
        enforce_safety(f"{env.mock_bank_base_url}/member/10234/edit", ActionType.CLICK, allowed_paths=PAGES)


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
    assert classify(step, "/member/ANY-ID", element_wording=[]) == SafetyTier.SAFE


def test_classify_member_edit_is_risky():
    step = _step("Submit profile update")
    assert classify(step, "/member/ANY-ID/edit", element_wording=[]) == SafetyTier.RISKY


def test_classify_billpay_form_submission_is_risky():
    step = _step("Submit payment form")
    assert classify(step, "/billpay", element_wording=[]) == SafetyTier.RISKY


def test_classify_confirm_payment_click_is_irreversible():
    step = _step("Click confirm payment button", locator_text="Confirm Payment")
    assert classify(step, "/billpay/confirm", element_wording=[]) == SafetyTier.IRREVERSIBLE


def test_classify_cancel_click_on_confirm_page_is_not_irreversible():
    step = _step("Click cancel link", locator_text="Cancel")
    assert classify(step, "/billpay/confirm", element_wording=["Cancel"]) == SafetyTier.SAFE


def test_classify_requires_the_elements_own_wording():
    # Keyword-only and required: no caller can quietly leave out the strongest signal.
    with pytest.raises(TypeError):
        classify(_step("Read member detail"), "/member/ANY-ID")


def test_element_wording_alone_makes_confirm_payment_irreversible():
    # Regression: the page repeats "Confirm Payment", so the text locator was rejected,
    # and the model's reason doesn't name the button. Its own value must still decide.
    step = _step("Submit it")
    assert classify(step, "/billpay/confirm", element_wording=["Confirm Payment"]) == SafetyTier.IRREVERSIBLE


def test_any_locator_value_is_a_signal_not_only_text_locators():
    step = Step(
        sequence_index=0,
        action=ActionType.CLICK,
        description="Submit it",
        locators=[Locator(type=LocatorType.CSS, priority=0,
                          value='form.actions input[type="submit"][value="Confirm Payment"]')],
    )
    assert classify(step, "/billpay/confirm", element_wording=[]) == SafetyTier.IRREVERSIBLE


def test_verify_tier_raises_when_artifact_underdeclares_risk():
    step = _step("Click confirm payment button", locator_text="Confirm Payment", safety_tier=SafetyTier.SAFE)
    with pytest.raises(SafetyEscalation):
        verify_tier(step, "/billpay/confirm", element_wording=[])


def test_verify_tier_reads_the_found_elements_wording_too():
    step = _step("Submit it", safety_tier=SafetyTier.SAFE)
    with pytest.raises(SafetyEscalation):
        verify_tier(step, "/billpay/confirm", element_wording=["Confirm Payment"])


def test_verify_tier_passes_when_declared_tier_matches():
    step = _step("Click confirm payment button", locator_text="Confirm Payment", safety_tier=SafetyTier.IRREVERSIBLE)
    assert verify_tier(step, "/billpay/confirm", element_wording=[]) == SafetyTier.IRREVERSIBLE


def test_verify_tier_passes_when_declared_tier_is_overcautious():
    step = _step("Read member detail", safety_tier=SafetyTier.RISKY)
    assert verify_tier(step, "/member/ANY-ID", element_wording=[]) == SafetyTier.SAFE


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


@pytest.mark.parametrize("text", ["call 602-555-0142 now", "Tel: 6025550142", "(602) 555-0142."])
def test_redact_text_scrubs_phone_numbers_in_their_usual_forms(text):
    assert "555" not in redact_text(text)


def test_digits_inside_a_longer_token_are_not_a_phone_number():
    # A real signature that was corrupted in the log before the pattern needed a token of its own.
    signature = "9c7233a39d653e94983bee0f30410a93587be6ee884607320325369bb12b8469"
    assert redact_text(signature) == signature
    assert sensitive_patterns_in(f"saved {signature}") == []


def test_redact_text_scrubs_ssn_pattern():
    assert redact_text("SSN on file: 123-45-6789") == "SSN on file: " + REDACTED


def test_redact_dict_applies_regex_backstop_to_non_sensitive_key():
    # "notes" is not in SENSITIVE_KEYS, but the value itself leaks an
    # email — the regex backstop should still catch it.
    data = {"notes": "please follow up with jane.doe@example.com"}
    result = redact_dict(data)
    assert REDACTED in result["notes"]
    assert "jane.doe@example.com" not in result["notes"]


# --- exact-value scrub of known secrets ---

def test_scrub_replaces_secret_inside_a_sentence():
    result = scrub_known_values("typing failed near s3cret-VALUE on step 3", ["s3cret-VALUE"])
    assert result == f"typing failed near {REDACTED} on step 3"


def test_scrub_reaches_nested_dicts_and_lists():
    data = {"error": {"lines": ["ok", "token s3cret-VALUE leaked"]}}
    result = scrub_known_values(data, ["s3cret-VALUE"])
    assert result == {"error": {"lines": ["ok", f"token {REDACTED} leaked"]}}


def test_scrub_leaves_other_text_untouched():
    data = {"step_id": "s1", "status": "PASSED", "count": 3}
    assert scrub_known_values(data, ["s3cret-VALUE"]) == data


def test_scrub_ignores_an_empty_secret():
    assert scrub_known_values("nothing to hide", [""]) == "nothing to hide"


def test_scrub_replaces_longer_secret_first():
    # If "abc" were replaced first, "abcdef" would leave "def" behind.
    result = scrub_known_values("value=abcdef", ["abc", "abcdef"])
    assert result == f"value={REDACTED}"
    assert "def" not in result


# --- integrity fingerprint (keyed HMAC-SHA256) ---

TEST_KEY = SecretStr("test-signing-key-0123456789-abcdef")
OTHER_KEY = SecretStr("other-signing-key-0123456789-abcdef")


def _unsigned_artifact() -> Artifact:
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    return Artifact(
        metadata=ArtifactMetadata(
            capability="member_servicing_and_bill_pay",
            description="For member {member_id}, pay {amount} to {payee_name}.",
            version="1.0.0",
            target_url="http://localhost:5000/",
            created_timestamp=now,
            last_updated_timestamp=now,
        ),
        input_parameters=[
            InputParamDefinition(key="member_id", type=ParamType.STRING, description="Member ID"),
            InputParamDefinition(key="amount", type=ParamType.NUMBER, description="Amount"),
            InputParamDefinition(key="payee_name", type=ParamType.STRING, description="Payee"),
        ],
        credentials=[
            CredentialDefinition(key="bank_password", kind=CredentialKind.SECRET, description="Password")
        ],
        steps=[
            Step(
                sequence_index=0,
                action=ActionType.TYPE,
                description="Type the password",
                locators=[Locator(type=LocatorType.CSS, value="input[name='password']", priority=0)],
                input_value="{credential:bank_password}",
            )
        ],
    )


def _signed_artifact() -> Artifact:
    return sign(_unsigned_artifact(), TEST_KEY)


def _hand_edited(artifact: Artifact, edit) -> Artifact:
    # Mirrors a real hand edit: change the saved JSON, then load it again.
    data = artifact.model_dump(mode="json")
    edit(data)
    return Artifact.model_validate(data)


def test_signed_artifact_verifies():
    verify(_signed_artifact(), TEST_KEY)  # should not raise


def test_signing_does_not_change_the_original():
    unsigned = _unsigned_artifact()
    sign(unsigned, TEST_KEY)
    assert unsigned.metadata.integrity_hash is None


def test_same_artifact_always_gets_the_same_signature():
    # One artifact signed twice; building two would give each its own random IDs.
    unsigned = _unsigned_artifact()
    assert sign(unsigned, TEST_KEY).metadata.integrity_hash == sign(unsigned, TEST_KEY).metadata.integrity_hash


def test_signature_survives_saving_and_loading_as_json():
    loaded = Artifact.model_validate_json(_signed_artifact().model_dump_json(indent=2))
    verify(loaded, TEST_KEY)  # should not raise


def test_unsigned_artifact_fails_verification():
    with pytest.raises(IntegrityCheckFailed, match="unsigned"):
        verify(_unsigned_artifact(), TEST_KEY)


def test_wrong_key_fails_verification():
    with pytest.raises(IntegrityCheckFailed, match="does not match"):
        verify(_signed_artifact(), OTHER_KEY)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["metadata"].update(description="Pay {amount} to {payee_name}."),
        lambda d: d["metadata"].update(capability="another_capability"),
        lambda d: d["metadata"].update(version="1.0.1"),
        lambda d: d["metadata"].update(target_url="http://localhost:5000/login"),
        lambda d: d["metadata"].update(tenant_override_url="http://localhost:5000/other"),
        lambda d: d["metadata"].update(artifact_id="00000000-0000-4000-8000-000000000000"),
        lambda d: d["metadata"].update(author="Someone"),
        lambda d: d["input_parameters"][1].update(type="string"),
        lambda d: d["credentials"][0].update(kind="config"),
        lambda d: d["steps"][0]["locators"][0].update(value="input[name='username']"),
        lambda d: d["steps"][0].update(safety_tier="IRREVERSIBLE"),
        lambda d: d["global_assertions"].append({"type": "final_url_match", "value": "/dashboard"}),
    ],
    ids=[
        "description", "capability", "version", "target_url", "tenant_override_url",
        "artifact_id", "author", "input_type", "credential_kind", "locator",
        "safety_tier", "global_assertion",
    ],
)
def test_editing_a_signed_field_fails_verification(edit):
    with pytest.raises(IntegrityCheckFailed, match="does not match"):
        verify(_hand_edited(_signed_artifact(), edit), TEST_KEY)


def test_editing_timestamps_still_verifies():
    def edit(d):
        d["metadata"]["created_timestamp"] = "2030-01-01T00:00:00Z"
        d["metadata"]["last_updated_timestamp"] = "2030-01-02T00:00:00Z"

    verify(_hand_edited(_signed_artifact(), edit), TEST_KEY)  # should not raise


def test_canonical_bytes_leave_out_the_signature_and_timestamps():
    content = json.loads(canonical_bytes(_signed_artifact()))
    assert not {"integrity_hash", "created_timestamp", "last_updated_timestamp"} & set(content["metadata"])


def test_canonical_bytes_are_compact_with_sorted_keys():
    raw = canonical_bytes(_signed_artifact()).decode("utf-8")
    content = json.loads(raw)
    assert raw == json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_failure_message_reveals_neither_key_nor_correct_signature():
    tampered = _hand_edited(_signed_artifact(), lambda d: d["metadata"].update(version="9.9.9"))
    with pytest.raises(IntegrityCheckFailed) as exc_info:
        verify(tampered, TEST_KEY)
    message = str(exc_info.value)
    assert TEST_KEY.get_secret_value() not in message
    assert compute_signature(tampered, TEST_KEY) not in message


# --- what may be typed where ---

SECRET_NAMES = {"bank_password"}


@pytest.mark.parametrize(
    "value, into_password_box, allowed",
    [
        pytest.param("{credential:bank_password}", True, True, id="secret into password box"),
        pytest.param("Zq9-not-the-real-password", True, False, id="literal into password box"),
        pytest.param("{member_id}", True, False, id="input into password box"),
        pytest.param("{credential:bank_username}", True, False, id="config credential into password box"),
        pytest.param(" {credential:bank_password}", True, False, id="space before the secret"),
        pytest.param("{credential:bank_password}x", True, False, id="text after the secret"),
        pytest.param("{credential:bank_password}{credential:bank_password}", True, False, id="secret twice"),
        pytest.param("", True, False, id="empty into password box"),
        pytest.param("{{credential:bank_password}}", True, False, id="doubled braces are literal text"),
        pytest.param("{credential:bank_password}", False, False, id="secret into text box"),
        pytest.param("Pass: {credential:bank_password}", False, False, id="secret inside other text"),
        pytest.param("{credential:bank_username}", False, True, id="config credential into text box"),
        pytest.param("{member_id}", False, True, id="input into text box"),
        pytest.param("50.00", False, True, id="literal into text box"),
        pytest.param("", False, True, id="empty into text box"),
    ],
)
def test_a_password_box_takes_exactly_one_secret_and_a_secret_goes_nowhere_else(value, into_password_box, allowed):
    refusal = typing_refusal(value, into_password_box=into_password_box, secret_names=SECRET_NAMES)
    assert (refusal is None) == allowed


@pytest.mark.parametrize(
    "value, into_password_box, secret_names, expected",
    [
        pytest.param("x", True, {"pin", "bank_password"},
                     "a password box only takes a secret reference, exactly {credential:bank_password} or "
                     "{credential:pin}, with nothing else", id="names every secret, sorted"),
        pytest.param("x", True, set(), "a password box only takes a secret reference, and this run has none",
                     id="run with no secrets"),
        pytest.param("{credential:bank_password}", False, SECRET_NAMES,
                     "{credential:bank_password} is a secret and can only be typed into a password box",
                     id="secret outside a password box"),
    ],
)
def test_refusal_wording(value, into_password_box, secret_names, expected):
    assert typing_refusal(value, into_password_box=into_password_box, secret_names=secret_names) == expected


# --- where a sandbox may be ---

@pytest.mark.parametrize(
    "url, allowed",
    [
        pytest.param("http://localhost:5000/login", True, id="localhost"),
        pytest.param("http://LOCALHOST:5000/", True, id="localhost in capitals"),
        pytest.param("http://127.0.0.1:5000/", True, id="IPv4 loopback"),
        pytest.param("http://[::1]:5000/", True, id="IPv6 loopback"),
        pytest.param("https://bank.example.com/login", False, id="remote bank"),
        pytest.param("http://localhost.bank.example.com/", False, id="localhost as a subdomain"),
        pytest.param("http://localhost@bank.example.com/", False, id="localhost as user info"),
        pytest.param("http://bank.example.com/?next=http://localhost/", False, id="localhost in the query"),
        pytest.param("http://10.0.0.5:5000/", False, id="another machine on the network"),
        pytest.param("http://0.0.0.0:5000/", False, id="all interfaces"),
        pytest.param("/login", False, id="no host"),
        pytest.param("http://[::1", False, id="unreadable address"),
    ],
)
def test_a_sandbox_must_be_on_this_machine(url, allowed):
    assert (sandbox_refusal(url) is None) == allowed


def test_sandbox_refusal_names_the_rule_and_the_host():
    assert sandbox_refusal("https://bank.example.com/login") == (
        "a sandbox must run on this machine (localhost, 127.0.0.1 or ::1), "
        "but the start address's host is bank.example.com"
    )
    assert sandbox_refusal("/login").endswith("but the start address has no host")
