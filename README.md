# bank-agent-automation

**Teach an AI agent a task on legacy banking software once. Replay it forever with no AI in the loop.**

Banks and credit unions run a long tail of back-office applications that have no API. The only
way in is the screen. This project is the layer that gives an AI agent hands on those screens,
and then takes the AI back out of the loop:

1. **Discovery.** A Claude model drives a real browser to complete a goal, seeing only a
   screenshot and a numbered list of the elements it can act on. Every action is checked,
   recorded and proven before it runs.
2. **The artifact.** The run becomes a typed, versioned, Ed25519-signed JSON capability: the
   steps, three proven locators per element, a checkpoint after every step, the inputs it
   takes, the values it returns, and the answers the bank may give.
3. **Replay.** The artifact runs deterministically, no model involved, in two to four seconds.
   It reports success with outputs, a known business answer (no such member, insufficient
   funds), or a failure that says which step, what was expected and what was seen.
4. **A person when needed.** When the run cannot safely continue, it hands its own live
   browser window to a person, records what they do, and takes the window back.

Built against a mock credit union teller portal in the style of a 2006-era legacy app:
server-rendered HTML, layout tables, no test ids, native confirm dialogs, random
interstitials, session expiry. Five tasks are learned and replayable today.

![Architecture](1A_Main_Architecture_Diagram.png)

## Highlights

- **One entry point.** `run "For member 10234, pay 50 to Sunbelt Electric Co"` reads the
  request with one small model call, replays the task if it has been learned, learns it
  otherwise, and refuses anything it does not know.
- **Deterministic replay.** No model. Locators are tried in priority order, each step's
  checkpoint must hold, declared outcomes and interruptions are recognised, nothing is guessed.
- **A real error taxonomy.** Business outcomes, recoverable interruptions and hard failures
  are three different things in the result contract, enforced by the schema.
- **Payments are checked, not trusted.** Before an irreversible click, replay reads the payee
  and amount off the confirmation screen by their labels and compares them to the request,
  within a bank-wide limit. A mismatch or an amount over the limit goes to a person.
- **Secrets never touch the model, the artifact or the log.** Credentials are placeholders
  everywhere; the real value exists only at the keystroke, the Playwright trace pauses around
  it, one logging chokepoint scrubs it, and a save-time scan refuses any literal that slipped
  through.
- **A live handoff, not a TODO.** One lock per run with a token per handoff, a control bar
  inside the run's own window, a WebSocket feed for anyone watching, and the person's actions
  recorded. In discovery, a person's clicks become steps of the artifact.
- **Evidence for every run.** A structured JSON log, a `result.json`, screenshots, a
  Playwright trace and a plain-English HTML report, plus an index page over all runs.
- **1,222 tests** including real-browser tests against an in-process mock bank, none of
  which call the model. Learning a task costs about 20 to 35 cents.

## How it works

```
                 request in words
                        │
                 ┌──────▼──────┐        one model call: which task, which inputs
                 │   Intake    │        (never guesses a value the request doesn't state)
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │   Router    │  saved, trusted artifact for this task?
                 └──┬───────┬──┘
              no    │       │   yes
        ┌───────────▼─┐   ┌─▼────────────┐
        │  Discovery  │   │    Replay    │
        │  observe →  │   │  load, verify│
        │  decide →   │   │  signature → │
        │  check →    │   │  find element│
        │  record →   │   │  → gate →    │
        │  act        │   │  act → check │
        └──────┬──────┘   └──────┬───────┘
               │  scan, sign,    │
               │  version, save  │
        ┌──────▼──────┐          │
        │  Artifact   │──────────┘
        └─────────────┘
   Both engines share: the safety gate (allowlist, risk tier, payment check),
   the browser surface, the handoff, and the evidence logger.
```

**Discovery** (`src/discovery/`). Each turn the model receives a screenshot with numbered
marks and a text list of the elements, and replies with exactly one tool call: click, type,
choose, read a value by its label, assert a phrase, mark the goal complete, or report stuck.
The loop owns every control point: step and time limits, the allowlist, the risk tier, what
gets recorded, and how the run ends. Locators are generated and proven on the live page by
code in a legacy-aware order (form field name, visible text, the field next to its label,
accessible name, position), never by the model. At the end the recording is validated,
scanned for literal secrets or sensitive-looking values, signed, versioned by what changed
(contract, flow or detail), and written to `artifacts/`.

**Replay** (`src/replay/`). Loads the task's latest version, verifies the signature against the
committed public keys, and refuses anything untrusted (never falling back to an older
version). For each step: clear any declared interruption that is showing, find the element
with the first locator that matches exactly one, pass the same safety gate discovery used,
act, then hold the step's checkpoints. When a check does not hold, a declared outcome on the
page becomes the result, a declared interruption is recovered from and the step retried,
and anything else is a failure with a screenshot.

**Handoff** (`src/handoff/`). Triggers are automatic: a stuck agent, a payment that cannot be
authorised, a step replay cannot recover. The run pauses, photographs the page, announces the
request with its context, and shows a bar at the top of its own window. After **Take over**
the lock belongs to the person; dialogs are theirs to answer; every click, field change (by
label, never the value) and page visited is logged. **Hand back** resumes the run on the same
page; **I finished it** reads the outputs off the page; **Stop** ends the run. A payment is
never made by the system once a person has had the window.

**The task contract** (`src/catalog.py`). Adding a task is a declaration, not engine code: the
goal template, the start page, typed inputs and outputs, the bank's known answers with the
text that signals each, the pages the task may visit, the interruptions it may recover from,
and the labels a payment must match on the confirmation screen.

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.12 | One language for the engines, the mock bank and the tests |
| Browser automation | Playwright (async API) | Real Chromium, auto-waiting, dialog control, trace recording |
| LLM | Anthropic SDK, Claude via `ANTHROPIC_MODEL` | Discovery and the intake; strict tool calls, prompt caching, one action per turn |
| Schemas and validation | Pydantic v2 | The artifact, step and result contracts, validated on every load and save |
| Mock banking portal | Flask + Jinja2 | Server-rendered legacy HTML: layout tables, no test ids, native dialogs |
| Artifact signing | `cryptography` (Ed25519) | Private key signs, committed public keys verify; any edit is detected |
| Handoff feed | `websockets` | Announces each handoff to anyone watching outside the browser window |
| Logging | `python-json-logger` | One JSON line per event, redacted at a single chokepoint |
| Perception marks | Pillow | Numbered boxes drawn on the model's copy of each screenshot |
| Tests | pytest + anyio | 1,222 tests; browser tests run against an in-process mock bank |
| Persistence | Local JSON files | Artifacts and evidence on disk, readable and diffable, no database |

## Quick start

Requirements: Python 3.11+, an Anthropic API key (only for learning a task).

```powershell
python -m venv venv
venv\Scripts\activate                  # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
copy .env.example .env                 # macOS/Linux: cp .env.example .env
```

Fill in `.env`: your `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`, a random
`MOCK_BANK_SECRET_KEY`, the mock bank's teller login (`MOCK_BANK_USERNAME=admin`,
`MOCK_BANK_PASSWORD=admin123`, test values only), and `TARGET_ENVIRONMENT=sandbox` so that
learning runs may confirm a payment on the local mock bank. All other settings have defaults.

Start the mock bank in its own terminal and leave it running:

```
python mock_bank/app.py
```

Give it a goal in plain words. The system decides the rest: it works out which task the
request is and what values it states, replays the task if it has already been learned, and
learns it with the model if not.

```
python -m src.main run "For member 40412, look up the checking balance"
```

The five tasks in `src/catalog.py` have all been learned, and their signed artifacts are
committed, so every request for them replays with no model call beyond the one that reads
the sentence. To make the system learn a task fresh, delete its folder under `artifacts/`
and make a signing key pair once with `python -m src.keys generate`; the next request for
that task runs discovery and saves a new artifact.

Run the tests (no API key, no running bank needed):

```
pytest tests/ -m "not llm"
```

## Usage

```
python -m src.main run "<goal in words>" [--headed] [--operator] [--slow-mo MS]
```

`--headed` shows the browser, `--slow-mo MS` paces each action for a person watching, and
`--operator` makes a person available: when the run needs one, it shows the window and hands
it over instead of stopping. A first learning run of a task can carry `--operator` so that
a stuck agent asks for help.

The command prints the structured result as JSON, then the paths of the run log, the report
and the evidence index. Exit codes: `0` done or the bank's own answer, `1` needs attention,
`2` could not start, `3` the request was not understood, so nothing ran.

Other commands:

```
python -m src.handoff.watch        print each handoff announcement (run before --operator)
python -m src.evidence.index       rebuild evidence/index.html and every run's report
python -m src.keys generate        make the key pair that signs artifacts
```

### Things to try

| What to see | Request |
|---|---|
| A value read and returned | `run "For member 40412, look up the checking balance"` |
| A business answer, not a failure | `run "For member 99999, look up the checking balance"` (`MEMBER_NOT_FOUND`); `run "Read the savings balance for member 20567"` (`NO_SAVINGS_ACCOUNT`) |
| A payment checked on screen, then made | `run "For member 40412, pay 25.50 to Desert Valley Water Utility"` |
| A payment over the limit, stopped | `run "For member 10234, pay 1050 to Sunbelt Electric Co"` (`HUMAN_ESCALATED`, nothing paid) |
| The same payment, confirmed by a person | add `--operator`; press Take over, confirm, press I finished it |
| A bank rule enforced | `run "For member 10234, change the phone number to 520-555-0199"` (`INVALID_PHONE`) |
| A request missing a value | `run "Pay Sunbelt Electric Co for member 10234"` (asks for the amount, runs nothing) |
| A task it does not know | `run "Close the account of member 10234"` (lists the tasks it knows, runs nothing) |
| A slow bank | restart the bank with `MOCK_BANK_SLOW_PAGES_MS=2500`; the run ends `TECHNICAL_FAIL` / `PAGE_TIMEOUT` |
| A changed page | restart with `MOCK_BANK_RENAMED_MENU=true`; the step trace shows a fallback locator in use |
| A tampered artifact | edit one character in the latest artifact file; the run is refused before a browser opens |

Prefix each request with `python -m src.main`.

### Developer commands

For working on the engines themselves, each one can be run directly for a named task, which
skips the intake and the router's decision:

```
python -m src.main replay   --capability NAME --input KEY=VALUE ...   force replay, no model
python -m src.main discover --capability NAME --input KEY=VALUE ...   force discovery, even for a task already learned
```

`discover` also takes `--max-steps N`. These are what the test suite and the development
runs use; a caller of the system uses `run`.

### The tasks

| Capability | Inputs | Returns | Known answers |
|---|---|---|---|
| `member_servicing_and_bill_pay` | `member_id`, `amount`, `payee_name` | balance before and after | `MEMBER_NOT_FOUND`, `INSUFFICIENT_FUNDS`, `ACCOUNT_RESTRICTED`, `PAYEE_NOT_FOUND` |
| `update_member_phone` | `member_id`, `new_phone` | confirmation | `MEMBER_NOT_FOUND`, `INVALID_PHONE` |
| `update_member_email` | `member_id`, `new_email` | confirmation | `MEMBER_NOT_FOUND`, `INVALID_EMAIL` |
| `look_up_checking_balance` | `member_id` | `checking_balance` | `MEMBER_NOT_FOUND` |
| `read_savings_balance` | `member_id` | `savings_balance` | `MEMBER_NOT_FOUND`, `NO_SAVINGS_ACCOUNT` |

Test members: `10234` and `40412` (checking and savings), `20567` (checking only, 512.75),
`30891` (restricted, bill pay refused). Payees: Sunbelt Electric Co, Desert Valley Water
Utility, Horizon Credit Card Services, Canyon Ridge Mortgage Co. All data is fictitious, and
the bank resets it on every start.

## The artifact

```jsonc
{
  "metadata":            { "capability", "description", "version", "integrity_hash", "target_url", ... },
  "input_parameters":    [ { "key": "member_id", "type": "string", "required": true, ... } ],
  "output_definitions":  [ { "key": "checking_balance", "type": "money", "currency": "USD", ... } ],
  "credentials":         [ { "key": "bank_password", "kind": "secret" } ],      // names only
  "known_outcomes":      [ { "code": "MEMBER_NOT_FOUND", "signal": "page_text", "text": "..." } ],
  "known_interruptions": [ { "code": "PROMO_POPUP", "signal": "element_visible", "recovery": "click", ... } ],
  "confirmation_checks": [ { "label": "Amount:", "input_key": "amount", "compare_as": "money" } ],
  "allowed_paths":       [ "/", "/login", "/dashboard", "/search", "/member/*", ... ],
  "steps": [
    { "sequence_index": 5, "action": "type", "input_value": "{member_id}",
      "locators": [ { "type": "css", "value": "input[name='member_id']", "priority": 0 }, ... ],
      "safety_tier": "SAFE",
      "checkpoints": [ { "type": "page_path", "expected_value": "/member/{member_id}" }, ... ] }
  ]
}
```

Inputs appear only as placeholders, credentials only by name. The signature covers everything
except the timestamps, so any edit is detected. Versions bump by what changed: a contract
change is major, a change of flow is minor, a detail is patch; an identical rediscovery writes
nothing.

## Evidence

`evidence/index.html` lists every committed run, learning runs first, each linking to its
own report. Every run has its own folder:

```
evidence/runs/2026-09-16_read_savings_balance_discovery_87432f5c/
├── log.json          one JSON line per event: model decisions and reasons, locators tried,
│                     recoveries, handoffs, tokens used
├── result.json       the structured result the caller receives
├── report.html       the same run for a non-technical reader
├── trace.zip         Playwright trace (playwright show-trace trace.zip), paused around the
│                     password keystroke
└── screenshots/      every page the model saw; the page a run ended on; every failure,
                      recovery and handoff
```

## Project structure

```
bank-agent-automation/
├── mock_bank/                      Mock credit union teller portal (Flask + Jinja2)
│   ├── app.py                      App factory; test switches (slow pages, renamed menu)
│   ├── blueprints/                 auth, member, billpay, activity
│   ├── data/members.json           Fictitious members and payees
│   ├── static/                     legacy.css, legacy.js (confirm dialog, session clock)
│   └── templates/                  Server-rendered pages, layout tables, no test ids
├── src/
│   ├── main.py                     CLI: run, discover, replay
│   ├── catalog.py                  The five task contracts
│   ├── intake.py                   A request in words → a task and its inputs
│   ├── router.py                   Learned → replay; otherwise → discovery
│   ├── keys.py                     Generate the signing key pair; re-sign old artifacts
│   ├── discovery/                  The LLM observe–decide–act loop
│   │   ├── agent.py                The loop and every control point
│   │   ├── perception.py           Screenshot + numbered element list (collect_elements.js)
│   │   ├── locators.py             Locators generated and proven on the live page
│   │   ├── recorder.py             An action becomes a step with automatic checkpoints
│   │   ├── backstop.py             Save-time scan for secrets and sensitive literals
│   │   ├── artifact_builder.py     Validate → scan → sign → version → write
│   │   ├── person_steps.py         A person's handoff actions recorded as steps
│   │   └── prompts.py              System prompt and tool definitions
│   ├── replay/                     Deterministic execution, no model
│   │   ├── executor.py             Load, verify, run each step, one structured result
│   │   ├── locator_resolver.py     Priority fallbacks with a retry budget
│   │   ├── checks.py               Checkpoints after every step
│   │   └── recovery_engine.py      Declared outcomes and interruptions
│   ├── safety/                     Allowlist, risk tiers, payment authorization, redaction,
│   │                               Ed25519 signing, key handling, secret typing rule
│   ├── handoff/                    Session lock, in-window control bar, WebSocket feed, watcher
│   ├── surface/browser.py          The one module that acts on the page for both modes
│   ├── locating/                   Shared locator resolution and page reading rules
│   ├── storage/artifacts.py        Saved versions; the latest trusted one
│   ├── observability/              Run logger, per-run evidence folders, plain-English summaries
│   ├── evidence/                   report.html per run and the index page
│   ├── types/                      Pydantic schemas: artifact, step, result; versioning rules
│   └── config/                     Environment and settings
├── tests/                          1,222 tests; browser tests use an in-process mock bank
├── artifacts/<capability>/         Learned, signed artifacts, one folder per task
├── evidence/                       Committed runs and the index page
├── keys/trusted/                   Public keys replay trusts
├── requirements.txt
└── .env.example
```

## Design notes and limits

- The model never sees selectors or coordinates, only a picture and a numbered list. The
  browser is one source for that list; `src/surface/` is the only place that acts, so another
  surface is another module there. Artifacts hold credential names, not values, so one
  artifact can serve several institutions with their own credentials and start page.
- Replay has no AI fallback on purpose: a page it cannot explain from the two declared lists
  is reported with a screenshot, never guessed at. Adding a new bank answer is a contract
  change, versioned and signed.
- A member's name on a screenshot cannot be redacted by a pattern. A value a person types
  during a handoff that is not one of the task's inputs cannot be stored, so that learning run
  is not saved.
- Not built: framesets and iframes in the mock bank, a desktop surface, multi-tenant
  plumbing, permission-denied and server-error pages in the mock bank, a CI pipeline.

See `REPORT.md` for the design write-up.
