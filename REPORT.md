# Design write-up

**In one line:** five credit-union tasks are learned once by a Claude model driving a real
browser against a mock legacy teller portal, saved as signed artifacts, and replayed with no
model in the loop.

| | |
|---|---|
| Tasks learned and replayable | 5 (bill pay, update phone, update email, checking balance, savings balance) |
| Cost to learn a task | $0.19 to $0.33 per run |
| Replay time | 2 to 4 seconds, no model call |
| Tests | 1,222, none call the model |
| Committed evidence | learning runs, a replay for every declared outcome, two person-confirmed payments, a tampered artifact refused |

---

## 1. Architecture

**The flow**

```
request in words ──▶ Intake ──▶ Router ──▶ Replay   (artifact saved and trusted)
                                     └────▶ Discovery (not yet learned) ──▶ Artifact
```

- **Intake.** One small model call with a strict tool per task. Its only job is to name the
  task and read the parameters from the sentence. A value the request does not state is never
  guessed; the caller is asked for it.
- **Router.** A trusted artifact for the task exists: replay. None: discovery. The model
  never decides which engine runs and never runs replay.
- **Discovery loop (our own, not an agent framework).** Each turn the model receives a
  screenshot with numbered marks plus a text list of the elements, and replies with exactly one
  tool call: click, type, choose, read a value by its label, assert a phrase, mark the goal
  complete, or report stuck. The loop owns every control point: limits, allowlist, risk tier,
  what is recorded, how the run ends. The model only chooses.
- **Locators come from code, never the model.** For the picked element our code:
  1. generates candidates in a legacy-aware order: form field name, visible text, the field
     next to its label, accessible name, position;
  2. proves each on the live page: it must match exactly one element, and that element;
  3. scans each for run data: a member ID inside a locator becomes a placeholder, or the
     candidate is dropped;
  4. keeps the best three as priorities 0, 1, 2.
- **Checkpoints are automatic.** After every action: the page path, the page title, and
  "the next step's element is present", plus any phrase the model asserts.
- **One surface, one gate.** A single module acts on the page for both engines (click, type,
  choose, dialogs, secrets typed at the keystroke). One safety gate is called at the same
  points in both. Replay imports nothing from discovery.

**Key decisions and their trade-offs**

| Decision | Instead of | Why |
|---|---|---|
| Screenshot + numbered element list | Pixel coordinates (computer-use tool) | Coordinates do not replay; an element number maps to a real element from which stable locators are derived |
| Per-task contract declared by an engineer | Learning outcomes and pages from the run | About twenty lines per task; replay never has to interpret page text |
| Own observe-decide-act loop | An agent framework | Every stopping condition and safety check is in our code, not behind a library |
| One process, local JSON files | Queue, database | Enough for one surface; nothing the brief evaluates needs more |

---

## 2. Artifact schema

An artifact is a capability an agent can call: **a contract plus a flow**.

**The contract**

| Field | Holds | Note |
|---|---|---|
| `input_parameters` | typed inputs (string, number, boolean) | what the caller supplies |
| `output_definitions` | typed outputs; money as exact decimal text with a currency | never a float |
| `credentials` | name and kind (config or secret) only | never a value |
| `known_outcomes` | the bank's answers: stable code + the signal that reveals it | a phrase on the page, or a dropdown missing the input's value |
| `allowed_paths` | the pages the task may visit | path patterns, not hosts |
| `known_interruptions` | obstacles replay clears itself: signal + one approved recovery | click a stated element, start over, or wait |
| `confirmation_checks` | labels on the confirmation screen that must equal the request | checked before any irreversible step |

**The flow.** One step per action, in order. Each step carries:

- its action and a safety tier: safe, risky, irreversible;
- up to three priority-ordered locators;
- the value to type or choose, written as a placeholder (`{member_id}`);
- its checkpoints and a retry budget.

**Rules the schema enforces**

- Placeholders are the only link between a step and an input.
- A save-time scan refuses a literal secret, an ambiguous literal, or anything that looks like
  an email, phone, SSN or card number.
- A dropdown's hidden option value is stored only for a fixed choice, never for one that comes
  from an input: a frozen value would pay the wrong payee on a later run.
- Every declared output is produced by exactly one read step; every placeholder names a
  declared input; a credential appears only in a typed value; the start page is on the allowed
  list. (Pydantic v2 cross-field validators.)

**Versioned and signed**

- SemVer, bumped by what changed: contract = major, flow = minor, detail = patch. An identical
  rediscovery writes nothing.
- Ed25519 signature over the canonical content: everything except the signature field itself
  and the two timestamps, so a field added later is signed by default. Replay verifies against
  public keys committed with the code. A machine that only replays cannot produce an artifact
  replay would trust.

---

## 3. Determinism & error handling

**No model in the loop.** For each replay:

1. Load the highest version; **verify the signature first**; refuse anything untrusted, never
   fall back to an older version.
2. Check the inputs against the declared types; never coerce.
3. For each step: clear any declared interruption showing → find the element with the first
   locator that matches exactly one → pass the safety gate → act → hold the checkpoints,
   polling within a timeout.
4. A risky or irreversible action is never retried: it could submit twice.

**Three kinds of trouble, decided by the contract alone.** When a check fails or an element is
missing, replay looks at the page in this order:

| Order | What replay looks for | Result |
|---|---|---|
| 1 | A declared **outcome** showing (no such member, insufficient funds, invalid phone, no savings account) | `BUSINESS_OUTCOME` with code, description, screenshot: an answer, not a failure |
| 2 | A declared **interruption** showing (promotion popup, expired session) | Recovered by its one approved action, step retried. Limits: same interruption at most twice; start over at most once, never after the irreversible step |
| 3 | Anything else | `TECHNICAL_FAIL` with the step, what was expected, what was seen, a screenshot |

- The result has five statuses; a validator refuses a result that mixes them (a success
  carries no error, an outcome carries no failure, and so on).
- A declared outcome appearing during a checkpoint's wait ends the wait early: a bank's answer
  arrives in under a second, not after the full timeout.

**Runtime conditions covered**

- Record not found, validation errors, a restricted account: declared outcomes.
- Popup and session expiry: declared interruptions.
- A native confirm dialog: dismissed, unless the system is performing the irreversible step
  it expects to ask.
- A slow page: waited for up to the page limit, then a timeout with a failure block.
- A payment that may have gone through when the run stopped: reported as "unknown, check
  before trying again", never assumed either way.

**Drift**

- A relabelled or moved element is found by a fallback locator; the step trace records which
  priority worked, so a fallback in use is visible evidence the page changed.
- A rediscovery of a changed flow produces a new version; the old one is kept.
- Repeatability is tested: the same request three times gives the same status, code, summary
  and step path; three payments in a row each move exactly the amount.

---

## 4. Heterogeneity & multi-tenant

**The seam**

- The model never sees a selector or a coordinate: only a picture and a numbered list of
  things it can act on. The DOM is one source for that list.
- Everything that touches the page (open, click, type, choose, read text, dialogs, screenshot)
  lives in one surface module that both engines call. The artifact and the replay engine touch
  nothing else.
- **Legacy web with framesets:** add a frame path to each locator.
- **Desktop app:** a second surface module offering the same operations, the element list
  built from the OS accessibility tree, locator kinds for it (automation id, name plus role,
  position). The loop, recorder, contract, safety gate, handoff and evidence stay as they are.
- Stated honestly: this is a design with one implementation. Today's locator kinds are web
  kinds.

**Reuse across institutions**

- An artifact holds credential **names**, not values; each institution's runtime supplies its
  own.
- A signed **per-tenant start URL override**; the start page and allowed pages are path
  patterns, not hosts.
- The contract's outcomes and interruptions are the bank's own wording, which is what varies
  between vendor versions. Next step: per-tenant overrides of exactly those lists (and of
  individual locators) layered on a base artifact, signed with a per-tenant key.
- **Drift detection** uses two signals that already exist: a fallback locator in use, and a
  checkpoint failing for no declared reason. A stability score would be replaying each
  artifact per tenant and counting fallbacks and failures per version.

---

## 5. Escalation & handoff

**Detection is automatic**

| Engine | Triggers |
|---|---|
| Discovery | the model reports stuck; two replies in a row with no action; step or time limit; outside a sandbox, the next action is irreversible |
| Replay | the payment check fails (mismatch, over the limit, no checks declared); a step is riskier than declared; an element is not found or a check fails with no declared explanation; a person already had control this run and the payment is reached |

Without a person available the run ends with that reason. With one (`--operator`), it hands
over instead.

**The request carries context.** The run photographs the page, writes a log line and sends a
WebSocket announcement with: the task, the goal, the step index and description, the reason
in plain words, the screenshot path, and the choices offered. A late listener is told about
the open handoff.

**The person gets the same live session**

| Stage | Who holds control | What happens |
|---|---|---|
| 1. Pause | automation | The run stops acting, photographs the page, logs the request, announces it on the feed |
| 2. Bar | automation | A control bar appears at the top of the run's own browser window; the page behind it is veiled until someone presses **Take over** |
| 3. Take over | person | The lock passes to the person with a fresh token; dialogs are now theirs to answer |
| 4. Person acts | person | Every click, field change, page visited and tab opened is logged; in discovery each click is also recorded as a step |
| 5. Choice | person | **Hand back**, **I finished it**, or **Stop**. A ten-minute timeout or a closed window counts as Stop |
| 6. Return | automation | Dialogs left open are dismissed, extra tabs closed, the page photographed again, the lock returned |

- One lock per run says who holds control. Each handoff has a token; a report from the bar
  counts only with the current token.
- After Take over, dialogs are left for the person; every click (by the element's wording),
  field change (by label, never the value), page visited and tab opened is logged.
- In discovery each click is held, its locators derived and proven and its tier classified
  exactly as the agent's would be, then let through and **recorded as the next step**. A typed
  value is stored as the matching input's placeholder; a value that matches no input means
  the run finishes but nothing is saved.

**Handing back**

- **Hand back:** the bar is removed, extra tabs closed, dialogs left open dismissed, a second
  snapshot taken, the lock returned. Replay carries on from wherever the page is: a click whose
  checks now hold counts as done by the person, any other step is done once more. Discovery
  tells the model which steps were recorded and resumes.
- **I finished it:** unread outputs are read off the page; the payment is judged from the
  page, not from the person's word.
- **Stop**, a ten-minute timeout, or a closed window: the run ends as escalated.
- Once a person has had the window, replay never makes the payment itself.

---

## 6. Safety

**Allowlist**

- The permitted domain comes from configuration; each task declares its pages; action types
  are a configured list.
- Checked: on the start page before it opens and where it lands; before every action; after
  every action lands; on a link's destination before the click.
- Discovery refuses the first violation and tells the model; the second ends the run. A
  person's clicks pass the same gate.

**Risk tiers**

- A rule table over the page and the element's own wording classifies each step. The wording
  is required, so "Confirm Payment" is irreversible whatever the model called it.
- Replay re-classifies with the element it actually found and never trusts a lower declared
  tier.
- An irreversible step is performed by discovery only in a sandbox on this machine, and by
  replay only after the confirmation screen, read by label, equals the request exactly and the
  amount is within the bank-wide limit. Otherwise a person decides.

**Secrets and regulated data**

- Credentials are placeholders in the prompt, the artifact and every log. The real value
  exists only inside the typing call; the Playwright trace is paused around that keystroke.
- A secret may be typed only into a password box, and a password box takes only a secret.
- Every log line passes one chokepoint: sensitive keys and patterns redacted, known secret
  values scrubbed by exact match. Result summaries pass the same filter.
- The save-time scan refuses an artifact holding a literal secret or a sensitive-looking value.
- Page text is data, never instructions: it reaches the model only inside action results.

**Limits**

- Pattern redaction cannot catch a member's name on a screenshot or in a phrase the model
  asserts; the residual guard is review.
- The classifier's rule table is per application.
- No draft-to-approved gate: a freshly learned artifact is trusted once signed.
- Permission denials and server errors are not simulated by the mock bank; they would surface
  as technical failures with the step and what was seen.

---

## 7. Cuts

**Left out on purpose**

- A bounded model call to recover a failed replay step. Replay stays model-free; an
  unexplained page is reported, not guessed at.
- A draft-to-approved review before an artifact is trusted.
- Framesets and iframes in the mock bank.
- A desktop surface and multi-tenant plumbing: design only.
- A CI pipeline.
- Combined requests ("read the balance, then pay if there is enough"): refused by the intake
  rather than half done.

**What I would build next**

1. An approval state on the artifact; the approve command re-signs, so only a key holder can
   approve.
2. Per-tenant overrides of outcomes, interruptions and locators on a base artifact, with a
   second mock variant to prove it.
3. A stability signal from replaying an artifact N times.
4. A surface interface with a second implementation.
5. A permission-denied page and a server error in the mock bank, as declared outcomes.
6. Splitting a combined request into an ordered plan of known tasks, one authorization
   covering the plan.
