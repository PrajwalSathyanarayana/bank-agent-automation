// The operator's control bar, shown inside the run's own page while a person is
// needed, and the record of what that person does on the page.
//
// Evaluated by src/handoff/control_bar.py as one function expression:
// (wordingOf, config) => ..., where wordingOf is the safety classifier's own
// reading of an element's wording, so the log and the tier describe an element
// alike. Nothing typed is ever read: a changed field is reported by its label.
//
// The bar sits at the top of the window, which is on screen whatever the
// window's size, and the page moves down by the bar's height so the bar covers
// nothing. Until the person presses Take over, a veil covers the page, so every
// action on it happens with the person in control and is recorded. Text from
// the run (why, the task, the step) is set as text, never as markup.
(wordingOf, config) => {
  const STATE = "__bankAgentHandoffBar";
  if (window[STATE]) {
    window[STATE].remove();
  }

  const MAX_TEXT = 80;
  // Clicks worth recording: elements that do something. Typing and choosing are
  // recorded as field changes instead.
  const ACTIONABLE =
    'a[href], button, input[type="submit"], input[type="button"], input[type="reset"], ' +
    'input[type="image"], [role="button"], [role="link"], [onclick]';
  const NOT_FIELDS = new Set(["submit", "button", "reset", "image", "hidden"]);

  let takenOver = config.takenOver;
  const fieldsReported = new WeakSet();

  function tidy(text) {
    return (text || "").replace(/\s+/g, " ").trim().slice(0, MAX_TEXT);
  }

  function report(payload) {
    const send = window[config.binding];
    if (typeof send !== "function") {
      return;
    }
    try {
      Promise.resolve(send({ ...payload, token: config.token })).catch(() => {});
    } catch (error) {
      // A report that can't be sent must never break the bank's own page.
    }
  }

  // ---------------------------------------------------------------------------
  // The bar
  // ---------------------------------------------------------------------------

  const host = document.createElement("div");
  host.setAttribute("data-bank-agent-handoff", "");
  // The host covers the window but lets the page receive events; only the veil
  // and the bar take them. The shadow root keeps the bank's styles off the bar.
  host.style.cssText = "position:fixed;inset:0;z-index:2147483647;pointer-events:none;";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>
      [hidden] { display: none !important; }
      .veil { position: absolute; inset: 0; background: rgba(0, 0, 0, 0.35); pointer-events: auto; }
      .bar {
        position: absolute; left: 0; right: 0; top: 0; pointer-events: auto;
        background: #12325a; color: #ffffff; font: 13px/1.35 Arial, Helvetica, sans-serif;
        padding: 6px 12px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
      }
      .row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; }
      .headline { flex: 1 1 320px; }
      .title { font-weight: bold; }
      .buttons { display: flex; flex-wrap: wrap; gap: 6px; }
      button {
        font: bold 13px Arial, Helvetica, sans-serif; margin: 0; padding: 5px 12px;
        background: #ffffff; color: #12325a; border: 0; border-radius: 3px; cursor: pointer;
      }
      button:disabled { opacity: 0.5; cursor: default; }
      .meta, .context { display: flex; flex-wrap: wrap; gap: 2px 16px; }
      .meta { margin-top: 3px; font-size: 12px; color: #d6e2f0; }
      .context { margin: 0; padding: 0; list-style: none; }
      .hint { margin-top: 2px; font-size: 12px; color: #ffd27f; }
    </style>
    <div class="veil"></div>
    <div class="bar" role="region" aria-label="Operator controls">
      <div class="row">
        <div class="headline"><span class="title"></span>: <span class="why"></span></div>
        <div class="buttons"></div>
      </div>
      <div class="meta">
        <span class="status"></span>
        <ul class="context"></ul>
        <span class="clock"></span>
      </div>
      <div class="hint"></div>
    </div>`;
  const part = (name) => root.querySelector(`.${name}`);
  const veil = part("veil");
  const bar = part("bar");
  const status = part("status");
  const buttonsBox = part("buttons");
  const hint = part("hint");
  const clock = part("clock");

  part("title").textContent = config.title;
  part("why").textContent = config.why;
  part("context").replaceChildren(
    ...config.context.map((line) => {
      const item = document.createElement("li");
      item.textContent = line;
      return item;
    })
  );

  function makeButton({ choice, label }) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => {
      report({ event: "choice", choice });
      if (choice === config.takeOver.choice) {
        takenOver = true;
        render();
        return;
      }
      // One choice per handoff: the buttons stay disabled until the bar is removed.
      for (const other of buttonsBox.querySelectorAll("button")) {
        other.disabled = true;
      }
      status.textContent = "Handing back to the automation…";
    });
    return button;
  }

  function render() {
    veil.hidden = takenOver;
    status.textContent = takenOver
      ? "You are in control of this page. When you're done, choose above."
      : "The page is paused. Press Take over to use it.";
    hint.textContent = takenOver
      ? "If the bank shows a pop-up box, answer it first: these buttons don't respond while it is open."
      : "";
    hint.hidden = !takenOver;
    const offered = takenOver ? config.buttons : [config.takeOver];
    buttonsBox.replaceChildren(...offered.map(makeButton));
  }

  function tick() {
    const left = Math.max(0, Math.round((config.deadlineMs - Date.now()) / 1000));
    clock.textContent =
      left > 0
        ? `Time left: ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`
        : "Time is up: the task will stop.";
  }

  // The page moves down by the bar's height, and back when the bar goes, so the
  // bar never covers anything the person needs.
  const body = document.body || document.documentElement;
  const inlinePadding = body.style.paddingTop;
  const basePadding = parseFloat(getComputedStyle(body).paddingTop) || 0;
  function makeRoom() {
    body.style.paddingTop = `${basePadding + bar.getBoundingClientRect().height}px`;
  }
  // The bar's height changes when the buttons change or the window narrows.
  const resized = new ResizeObserver(makeRoom);

  // ---------------------------------------------------------------------------
  // What the person does
  // ---------------------------------------------------------------------------

  function fromBar(event) {
    return event.composedPath().includes(host);
  }

  function clickKind(el) {
    if (el.tagName === "A" || el.getAttribute("role") === "link") {
      return "link";
    }
    if (el.tagName === "BUTTON" || el.tagName === "INPUT" || el.getAttribute("role") === "button") {
      return "button";
    }
    return el.tagName.toLowerCase();
  }

  function onClick(event) {
    if (!takenOver || fromBar(event) || !(event.target instanceof Element)) {
      return;
    }
    const el = event.target.closest(ACTIONABLE);
    if (el === null) {
      return;
    }
    report({
      event: "click",
      what: tidy(wordingOf(el)[0]),
      element_kind: clickKind(el),
      page_path: location.pathname,
    });
  }

  // The nearest cell to the left with text, stopping at a cell holding another
  // control: legacy forms put a field's label there. A readable name for the
  // log, a subset of the element list's label rules — never a locator.
  function leftCellText(el) {
    const cell = el.closest("td, th");
    if (cell === null) {
      return "";
    }
    for (let left = cell.previousElementSibling; left; left = left.previousElementSibling) {
      if (left.querySelector("input:not([type=hidden]), select, textarea, button")) {
        return "";
      }
      const text = tidy(left.textContent);
      if (text) {
        return text;
      }
    }
    return "";
  }

  function fieldLabel(el) {
    const labels = el.labels ? Array.from(el.labels).map((label) => label.textContent).join(" ") : "";
    return (
      tidy(el.getAttribute("aria-label")) ||
      tidy(labels) ||
      leftCellText(el) ||
      tidy(el.getAttribute("placeholder")) ||
      tidy(el.getAttribute("name"))
    );
  }

  function fieldKind(el) {
    if (el.tagName === "SELECT") {
      return "dropdown";
    }
    if (el.tagName === "TEXTAREA") {
      return "text area";
    }
    const kinds = { password: "password box", checkbox: "checkbox", radio: "radio button" };
    return kinds[el.type] || "text box";
  }

  function onChange(event) {
    const el = event.target;
    if (!takenOver || fromBar(event) || !(el instanceof Element) || !el.matches("input, select, textarea")) {
      return;
    }
    if ((el.tagName === "INPUT" && NOT_FIELDS.has(el.type)) || fieldsReported.has(el)) {
      return;
    }
    fieldsReported.add(el);
    report({ event: "field_changed", what: fieldLabel(el), element_kind: fieldKind(el), page_path: location.pathname });
  }

  // On the window, in the capture phase: seen before any handler of the page's
  // own can stop the event.
  window.addEventListener("click", onClick, true);
  window.addEventListener("change", onChange, true);
  tick();
  const timer = setInterval(tick, 1000);

  window[STATE] = {
    remove() {
      window.removeEventListener("click", onClick, true);
      window.removeEventListener("change", onChange, true);
      clearInterval(timer);
      resized.disconnect();
      host.remove();
      body.style.paddingTop = inlinePadding;
      delete window[STATE];
    },
  };
  document.documentElement.appendChild(host);
  render();
  resized.observe(bar);
  makeRoom();
}
