// Fact collector for perception.py. Playwright evaluates this file's text
// inside the bank's page, so the whole file is one function expression.
// It only reads the page; it never changes it.
//
// Returns one object, {facts, elements}: two arrays in the same order, so
// facts[i] describes elements[i]. Every candidate is returned; perception.py
// chooses which ones to list and asks for handles to those only.
//
// Scope: the page's own document. Elements inside iframes or shadow roots
// are not collected.
() => {
  // -------------------------------------------------------------------------
  // Section 1: which elements are candidates
  // -------------------------------------------------------------------------

  // Standard interactive tags. An <a> without an href is not a link.
  const STANDARD_TAGS = "a[href], input, select, textarea, button";

  // Legacy pages also make other elements clickable in their HTML: an onclick
  // attribute, a clickable role, or a tabindex. A handler attached only from
  // JavaScript (addEventListener) leaves no trace in the HTML and is not found.
  const MARKED_CLICKABLE = "[onclick], [role], [tabindex]";

  // Only roles that mean "click me". Widget roles (checkbox, radio, tab,
  // switch, ...) are deliberately not handled: listing them properly would
  // also mean describing their checked or selected state.
  const CLICKABLE_ROLES = new Set(["button", "link", "menuitem"]);

  // A role attribute may hold several space-separated roles. Browsers use the
  // first one they recognise; we take the first one written, which is the
  // same on any page that gives a single role.
  function firstRole(el) {
    const role = (el.getAttribute("role") || "").trim().toLowerCase();
    return role.split(/\s+/)[0];
  }

  // A tabindex of 0 or more puts the element in the keyboard Tab order: a
  // sign it is meant to be operated. -1 is used for focus management (dialog
  // containers, the main content area), not for controls. el.tabIndex is the
  // browser's own reading of the attribute, so invalid values ("abc") come
  // back as -1 exactly as the browser treats them.
  function inTabOrder(el) {
    return el.hasAttribute("tabindex") && el.tabIndex >= 0;
  }

  function isMarkedClickable(el) {
    return el.hasAttribute("onclick") || CLICKABLE_ROLES.has(firstRole(el)) || inTabOrder(el);
  }

  // A page-wide handler on <html> or <body> (e.g. "close menus on any click")
  // is not a target; listing it would put a box around the whole screen.
  function isPageRoot(el) {
    return el === document.documentElement || el === document.body;
  }

  // Three ways a page marks something disabled:
  // - :disabled covers form controls, including those inside a disabled
  //   <fieldset>; it never matches other elements.
  // - a plain disabled attribute on any other element (old Internet Explorer
  //   honoured it everywhere, so legacy portals still write it);
  // - aria-disabled="true" on the element or a container around it, which
  //   disables everything inside. The "i" makes the value case-insensitive.
  function isDisabled(el) {
    return (
      el.matches(":disabled") ||
      el.hasAttribute("disabled") ||
      el.closest('[aria-disabled="true" i]') !== null
    );
  }

  // Visible means: rendered (no display:none on the element or any
  // ancestor), not visibility:hidden or collapse (the computed value is
  // inherited, so a hidden ancestor counts), a non-zero size, and a box that
  // overlaps the page. Elements covered by a popup stay listed: the
  // screenshot shows the popup, and a blocked click comes back as a failure
  // the model can react to.
  function isVisible(el) {
    if (!el.checkVisibility()) {
      return false;
    }
    if (getComputedStyle(el).visibility !== "visible") {
      return false;
    }
    const box = el.getBoundingClientRect();
    if (box.width <= 0 || box.height <= 0) {
      return false;
    }
    return overlapsPage(box);
  }

  // Rejects elements parked off the page, such as a skip link at
  // left:-9999px. The box is relative to the window, so the scroll position
  // is added to get page coordinates: content scrolled above the window is
  // still on the page and still reachable. scrollingElement is the element
  // that scrolls the page in both standards and quirks mode (the bank's
  // pages render in quirks mode).
  function overlapsPage(box) {
    const page = document.scrollingElement || document.documentElement;
    const left = box.left + window.scrollX;
    const top = box.top + window.scrollY;
    return (
      left + box.width > 0 &&
      top + box.height > 0 &&
      left < page.scrollWidth &&
      top < page.scrollHeight
    );
  }

  function isCandidate(el) {
    if (isPageRoot(el)) {
      return false;
    }
    if (!el.matches(STANDARD_TAGS) && !isMarkedClickable(el)) {
      return false;
    }
    // el.type is normalised by the browser, so TYPE="HIDDEN" is caught too.
    if (el.tagName === "INPUT" && el.type === "hidden") {
      return false;
    }
    if (isDisabled(el)) {
      return false;
    }
    return isVisible(el);
  }

  // querySelectorAll returns page order with no repeats. An element inside
  // another candidate (a link inside a clickable row) is a candidate too:
  // each one is its own action.
  const candidates = Array.from(
    document.querySelectorAll(`${STANDARD_TAGS}, ${MARKED_CLICKABLE}`)
  ).filter(isCandidate);

  // -------------------------------------------------------------------------
  // Section 2: the text of an element
  // -------------------------------------------------------------------------

  // Input types whose value is text shown in the box or on the button.
  const TEXT_INPUT_TYPES = new Set(["text", "email", "search", "tel", "url", "number"]);
  const BUTTON_INPUT_TYPES = new Set(["submit", "button", "reset"]);
  const FORM_CONTROL_TAGS = new Set(["INPUT", "SELECT", "TEXTAREA"]);

  // Long texts are cut here: descriptions show at most 80 characters, and a
  // clickable container can hold a whole table's worth of text.
  const TEXT_MAX = 200;

  // The text a person reads in `root`: used for a link's or button's own text
  // and for label text. `named` is the element being described: it never
  // contributes its own value (a text box inside its own <label> is skipped).
  // `readHidden` is set only for a label pointed at directly by
  // aria-labelledby: the standard reads such a label even when it is hidden,
  // a common way to name an icon.
  //
  // Visibility is checked on the element holding each piece of text, so a
  // child made visible inside a visibility:hidden parent is read: it is on
  // screen, and Chromium's own accessibility tree reads it too.
  function textOf(root, named, readHidden = false) {
    function walk(node) {
      if (node.nodeType === Node.TEXT_NODE) {
        return readHidden || isShown(node.parentElement) ? node.data : "";
      }
      if (node.nodeType !== Node.ELEMENT_NODE) {
        return "";
      }
      if (node === named && node !== root) {
        return "";
      }
      if (!readHidden && isHiddenSubtree(node)) {
        return "";
      }
      if (node.tagName === "BR") {
        return " ";
      }
      const text = ownText(node);
      // Block-level elements (cells, divs, paragraphs) separate words.
      return getComputedStyle(node).display === "inline" ? text : ` ${text} `;
    }

    function ownText(node) {
      // A child's aria-label replaces its content: <span aria-label="Close">×</span>
      // reads "Close". The root's own aria-label is a label rule, not text.
      // A control inside the text keeps showing its value instead, as the
      // standard says.
      const ariaLabel = (node.getAttribute("aria-label") || "").trim();
      if (node !== root && ariaLabel && !FORM_CONTROL_TAGS.has(node.tagName)) {
        return ariaLabel;
      }
      const control = node === named ? null : controlText(node);
      const inner =
        control !== null ? control : Array.from(node.childNodes, walk).join("");
      return pseudoText(node, "::before") + inner + pseudoText(node, "::after");
    }

    return tidy(walk(root));
  }

  // One line, bounded: whitespace collapsed, cut at TEXT_MAX.
  function tidy(text) {
    const line = text.replace(/\s+/g, " ").trim();
    return line.length > TEXT_MAX ? line.slice(0, TEXT_MAX) : line;
  }

  // display:none or aria-hidden="true" hides an element and everything
  // inside it; nothing inside can undo either.
  function isHiddenSubtree(el) {
    return (
      getComputedStyle(el).display === "none" ||
      (el.getAttribute("aria-hidden") || "").trim().toLowerCase() === "true"
    );
  }

  // visibility is different: a child can set itself visible inside a hidden
  // parent, so it is checked on the element that holds each piece of text.
  function isShown(el) {
    return getComputedStyle(el).visibility === "visible";
  }

  // What an element inside the text contributes instead of its children:
  // an image its alt text, a control what it currently shows. null means
  // "not a control; read its children".
  function controlText(el) {
    switch (el.tagName) {
      case "IMG":
        return el.getAttribute("alt") || "";
      case "SELECT":
        // option.label is what the closed dropdown shows (its label
        // attribute, else its text), and what Playwright selects by.
        return Array.from(el.selectedOptions, (option) => option.label).join(" ");
      case "TEXTAREA":
        return el.value;
      case "INPUT":
        return inputText(el);
      default:
        return null;
    }
  }

  function inputText(el) {
    // A password's value is never read, wherever the box appears.
    if (el.type === "password") {
      return "";
    }
    if (TEXT_INPUT_TYPES.has(el.type) || BUTTON_INPUT_TYPES.has(el.type)) {
      return el.value;
    }
    if (el.type === "image") {
      return el.getAttribute("alt") || "";
    }
    // Checkboxes, radio buttons, file pickers and the like show no text.
    return "";
  }

  // Text added by CSS (::before / ::after). The computed value is "none",
  // "normal", or a list of parts; only the quoted strings are text.
  function pseudoText(el, which) {
    if (!isShown(el)) {
      return "";
    }
    const content = getComputedStyle(el, which).content;
    const strings = content.match(/"(?:[^"\\]|\\.)*"/g) || [];
    return strings.map((s) => s.slice(1, -1).replace(/\\(.)/g, "$1")).join("");
  }

  // -------------------------------------------------------------------------
  // Section 3: the label of an element, and where it came from
  // -------------------------------------------------------------------------

  // The label rules, in order; the first that finds text wins. Its source is
  // reported so the description can say whether the label is real or guessed.
  // The table rules run for form fields only: next to a link or button in a
  // data grid, the neighbouring cell is usually another column's value, not
  // a label.
  function labelOf(el) {
    const name = accessibleName(el);
    if (name) {
      return { label: name, source: "accessible name" };
    }
    if (isFormField(el)) {
      const left = leftCellLabel(el);
      if (left) {
        return { label: left, source: "left label" };
      }
      const above = cellAboveLabel(el);
      if (above) {
        return { label: above, source: "label above" };
      }
      const placeholder = tidy(el.getAttribute("placeholder") || "");
      if (placeholder) {
        return { label: placeholder, source: "placeholder" };
      }
    }
    const title = tidy(el.getAttribute("title") || "");
    if (title) {
      return { label: title, source: "title" };
    }
    return { label: "", source: "" };
  }

  // A practical subset of the standard's accessible name, in its order:
  // aria-labelledby, then aria-label, then native <label> elements.
  function accessibleName(el) {
    // Referenced elements are read even when hidden, and only one level
    // deep: textOf never follows aria-labelledby, so references cannot loop.
    // Ids that match nothing are skipped.
    const ids = (el.getAttribute("aria-labelledby") || "").split(/\s+/).filter(Boolean);
    const referenced = ids
      .map((id) => document.getElementById(id))
      .filter((ref) => ref !== null)
      .map((ref) => textOf(ref, el, true));
    const byReference = tidy(referenced.join(" "));
    if (byReference) {
      return byReference;
    }
    const ariaLabel = tidy(el.getAttribute("aria-label") || "");
    if (ariaLabel) {
      return ariaLabel;
    }
    // el.labels holds every <label for="..."> pointing at the element and a
    // <label> wrapped around it; elements that cannot have labels have none.
    const labels = el.labels ? Array.from(el.labels) : [];
    return tidy(labels.map((label) => textOf(label, el)).join(" "));
  }

  // Inputs a person types into or ticks, selects and text areas. Button and
  // image inputs are buttons, described by their own text.
  function isFormField(el) {
    if (!FORM_CONTROL_TAGS.has(el.tagName)) {
      return false;
    }
    const isButtonInput =
      el.tagName === "INPUT" && (BUTTON_INPUT_TYPES.has(el.type) || el.type === "image");
    return !isButtonInput;
  }

  // A cell holding a control belongs to another field, not a label. The
  // table searches stop there, so another field's value is never read as
  // this field's label.
  function holdsControl(cell) {
    return cell.querySelector("input:not([type=hidden]), select, textarea, button") !== null;
  }

  // Nearest cell to the left in the same row that has text; empty spacer
  // cells are skipped.
  function leftCellLabel(el) {
    const cell = el.closest("td, th");
    if (cell === null) {
      return "";
    }
    for (let left = cell.previousElementSibling; left; left = left.previousElementSibling) {
      if (left.tagName !== "TD" && left.tagName !== "TH") {
        continue;
      }
      if (holdsControl(left)) {
        return "";
      }
      const text = textOf(left, el);
      if (text) {
        return text;
      }
    }
    return "";
  }

  // Nearest cell above in the same column that has text. The column is found
  // by on-screen position (the cell above spans the middle of this cell), so
  // cells spanning several columns are handled.
  function cellAboveLabel(el) {
    const cell = el.closest("td, th");
    const row = cell === null ? null : cell.parentElement;
    const table = row === null ? null : row.closest("table");
    if (table === null || row.rowIndex < 0) {
      return "";
    }
    const box = cell.getBoundingClientRect();
    const middle = box.left + box.width / 2;
    for (let i = row.rowIndex - 1; i >= 0; i--) {
      const above = Array.from(table.rows[i].cells).find((candidate) => {
        const candidateBox = candidate.getBoundingClientRect();
        return candidateBox.left <= middle && middle < candidateBox.right;
      });
      if (above === undefined) {
        continue;
      }
      if (holdsControl(above)) {
        return "";
      }
      const text = textOf(above, el);
      if (text) {
        return text;
      }
    }
    return "";
  }

  // -------------------------------------------------------------------------
  // Section 4: one facts object per candidate
  // -------------------------------------------------------------------------

  // What the browser shows on submit and reset buttons with no value
  // attribute. A plain button input with no value shows nothing.
  const DEFAULT_BUTTON_TEXT = { submit: "Submit", reset: "Reset" };

  // The keys match ElementFacts in perception.py exactly (a test checks
  // this). Box coordinates are CSS pixels relative to the window, the same
  // space as the screenshot.
  function factsOf(el) {
    const box = el.getBoundingClientRect();
    const { label, source } = labelOf(el);
    const facts = {
      tag: el.tagName.toLowerCase(),
      box: { x: box.x, y: box.y, width: box.width, height: box.height },
      input_type: el.tagName === "INPUT" ? el.type : "",
      role: firstRole(el),
      text: isFormField(el) ? "" : visibleText(el),
      label: label,
      label_source: source,
      value: null,
      filled: false,
      checked: null,
      selected: "",
      options: [],
    };
    if (el.tagName === "SELECT") {
      // Every option is sent; the description shows the first ten and a
      // count, and choosing by label works beyond those ten.
      facts.selected = tidy(Array.from(el.selectedOptions, (option) => option.label).join(", "));
      facts.options = Array.from(el.options, (option) => tidy(option.label));
    } else if (el.tagName === "INPUT" && (el.type === "checkbox" || el.type === "radio")) {
      facts.checked = el.checked;
    } else if (el.tagName === "INPUT" && el.type === "password") {
      // Only whether it is filled: the value is compared here, inside the
      // page, and never leaves it.
      facts.filled = el.value !== "";
    } else if (isFormField(el)) {
      facts.value = tidy(el.value);
    }
    return facts;
  }

  // The visible text of a link, button or clickable area.
  function visibleText(el) {
    if (el.tagName === "INPUT") {
      if (el.type === "image") {
        return tidy(el.getAttribute("alt") || "");
      }
      const shown = el.hasAttribute("value") ? el.value : DEFAULT_BUTTON_TEXT[el.type] || "";
      return tidy(shown);
    }
    return textOf(el, el);
  }

  return { facts: candidates.map(factsOf), elements: candidates };
}
