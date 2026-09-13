// Raw parts of one element for locators.py to build candidate locators from.
// Playwright runs this on the element's live handle; it reads the page, never changes it.
//
// Attribute values come back unescaped: Python writes them into selectors with its own
// escaping. Ids and class names are escaped here with CSS.escape, the browser's own
// escaper for CSS names.
(el) => {
  const tag = el.tagName.toLowerCase();

  function attribute(name) {
    return el.hasAttribute(name) ? el.getAttribute(name) : null;
  }

  // What Playwright's exact text lookup compares against: the text content with every
  // run of whitespace, &nbsp; included, as one space.
  function playwrightText(node) {
    return (node.textContent || "").replace(/​/g, "").replace(/\s+/g, " ").trim();
  }

  // One scope per id or class of a container: its selector ("div.actions") and the
  // raw id or class name, which the data scan checks for member data.
  function scopesOf(node) {
    const name = node.tagName.toLowerCase();
    const scopes = [];
    if (node.id) {
      scopes.push({ selector: `${name}#${CSS.escape(node.id)}`, token: node.id });
    }
    for (const token of node.classList) {
      scopes.push({ selector: `${name}.${CSS.escape(token)}`, token: token });
    }
    return scopes;
  }

  function isUnique(selector) {
    return document.querySelectorAll(selector).length === 1;
  }

  // One step of a position path: the tag and its place among siblings of that tag.
  function stepOf(node) {
    let index = 1;
    for (let sibling = node.previousElementSibling; sibling; sibling = sibling.previousElementSibling) {
      if (sibling.tagName === node.tagName) {
        index += 1;
      }
    }
    return `${node.tagName.toLowerCase()}:nth-of-type(${index})`;
  }

  // Walk up from the element. Every container with an id or class gives scopes (for
  // the scoped candidates); a container whose id or class is unique on the page also
  // anchors a position path. Both lists are nearest container first.
  const scopes = [];
  const positions = [];
  const stepsUp = [stepOf(el)];
  for (let node = el.parentElement; node && node !== document.documentElement; node = node.parentElement) {
    const nodeScopes = scopesOf(node);
    scopes.push(...nodeScopes);
    const anchor = nodeScopes.find((scope) => isUnique(scope.selector));
    if (anchor !== undefined) {
      const path = [anchor.selector, ...stepsUp.slice().reverse()].join(" > ");
      positions.push({ path: path, token: anchor.token });
    }
    stepsUp.push(stepOf(node));
  }
  // The full path from the top of the page comes last, for pages with no usable container.
  positions.push({ path: ["html", ...stepsUp.slice().reverse()].join(" > "), token: null });

  return {
    tag: tag,
    type: tag === "input" ? el.type : "",
    name: attribute("name"),
    value: attribute("value"),
    href: attribute("href"),
    text: playwrightText(el),
    scopes: scopes,
    positions: positions,
  };
}
