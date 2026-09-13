"""Reading the page the same way in discovery and replay."""
from playwright.async_api import ElementHandle

_WORDING = """(element) => {
  const words = [element.textContent || ""];
  // A button input's value is its label. Other inputs' values are typed data, and a
  // password's value is never read.
  if (element.tagName === "INPUT" && ["submit", "button", "reset"].includes(element.type)) {
    words.push(element.value || "");
  }
  for (const name of ["aria-label", "title", "alt"]) {
    words.push(element.getAttribute(name) || "");
  }
  return words.map((word) => word.replace(/\\s+/g, " ").trim()).filter((word) => word);
}"""


async def element_wording(element: ElementHandle) -> list[str]:
    """What the element itself says, for the safety classifier.

    Its text (hidden parts included, which can only make the tier stricter), a button's
    value, and its aria-label, title and alt. Read from the live element, so the tier
    doesn't depend on which locators survived or how the model described the step.
    """
    return await element.evaluate(_WORDING)
