from __future__ import annotations

import json
from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass

from playwright.async_api import Page, Locator
from utils.logger import get_logger

log = get_logger("selector")


class SelectorStrategy(str, Enum):
    ROLE = "role"
    TEXT = "text"
    LABEL = "label"
    ARIA_LABEL = "aria_label"
    PLACEHOLDER = "placeholder"
    CSS = "css"
    XPATH = "xpath"
    LLM_ASSISTED = "llm_assisted"


@dataclass
class SelectorResult:
    strategy: SelectorStrategy
    selector: str
    locator: Optional[Locator]
    found: bool
    confidence: float = 1.0


class SelectorEngine:
    """
    Semantic-first selector engine.
    Priority: role → text → aria-label → placeholder → css → xpath → LLM fallback.
    No hardcoded selectors. All strategies are semantic or computed at runtime.
    """

    def __init__(self, page: Page, llm=None):
        self.page = page
        self.llm = llm

    async def find(
        self,
        description: str,
        element_type: Optional[str] = None,
        timeout: int = 5000,
    ) -> SelectorResult:
        """
        Find an element using progressive semantic strategies.
        Returns the first successful SelectorResult.
        """
        strategies = [
            self._try_role(description, element_type),
            self._try_text(description),
            self._try_aria_label(description),
            self._try_placeholder(description),
            self._try_label(description),
        ]

        for coro in strategies:
            result = await coro
            if result.found:
                log.debug(
                    f"Found '{description}' via {result.strategy.value}: {result.selector}"
                )
                return result

        # LLM-assisted fallback
        if self.llm:
            result = await self._try_llm_assisted(description, element_type)
            if result.found:
                log.info(
                    f"Found '{description}' via LLM-assisted selector: {result.selector}"
                )
                return result

        log.warning(f"Element not found for description: '{description}'")
        return SelectorResult(
            strategy=SelectorStrategy.CSS,
            selector="",
            locator=None,
            found=False,
            confidence=0.0,
        )

    async def _try_role(
        self, description: str, element_type: Optional[str]
    ) -> SelectorResult:
        """Try role-based selection (ARIA roles)."""
        role_map = {
            "button": "button",
            "link": "link",
            "textbox": "textbox",
            "input": "textbox",
            "checkbox": "checkbox",
            "radio": "radio",
            "combobox": "combobox",
            "listbox": "listbox",
            "option": "option",
            "menuitem": "menuitem",
            "tab": "tab",
            "heading": "heading",
            "img": "img",
            "search": "searchbox",
        }

        roles_to_try = []
        if element_type and element_type.lower() in role_map:
            roles_to_try = [role_map[element_type.lower()]]
        else:
            # Infer role from description keywords
            desc_lower = description.lower()
            if any(w in desc_lower for w in ["button", "click", "submit", "btn"]):
                roles_to_try = ["button"]
            elif any(w in desc_lower for w in ["link", "href", "anchor"]):
                roles_to_try = ["link"]
            elif any(w in desc_lower for w in ["input", "field", "type", "enter", "write"]):
                roles_to_try = ["textbox", "combobox", "searchbox"]
            elif any(w in desc_lower for w in ["check", "checkbox"]):
                roles_to_try = ["checkbox"]
            else:
                roles_to_try = ["button", "link", "textbox"]

        for role in roles_to_try:
            try:
                locator = self.page.get_by_role(role, name=description)
                count = await locator.count()
                if count > 0:
                    return SelectorResult(
                        strategy=SelectorStrategy.ROLE,
                        selector=f"role={role}[name='{description}']",
                        locator=locator.first,
                        found=True,
                    )
                # Try partial match
                locator = self.page.get_by_role(role, name=description, exact=False)
                count = await locator.count()
                if count > 0:
                    return SelectorResult(
                        strategy=SelectorStrategy.ROLE,
                        selector=f"role={role}[name~='{description}']",
                        locator=locator.first,
                        found=True,
                    )
            except Exception:
                pass

        return SelectorResult(
            strategy=SelectorStrategy.ROLE, selector="", locator=None, found=False
        )

    async def _try_text(self, description: str) -> SelectorResult:
        """Try text-based selection."""
        try:
            locator = self.page.get_by_text(description, exact=True)
            if await locator.count() > 0:
                return SelectorResult(
                    strategy=SelectorStrategy.TEXT,
                    selector=f"text='{description}'",
                    locator=locator.first,
                    found=True,
                )
            locator = self.page.get_by_text(description, exact=False)
            if await locator.count() > 0:
                return SelectorResult(
                    strategy=SelectorStrategy.TEXT,
                    selector=f"text~='{description}'",
                    locator=locator.first,
                    found=True,
                )
        except Exception:
            pass
        return SelectorResult(
            strategy=SelectorStrategy.TEXT, selector="", locator=None, found=False
        )

    async def _try_aria_label(self, description: str) -> SelectorResult:
        """Try aria-label selection."""
        try:
            locator = self.page.get_by_label(description, exact=False)
            if await locator.count() > 0:
                return SelectorResult(
                    strategy=SelectorStrategy.ARIA_LABEL,
                    selector=f"aria-label~='{description}'",
                    locator=locator.first,
                    found=True,
                )
        except Exception:
            pass
        return SelectorResult(
            strategy=SelectorStrategy.ARIA_LABEL, selector="", locator=None, found=False
        )

    async def _try_placeholder(self, description: str) -> SelectorResult:
        """Try placeholder-based selection."""
        try:
            locator = self.page.get_by_placeholder(description, exact=False)
            if await locator.count() > 0:
                return SelectorResult(
                    strategy=SelectorStrategy.PLACEHOLDER,
                    selector=f"placeholder~='{description}'",
                    locator=locator.first,
                    found=True,
                )
        except Exception:
            pass
        return SelectorResult(
            strategy=SelectorStrategy.PLACEHOLDER,
            selector="",
            locator=None,
            found=False,
        )

    async def _try_label(self, description: str) -> SelectorResult:
        """Try form label association."""
        try:
            locator = self.page.get_by_label(description)
            if await locator.count() > 0:
                return SelectorResult(
                    strategy=SelectorStrategy.LABEL,
                    selector=f"label='{description}'",
                    locator=locator.first,
                    found=True,
                )
        except Exception:
            pass
        return SelectorResult(
            strategy=SelectorStrategy.LABEL, selector="", locator=None, found=False
        )

    async def _try_llm_assisted(
        self, description: str, element_type: Optional[str]
    ) -> SelectorResult:
        """
        Use the LLM to generate a CSS/XPath selector from the current DOM snapshot.
        """
        try:
            dom_snapshot = await self._get_dom_snapshot()
            prompt = self._build_selector_prompt(description, element_type, dom_snapshot)

            from langchain_core.messages import HumanMessage

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            selector_data = self._parse_selector_response(response.content)

            if selector_data:
                css_sel = selector_data.get("selector", "")
                strategy_str = selector_data.get("strategy", "css")

                if css_sel:
                    locator = self.page.locator(css_sel)
                    if await locator.count() > 0:
                        return SelectorResult(
                            strategy=SelectorStrategy.LLM_ASSISTED,
                            selector=css_sel,
                            locator=locator.first,
                            found=True,
                            confidence=float(selector_data.get("confidence", 0.7)),
                        )
        except Exception as e:
            log.warning(f"LLM-assisted selector failed: {e}")

        return SelectorResult(
            strategy=SelectorStrategy.LLM_ASSISTED,
            selector="",
            locator=None,
            found=False,
            confidence=0.0,
        )

    async def _get_dom_snapshot(self, max_length: int = 8000) -> str:
        """Get a simplified DOM snapshot for LLM analysis."""
        try:
            snapshot = await self.page.evaluate("""() => {
                function simplify(el, depth=0) {
                    if (depth > 6) return '';
                    const tag = el.tagName ? el.tagName.toLowerCase() : '';
                    if (['script', 'style', 'svg', 'path', 'noscript'].includes(tag)) return '';
                    const attrs = [];
                    if (el.id) attrs.push(`id="${el.id}"`);
                    if (el.className && typeof el.className === 'string') {
                        attrs.push(`class="${el.className.slice(0, 60)}"`);
                    }
                    if (el.name) attrs.push(`name="${el.name}"`);
                    if (el.type) attrs.push(`type="${el.type}"`);
                    if (el.placeholder) attrs.push(`placeholder="${el.placeholder}"`);
                    if (el.getAttribute && el.getAttribute('aria-label')) {
                        attrs.push(`aria-label="${el.getAttribute('aria-label')}"`);
                    }
                    if (el.getAttribute && el.getAttribute('role')) {
                        attrs.push(`role="${el.getAttribute('role')}"`);
                    }
                    const text = (el.innerText || '').trim().slice(0, 80);
                    const attrStr = attrs.length ? ' ' + attrs.join(' ') : '';
                    let result = `${'  '.repeat(depth)}<${tag}${attrStr}>`;
                    if (text && el.children.length === 0) result += text;
                    for (const child of el.children) {
                        const childStr = simplify(child, depth + 1);
                        if (childStr) result += '\\n' + childStr;
                    }
                    result += `</${tag}>`;
                    return result;
                }
                return simplify(document.body);
            }""")
            return str(snapshot)[:max_length]
        except Exception:
            return "<dom>Unable to capture snapshot</dom>"

    def _build_selector_prompt(
        self, description: str, element_type: Optional[str], dom: str
    ) -> str:
        return f"""You are a web automation expert. Analyze the DOM and generate a CSS selector.

Element to find: "{description}"
Element type hint: {element_type or 'unknown'}

DOM Snapshot (simplified):
{dom}

Respond ONLY with valid JSON in this exact format:
{{
  "selector": "css_selector_here",
  "strategy": "css",
  "confidence": 0.85,
  "reasoning": "brief explanation"
}}

Rules:
- Prefer data-testid, aria-label, name, id attributes
- Avoid position-based selectors (nth-child) unless necessary
- The selector must be valid CSS
- confidence is between 0.0 and 1.0"""

    def _parse_selector_response(self, content: str) -> Optional[dict]:
        """Parse LLM response to extract JSON selector data."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass
        return None

    async def find_by_css(self, selector: str) -> SelectorResult:
        """Direct CSS selector lookup."""
        try:
            locator = self.page.locator(selector)
            if await locator.count() > 0:
                return SelectorResult(
                    strategy=SelectorStrategy.CSS,
                    selector=selector,
                    locator=locator.first,
                    found=True,
                )
        except Exception:
            pass
        return SelectorResult(
            strategy=SelectorStrategy.CSS, selector=selector, locator=None, found=False
        )

    async def find_by_xpath(self, xpath: str) -> SelectorResult:
        """Direct XPath selector lookup."""
        try:
            locator = self.page.locator(f"xpath={xpath}")
            if await locator.count() > 0:
                return SelectorResult(
                    strategy=SelectorStrategy.XPATH,
                    selector=xpath,
                    locator=locator.first,
                    found=True,
                )
        except Exception:
            pass
        return SelectorResult(
            strategy=SelectorStrategy.XPATH, selector=xpath, locator=None, found=False
        )
