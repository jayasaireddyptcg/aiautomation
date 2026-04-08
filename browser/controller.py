from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from browser.selector_engine import SelectorEngine, SelectorResult
from config.settings import get_settings
from utils.logger import get_logger

log = get_logger("browser")


class BrowserController:
    """
    Central Playwright controller.
    Wraps browser lifecycle, navigation, interactions, and DOM queries.
    All element resolution goes through SelectorEngine (semantic-first).
    """

    def __init__(self, llm=None):
        self._settings = get_settings().browser
        self._llm = llm
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._selector_engine: Optional[SelectorEngine] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch browser and create a new page."""
        log.info(
            f"Starting {self._settings.browser_type} browser "
            f"(headless={self._settings.headless})"
        )
        self._playwright = await async_playwright().start()

        launcher = getattr(self._playwright, self._settings.browser_type)
        self._browser = await launcher.launch(
            headless=self._settings.headless,
            slow_mo=self._settings.slow_mo,
        )
        self._context = await self._browser.new_context(
            viewport={
                "width": self._settings.viewport_width,
                "height": self._settings.viewport_height,
            }
        )
        self._page = await self._context.new_page()
        self._selector_engine = SelectorEngine(self._page, self._llm)
        self._page.set_default_timeout(self._settings.timeout)
        log.info("Browser started successfully.")

    async def stop(self) -> None:
        """Close browser and release resources."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        log.info("Browser stopped.")

    async def __aenter__(self) -> "BrowserController":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    @property
    def selector_engine(self) -> SelectorEngine:
        if self._selector_engine is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._selector_engine

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL and wait for page load."""
        log.info(f"Navigating to: {url}")
        try:
            response = await self._page.goto(url, wait_until="domcontentloaded")
            await self._page.wait_for_load_state("networkidle", timeout=15000)
            status = response.status if response else 0
            title = await self._page.title()
            log.info(f"Navigated to '{title}' (HTTP {status})")
            return {"success": True, "url": self._page.url, "title": title, "status": status}
        except PlaywrightTimeoutError:
            title = await self._page.title()
            return {"success": True, "url": self._page.url, "title": title, "status": 0}
        except Exception as e:
            log.error(f"Navigation failed: {e}")
            return {"success": False, "error": str(e), "url": url}

    async def get_current_url(self) -> str:
        return self._page.url

    async def get_page_title(self) -> str:
        return await self._page.title()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    async def click(
        self,
        description: str,
        element_type: Optional[str] = None,
        css_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Click an element. Resolves via SelectorEngine if no explicit selector."""
        try:
            result = await self._resolve(description, element_type, css_selector)
            if not result.found or result.locator is None:
                return {
                    "success": False,
                    "error": f"Element not found: '{description}'",
                    "strategy": None,
                }
            await result.locator.scroll_into_view_if_needed()
            await result.locator.click()
            log.info(f"Clicked '{description}' via {result.strategy.value}")
            return {
                "success": True,
                "description": description,
                "strategy": result.strategy.value,
                "selector": result.selector,
            }
        except Exception as e:
            log.error(f"Click failed on '{description}': {e}")
            return {"success": False, "error": str(e), "description": description}

    async def type_text(
        self,
        description: str,
        text: str,
        clear_first: bool = True,
        element_type: Optional[str] = None,
        css_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Type text into an input element."""
        try:
            result = await self._resolve(description, element_type, css_selector)
            if not result.found or result.locator is None:
                return {
                    "success": False,
                    "error": f"Input not found: '{description}'",
                }
            await result.locator.scroll_into_view_if_needed()
            if clear_first:
                await result.locator.clear()
            await result.locator.type(text, delay=30)
            log.info(
                f"Typed into '{description}' via {result.strategy.value}: "
                f"'{text[:40]}{'...' if len(text) > 40 else ''}'"
            )
            return {
                "success": True,
                "description": description,
                "strategy": result.strategy.value,
                "selector": result.selector,
                "text_length": len(text),
            }
        except Exception as e:
            log.error(f"Type failed on '{description}': {e}")
            return {"success": False, "error": str(e), "description": description}

    async def fill(
        self,
        description: str,
        text: str,
        element_type: Optional[str] = None,
        css_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fill (instant) an input element."""
        try:
            result = await self._resolve(description, element_type, css_selector)
            if not result.found or result.locator is None:
                return {"success": False, "error": f"Input not found: '{description}'"}
            await result.locator.fill(text)
            log.info(f"Filled '{description}' via {result.strategy.value}")
            return {
                "success": True,
                "description": description,
                "strategy": result.strategy.value,
            }
        except Exception as e:
            log.error(f"Fill failed on '{description}': {e}")
            return {"success": False, "error": str(e)}

    async def press_key(self, key: str) -> Dict[str, Any]:
        """Press a keyboard key (e.g., 'Enter', 'Tab', 'Escape')."""
        try:
            await self._page.keyboard.press(key)
            log.info(f"Pressed key: {key}")
            return {"success": True, "key": key}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def hover(
        self,
        description: str,
        element_type: Optional[str] = None,
        css_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Hover over an element."""
        try:
            result = await self._resolve(description, element_type, css_selector)
            if not result.found or result.locator is None:
                return {"success": False, "error": f"Element not found: '{description}'"}
            await result.locator.hover()
            return {"success": True, "description": description}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def select_option(
        self,
        description: str,
        value: str,
        css_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Select an option from a dropdown."""
        try:
            result = await self._resolve(description, "combobox", css_selector)
            if not result.found or result.locator is None:
                return {"success": False, "error": f"Dropdown not found: '{description}'"}
            await result.locator.select_option(label=value)
            log.info(f"Selected '{value}' in '{description}'")
            return {"success": True, "description": description, "value": value}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def scroll(self, direction: str = "down", amount: int = 300) -> Dict[str, Any]:
        """Scroll the page."""
        try:
            dy = amount if direction == "down" else -amount
            await self._page.evaluate(f"window.scrollBy(0, {dy})")
            return {"success": True, "direction": direction, "amount": amount}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    async def wait_for_element(
        self,
        description: str,
        timeout: int = 10000,
        css_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Wait until an element is visible."""
        try:
            if css_selector:
                await self._page.wait_for_selector(css_selector, timeout=timeout)
                return {"success": True, "found": True, "selector": css_selector}

            deadline = asyncio.get_event_loop().time() + timeout / 1000
            while asyncio.get_event_loop().time() < deadline:
                result = await self._selector_engine.find(description)
                if result.found:
                    return {
                        "success": True,
                        "found": True,
                        "strategy": result.strategy.value,
                    }
                await asyncio.sleep(0.5)
            return {"success": False, "found": False, "error": "Timeout waiting for element"}
        except PlaywrightTimeoutError:
            return {"success": False, "found": False, "error": "Timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_for_url(self, url_pattern: str, timeout: int = 15000) -> Dict[str, Any]:
        """Wait for URL to match pattern."""
        try:
            await self._page.wait_for_url(f"**{url_pattern}**", timeout=timeout)
            return {"success": True, "url": self._page.url}
        except PlaywrightTimeoutError:
            return {"success": False, "error": "URL pattern not matched within timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_seconds(self, seconds: float) -> Dict[str, Any]:
        """Explicit wait."""
        await asyncio.sleep(seconds)
        return {"success": True, "waited_seconds": seconds}

    # ------------------------------------------------------------------
    # DOM Queries
    # ------------------------------------------------------------------

    async def get_page_text(self) -> str:
        """Extract visible text from the page."""
        try:
            return await self._page.evaluate(
                "() => document.body.innerText"
            )
        except Exception:
            return ""

    async def get_page_html(self, max_length: int = 20000) -> str:
        """Get page HTML (truncated)."""
        try:
            html = await self._page.content()
            return html[:max_length]
        except Exception:
            return ""

    async def get_element_text(
        self,
        description: str,
        css_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get inner text of an element."""
        try:
            result = await self._resolve(description, None, css_selector)
            if not result.found or result.locator is None:
                return {"success": False, "error": f"Element not found: '{description}'"}
            text = await result.locator.inner_text()
            return {"success": True, "text": text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def element_exists(self, description: str, css_selector: Optional[str] = None) -> bool:
        """Check if an element exists."""
        result = await self._resolve(description, None, css_selector)
        return result.found

    async def get_all_links(self) -> List[Dict[str, str]]:
        """Get all anchor links on the page."""
        try:
            links = await self._page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => ({
                    text: a.innerText.trim(),
                    href: a.href
                })).filter(l => l.href && l.text);
            }""")
            return links
        except Exception:
            return []

    async def get_form_fields(self) -> List[Dict[str, str]]:
        """Get all form fields on the page."""
        try:
            fields = await self._page.evaluate("""() => {
                const inputs = Array.from(document.querySelectorAll(
                    'input, textarea, select'
                ));
                return inputs.map(el => ({
                    tag: el.tagName.toLowerCase(),
                    type: el.type || '',
                    name: el.name || '',
                    id: el.id || '',
                    placeholder: el.placeholder || '',
                    ariaLabel: el.getAttribute('aria-label') || '',
                    value: el.value || ''
                }));
            }""")
            return fields
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Screenshots
    # ------------------------------------------------------------------

    async def screenshot(self, path: Optional[str] = None) -> str:
        """Take a screenshot and return as base64 or save to path."""
        try:
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                await self._page.screenshot(path=path, full_page=False)
                log.debug(f"Screenshot saved to: {path}")
                return path
            else:
                data = await self._page.screenshot(full_page=False)
                return base64.b64encode(data).decode("utf-8")
        except Exception as e:
            log.warning(f"Screenshot failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve(
        self,
        description: str,
        element_type: Optional[str],
        css_selector: Optional[str],
    ) -> SelectorResult:
        """Resolve element: use explicit CSS if given, else semantic lookup."""
        if css_selector:
            return await self._selector_engine.find_by_css(css_selector)
        return await self._selector_engine.find(description, element_type)

    async def get_dom_state(self) -> Dict[str, Any]:
        """Return a structured snapshot of the current DOM state."""
        try:
            url = self._page.url
            title = await self._page.title()
            text = (await self.get_page_text())[:3000]
            fields = await self.get_form_fields()
            links = (await self.get_all_links())[:20]
            return {
                "url": url,
                "title": title,
                "page_text_snippet": text,
                "form_fields": fields,
                "links": links,
            }
        except Exception as e:
            return {"error": str(e)}
