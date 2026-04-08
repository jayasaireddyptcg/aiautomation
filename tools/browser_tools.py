from __future__ import annotations

import json
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from browser.controller import BrowserController
from utils.logger import get_logger

log = get_logger("tools")


# ------------------------------------------------------------------
# Input schemas
# ------------------------------------------------------------------


class NavigateInput(BaseModel):
    url: str = Field(description="The full URL to navigate to (include https://).")


class ClickInput(BaseModel):
    description: str = Field(
        description="Semantic description of the element to click (e.g., 'Sign In button', 'Submit')."
    )
    element_type: Optional[str] = Field(
        default=None,
        description="Optional element type hint: button, link, checkbox, etc.",
    )
    css_selector: Optional[str] = Field(
        default=None,
        description="Optional explicit CSS selector (fallback if semantic fails).",
    )


class TypeInput(BaseModel):
    description: str = Field(
        description="Semantic description of the input field (e.g., 'Email address field', 'Search box')."
    )
    text: str = Field(description="The text to type into the field.")
    clear_first: bool = Field(
        default=True, description="Whether to clear existing content before typing."
    )
    css_selector: Optional[str] = Field(
        default=None, description="Optional explicit CSS selector."
    )


class FillInput(BaseModel):
    description: str = Field(description="Semantic description of the input field.")
    text: str = Field(description="The text to fill into the field.")
    css_selector: Optional[str] = Field(default=None)


class PressKeyInput(BaseModel):
    key: str = Field(
        description="Key to press. Examples: 'Enter', 'Tab', 'Escape', 'Control+a'."
    )


class WaitForElementInput(BaseModel):
    description: str = Field(
        description="Semantic description of the element to wait for."
    )
    timeout: int = Field(default=10000, description="Timeout in milliseconds.")
    css_selector: Optional[str] = Field(default=None)


class WaitSecondsInput(BaseModel):
    seconds: float = Field(description="Number of seconds to wait.")


class GetElementTextInput(BaseModel):
    description: str = Field(description="Semantic description of the element.")
    css_selector: Optional[str] = Field(default=None)


class ScrollInput(BaseModel):
    direction: str = Field(
        default="down", description="Scroll direction: 'up' or 'down'."
    )
    amount: int = Field(default=300, description="Pixels to scroll.")


class ElementExistsInput(BaseModel):
    description: str = Field(description="Semantic description of the element to check.")
    css_selector: Optional[str] = Field(default=None)


class ScreenshotInput(BaseModel):
    path: Optional[str] = Field(
        default=None,
        description="Optional file path to save screenshot. If omitted, returns base64.",
    )


class HoverInput(BaseModel):
    description: str = Field(description="Semantic description of the element to hover.")
    css_selector: Optional[str] = Field(default=None)


class SelectOptionInput(BaseModel):
    description: str = Field(description="Semantic description of the dropdown element.")
    value: str = Field(description="The option label/text to select.")
    css_selector: Optional[str] = Field(default=None)


class NoInput(BaseModel):
    """Empty schema for tools that require no input arguments."""


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------


class NavigateTool(BaseTool):
    name: str = "navigate"
    description: str = (
        "Navigate the browser to a URL. Always include the full URL with https://."
    )
    args_schema: Type[BaseModel] = NavigateInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(self, url: str) -> str:
        result = await self.controller.navigate(url)
        return json.dumps(result)

    def _run(self, url: str) -> str:
        raise NotImplementedError("Use async version.")


class ClickTool(BaseTool):
    name: str = "click"
    description: str = (
        "Click an element on the page. Describe the element semantically "
        "(e.g., 'Sign In button', 'Next link', 'Accept cookies button'). "
        "Avoid hardcoded selectors."
    )
    args_schema: Type[BaseModel] = ClickInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(
        self,
        description: str,
        element_type: Optional[str] = None,
        css_selector: Optional[str] = None,
    ) -> str:
        result = await self.controller.click(description, element_type, css_selector)
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class TypeTool(BaseTool):
    name: str = "type_text"
    description: str = (
        "Type text into an input field, textarea, or search box. "
        "Describe the field semantically (e.g., 'Email address field', 'Password input')."
    )
    args_schema: Type[BaseModel] = TypeInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(
        self,
        description: str,
        text: str,
        clear_first: bool = True,
        css_selector: Optional[str] = None,
    ) -> str:
        result = await self.controller.type_text(
            description, text, clear_first, css_selector=css_selector
        )
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class FillTool(BaseTool):
    name: str = "fill"
    description: str = (
        "Instantly fill an input field with text (faster than type_text, no human-like delay). "
        "Use for large text, email body, or when speed matters."
    )
    args_schema: Type[BaseModel] = FillInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(
        self, description: str, text: str, css_selector: Optional[str] = None
    ) -> str:
        result = await self.controller.fill(description, text, css_selector=css_selector)
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class PressKeyTool(BaseTool):
    name: str = "press_key"
    description: str = (
        "Press a keyboard key. Examples: 'Enter' to submit a form, "
        "'Tab' to move focus, 'Escape' to close a dialog."
    )
    args_schema: Type[BaseModel] = PressKeyInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(self, key: str) -> str:
        result = await self.controller.press_key(key)
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class WaitForElementTool(BaseTool):
    name: str = "wait_for_element"
    description: str = (
        "Wait until a specific element appears on the page. "
        "Use after actions that trigger page transitions or dynamic content loading."
    )
    args_schema: Type[BaseModel] = WaitForElementInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(
        self,
        description: str,
        timeout: int = 10000,
        css_selector: Optional[str] = None,
    ) -> str:
        result = await self.controller.wait_for_element(
            description, timeout, css_selector
        )
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class WaitSecondsTool(BaseTool):
    name: str = "wait_seconds"
    description: str = (
        "Wait a fixed number of seconds. Use sparingly when other wait strategies are insufficient."
    )
    args_schema: Type[BaseModel] = WaitSecondsInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(self, seconds: float) -> str:
        result = await self.controller.wait_seconds(seconds)
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class GetPageTextTool(BaseTool):
    name: str = "get_page_text"
    description: str = (
        "Get the visible text content of the current page. "
        "Use to read page content, verify success messages, or extract information."
    )
    args_schema: Type[BaseModel] = NoInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(self, **kwargs) -> str:
        text = await self.controller.get_page_text()
        return text[:5000]

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class GetElementTextTool(BaseTool):
    name: str = "get_element_text"
    description: str = (
        "Get the text content of a specific element on the page."
    )
    args_schema: Type[BaseModel] = GetElementTextInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(
        self, description: str, css_selector: Optional[str] = None
    ) -> str:
        result = await self.controller.get_element_text(description, css_selector)
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class GetDomStateTool(BaseTool):
    name: str = "get_dom_state"
    description: str = (
        "Get a structured snapshot of the current page state including URL, title, "
        "form fields, links, and page text. Use to understand what is on the screen."
    )
    args_schema: Type[BaseModel] = NoInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(self, **kwargs) -> str:
        state = await self.controller.get_dom_state()
        return json.dumps(state, indent=2)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class ElementExistsTool(BaseTool):
    name: str = "element_exists"
    description: str = (
        "Check whether a specific element exists on the current page. "
        "Returns true or false."
    )
    args_schema: Type[BaseModel] = ElementExistsInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(
        self, description: str, css_selector: Optional[str] = None
    ) -> str:
        exists = await self.controller.element_exists(description, css_selector)
        return json.dumps({"exists": exists, "description": description})

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class ScrollTool(BaseTool):
    name: str = "scroll"
    description: str = "Scroll the page up or down by a specified number of pixels."
    args_schema: Type[BaseModel] = ScrollInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(self, direction: str = "down", amount: int = 300) -> str:
        result = await self.controller.scroll(direction, amount)
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class ScreenshotTool(BaseTool):
    name: str = "take_screenshot"
    description: str = (
        "Take a screenshot of the current page for debugging or verification."
    )
    args_schema: Type[BaseModel] = ScreenshotInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(self, path: Optional[str] = None) -> str:
        result = await self.controller.screenshot(path)
        if path:
            return json.dumps({"success": True, "saved_to": path})
        return json.dumps({"success": bool(result), "base64_length": len(result)})

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class HoverTool(BaseTool):
    name: str = "hover"
    description: str = (
        "Hover over an element. Useful to reveal dropdown menus or tooltips."
    )
    args_schema: Type[BaseModel] = HoverInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(
        self, description: str, css_selector: Optional[str] = None
    ) -> str:
        result = await self.controller.hover(description, css_selector=css_selector)
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


class SelectOptionTool(BaseTool):
    name: str = "select_option"
    description: str = (
        "Select an option from a dropdown/select element by its visible label."
    )
    args_schema: Type[BaseModel] = SelectOptionInput
    controller: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _arun(
        self, description: str, value: str, css_selector: Optional[str] = None
    ) -> str:
        result = await self.controller.select_option(description, value, css_selector)
        return json.dumps(result)

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use async version.")


# ------------------------------------------------------------------
# Toolkit factory
# ------------------------------------------------------------------


class BrowserToolkit:
    """Assembles all browser tools bound to a single BrowserController."""

    def __init__(self, controller: BrowserController):
        self._ctrl = controller

    def get_tools(self) -> list:
        ctrl = self._ctrl
        return [
            NavigateTool(controller=ctrl),
            ClickTool(controller=ctrl),
            TypeTool(controller=ctrl),
            FillTool(controller=ctrl),
            PressKeyTool(controller=ctrl),
            WaitForElementTool(controller=ctrl),
            WaitSecondsTool(controller=ctrl),
            GetPageTextTool(controller=ctrl),
            GetElementTextTool(controller=ctrl),
            GetDomStateTool(controller=ctrl),
            ElementExistsTool(controller=ctrl),
            ScrollTool(controller=ctrl),
            ScreenshotTool(controller=ctrl),
            HoverTool(controller=ctrl),
            SelectOptionTool(controller=ctrl),
        ]
