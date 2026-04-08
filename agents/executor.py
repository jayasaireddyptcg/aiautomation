from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from agents.planner import TaskStep
from browser.controller import BrowserController
from utils.logger import get_logger

log = get_logger("executor")

EXECUTOR_SYSTEM_PROMPT = """You are the Executor Agent of an autonomous web automation system.

Your job: execute a single step using the available browser tools.

RULES:
1. You have ONE step to execute. Call the appropriate tool exactly once (or a short sequence if needed).
2. Always verify the action was successful using the tool's response.
3. Prefer semantic descriptions over hardcoded selectors.
4. If a tool call fails, report the failure clearly.
5. Do NOT attempt more than what the current step requires.
6. After executing, confirm success or explain the failure.

Available tools: navigate, click, type_text, fill, press_key, wait_for_element,
wait_seconds, get_page_text, get_dom_state, element_exists, scroll, take_screenshot,
hover, select_option, get_element_text.

Respond with JSON:
{
  "status": "success | failed | partial",
  "tool_called": "tool name",
  "tool_result": {...},
  "observation": "what happened after executing",
  "error": "error message if failed"
}"""


class StepResult(BaseModel):
    step_id: int
    status: str  # success | failed | partial
    tool_called: str = ""
    tool_result: Dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    error: str = ""
    dom_snapshot: Optional[Dict[str, Any]] = None


class ExecutorAgent:
    """
    Executor Agent: executes individual TaskSteps using browser tools.
    Uses LLM to reason about which tool to call and validates results.
    """

    def __init__(self, llm, controller: BrowserController, tools: List[BaseTool]):
        self.llm = llm
        self.controller = controller
        self.tools = {t.name: t for t in tools}
        self._log = get_logger("executor")

    async def execute_step(
        self,
        step: TaskStep,
        dom_state: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Execute a single TaskStep."""
        self._log.info(
            f"Executing step {step.step_id}: [{step.action}] {step.description}"
        )

        # Try direct execution first (deterministic, no LLM round-trip)
        direct_result = await self._direct_execute(step)
        if direct_result is not None:
            return direct_result

        # Fall back to LLM-guided execution
        return await self._llm_guided_execute(step, dom_state)

    async def _direct_execute(self, step: TaskStep) -> Optional[StepResult]:
        """
        Execute the step directly using its declared action and parameters.
        Returns None if the action type cannot be handled deterministically.
        """
        action = step.action.lower()
        params = step.parameters or {}

        tool = self.tools.get(action)
        if tool is None:
            return None

        try:
            raw = await tool._arun(**params)
            result = self._parse_tool_result(raw)
            success = result.get("success", True)

            self._log.info(
                f"Step {step.step_id} {'succeeded' if success else 'failed'} "
                f"via direct execution"
            )
            return StepResult(
                step_id=step.step_id,
                status="success" if success else "failed",
                tool_called=action,
                tool_result=result,
                observation=self._summarize_result(action, result),
                error=result.get("error", "") if not success else "",
            )
        except Exception as e:
            self._log.warning(f"Direct execution of step {step.step_id} failed: {e}")
            return None

    async def _llm_guided_execute(
        self,
        step: TaskStep,
        dom_state: Optional[Dict[str, Any]],
    ) -> StepResult:
        """Use LLM to decide how to execute an ambiguous or complex step."""
        prompt = self._build_execution_prompt(step, dom_state)

        llm_with_tools = self.llm.bind_tools(list(self.tools.values()))
        messages = [
            SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await llm_with_tools.ainvoke(messages)

            # Execute any tool calls from LLM
            if hasattr(response, "tool_calls") and response.tool_calls:
                return await self._execute_tool_calls(step, response, messages)

            # Parse text response
            result_data = self._parse_json_response(response.content)
            return StepResult(
                step_id=step.step_id,
                status=result_data.get("status", "failed"),
                tool_called=result_data.get("tool_called", ""),
                tool_result=result_data.get("tool_result", {}),
                observation=result_data.get("observation", response.content[:500]),
                error=result_data.get("error", ""),
            )

        except Exception as e:
            self._log.error(f"LLM-guided execution failed for step {step.step_id}: {e}")
            return StepResult(
                step_id=step.step_id,
                status="failed",
                error=str(e),
                observation=f"LLM-guided execution raised exception: {e}",
            )

    async def _execute_tool_calls(
        self,
        step: TaskStep,
        ai_message: AIMessage,
        messages: list,
    ) -> StepResult:
        """Process tool calls made by the LLM."""
        last_result = {}
        last_tool_name = ""

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool = self.tools.get(tool_name)
            if tool is None:
                self._log.warning(f"LLM requested unknown tool: {tool_name}")
                continue

            try:
                self._log.debug(f"Tool call: {tool_name}({tool_args})")
                raw = await tool._arun(**tool_args)
                last_result = self._parse_tool_result(raw)
                last_tool_name = tool_name

                # Add to message chain for multi-turn
                messages.append(ai_message)
                messages.append(
                    ToolMessage(content=raw, tool_call_id=tool_id)
                )
            except Exception as e:
                self._log.error(f"Tool '{tool_name}' raised exception: {e}")
                last_result = {"success": False, "error": str(e)}
                last_tool_name = tool_name

        success = last_result.get("success", False)
        return StepResult(
            step_id=step.step_id,
            status="success" if success else "failed",
            tool_called=last_tool_name,
            tool_result=last_result,
            observation=self._summarize_result(last_tool_name, last_result),
            error=last_result.get("error", "") if not success else "",
        )

    def _build_execution_prompt(
        self, step: TaskStep, dom_state: Optional[Dict[str, Any]]
    ) -> str:
        parts = [
            f"STEP TO EXECUTE: {step.description}",
            f"ACTION TYPE: {step.action}",
            f"PARAMETERS: {json.dumps(step.parameters, indent=2)}",
            f"EXPECTED OUTCOME: {step.expected_outcome}",
        ]
        if dom_state:
            state_str = json.dumps(dom_state, indent=2)[:2000]
            parts.append(f"\nCURRENT PAGE STATE:\n{state_str}")

        parts.append("\nExecute this step now using the appropriate browser tool.")
        return "\n".join(parts)

    def _parse_tool_result(self, raw: str) -> Dict[str, Any]:
        """Parse JSON tool response."""
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return {"raw_output": str(raw)[:500], "success": bool(raw)}

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass
        return {"status": "failed", "observation": content[:500]}

    def _summarize_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Generate a human-readable observation from a tool result."""
        if not result.get("success", True):
            return f"Tool '{tool_name}' failed: {result.get('error', 'Unknown error')}"

        summaries = {
            "navigate": lambda r: f"Navigated to '{r.get('title', r.get('url', '?'))}'",
            "click": lambda r: f"Clicked '{r.get('description', '?')}' via {r.get('strategy', '?')}",
            "type_text": lambda r: f"Typed {r.get('text_length', '?')} chars into '{r.get('description', '?')}'",
            "fill": lambda r: f"Filled '{r.get('description', '?')}'",
            "press_key": lambda r: f"Pressed key '{r.get('key', '?')}'",
            "wait_for_element": lambda r: f"Element {'found' if r.get('found') else 'not found'}",
            "element_exists": lambda r: f"Element {'exists' if r.get('exists') else 'does not exist'}",
            "get_page_text": lambda r: f"Got page text ({len(str(r))} chars)",
            "scroll": lambda r: f"Scrolled {r.get('direction', '?')} {r.get('amount', '?')}px",
            "take_screenshot": lambda r: f"Screenshot {'saved' if r.get('success') else 'failed'}",
        }

        fn = summaries.get(tool_name)
        if fn:
            try:
                return fn(result)
            except Exception:
                pass
        return f"Tool '{tool_name}' completed successfully"
