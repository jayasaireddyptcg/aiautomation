from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from agents.planner import TaskStep
from agents.executor import StepResult
from browser.controller import BrowserController
from utils.logger import get_logger

log = get_logger("observer")

OBSERVER_SYSTEM_PROMPT = """You are the Observer Agent of an autonomous web automation system.

Your job: analyze the current browser state and step result to determine whether the step succeeded,
detect errors, and provide structured observations for the Recovery and Orchestrator agents.

RULES:
1. Be objective — report what IS on the page, not what should be.
2. Detect common web failure patterns: error messages, CAPTCHA, login walls, rate limits,
   404/500 pages, modal dialogs blocking action, stale elements, redirects.
3. Assess success by comparing expected_outcome to actual page state.
4. Provide actionable observations, not vague summaries.

OUTPUT FORMAT (strict JSON):
{
  "step_succeeded": true | false,
  "confidence": 0.0-1.0,
  "page_state": {
    "url": "current url",
    "title": "page title",
    "has_error": true | false,
    "error_type": "none | captcha | login_wall | rate_limit | not_found | server_error | element_not_found | unexpected_redirect | form_error | dialog_blocking | unknown",
    "error_message": "extracted error text if any",
    "success_indicators": ["list of observed success signals"],
    "failure_indicators": ["list of observed failure signals"]
  },
  "action_effect": "description of what visibly changed on the page",
  "recommendation": "continue | retry | wait | replan | abort",
  "notes": "any additional observations"
}"""


class Observation(BaseModel):
    step_id: int
    step_succeeded: bool
    confidence: float = 1.0
    page_url: str = ""
    page_title: str = ""
    has_error: bool = False
    error_type: str = "none"
    error_message: str = ""
    success_indicators: List[str] = Field(default_factory=list)
    failure_indicators: List[str] = Field(default_factory=list)
    action_effect: str = ""
    recommendation: str = "continue"  # continue | retry | wait | replan | abort
    notes: str = ""
    dom_state: Optional[Dict[str, Any]] = None


class ObserverAgent:
    """
    Observer Agent: monitors DOM state after each step and reports structured observations.
    Detects success, failures, CAPTCHAs, rate limits, unexpected UI states.
    """

    def __init__(self, llm, controller: BrowserController):
        self.llm = llm
        self.controller = controller
        self._log = get_logger("observer")

    async def observe(
        self,
        step: TaskStep,
        step_result: StepResult,
        previous_observations: Optional[List[Observation]] = None,
    ) -> Observation:
        """Observe the browser state after a step execution."""
        self._log.info(f"Observing step {step.step_id} result...")

        # Capture current DOM state
        dom_state = await self.controller.get_dom_state()

        # Fast-path: tool reported clear success/failure without ambiguity
        fast_obs = self._fast_observe(step, step_result, dom_state)
        if fast_obs is not None:
            return fast_obs

        # LLM-based deep observation
        obs = await self._llm_observe(step, step_result, dom_state, previous_observations)
        obs.dom_state = dom_state
        return obs

    def _fast_observe(
        self,
        step: TaskStep,
        step_result: StepResult,
        dom_state: Dict[str, Any],
    ) -> Optional[Observation]:
        """
        Quick heuristic checks — skip LLM if outcome is unambiguous.
        """
        page_text = dom_state.get("page_text_snippet", "").lower()
        url = dom_state.get("url", "")

        # Clear tool-level failure
        if step_result.status == "failed" and step_result.error:
            error_type = self._classify_error(step_result.error, page_text, url)
            return Observation(
                step_id=step.step_id,
                step_succeeded=False,
                confidence=0.9,
                page_url=url,
                page_title=dom_state.get("title", ""),
                has_error=True,
                error_type=error_type,
                error_message=step_result.error,
                failure_indicators=[step_result.error],
                action_effect="Action failed at tool level",
                recommendation=self._recommend_for_error(error_type),
                dom_state=dom_state,
            )

        # Navigation success
        if step.action == "navigate" and step_result.status == "success":
            expected_url_fragment = step.parameters.get("url", "")
            if expected_url_fragment and url:
                return Observation(
                    step_id=step.step_id,
                    step_succeeded=True,
                    confidence=0.95,
                    page_url=url,
                    page_title=dom_state.get("title", ""),
                    action_effect=f"Navigated to {url}",
                    recommendation="continue",
                    dom_state=dom_state,
                )

        return None  # Needs LLM observation

    async def _llm_observe(
        self,
        step: TaskStep,
        step_result: StepResult,
        dom_state: Dict[str, Any],
        previous_observations: Optional[List[Observation]],
    ) -> Observation:
        """Deep LLM-based observation."""
        prompt = self._build_observe_prompt(
            step, step_result, dom_state, previous_observations
        )

        try:
            response = await self.llm.ainvoke(
                [SystemMessage(content=OBSERVER_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            obs_data = self._parse_response(response.content)
            return self._build_observation(step.step_id, obs_data, dom_state)
        except Exception as e:
            self._log.error(f"LLM observation failed: {e}")
            # Fallback: trust the tool result
            succeeded = step_result.status == "success"
            return Observation(
                step_id=step.step_id,
                step_succeeded=succeeded,
                confidence=0.5,
                page_url=dom_state.get("url", ""),
                page_title=dom_state.get("title", ""),
                action_effect=step_result.observation,
                recommendation="continue" if succeeded else "retry",
                notes=f"LLM observation failed: {e}",
            )

    def _build_observe_prompt(
        self,
        step: TaskStep,
        step_result: StepResult,
        dom_state: Dict[str, Any],
        previous_observations: Optional[List[Observation]],
    ) -> str:
        dom_str = json.dumps(dom_state, indent=2)[:3000]
        prev_context = ""
        if previous_observations:
            last = previous_observations[-1]
            prev_context = (
                f"\nPREVIOUS STEP OBSERVATION: {last.action_effect} "
                f"(URL was: {last.page_url})"
            )

        return f"""STEP EXECUTED:
  ID: {step.step_id}
  Action: {step.action}
  Description: {step.description}
  Expected Outcome: {step.expected_outcome}
  Parameters: {json.dumps(step.parameters)}

TOOL EXECUTION RESULT:
  Status: {step_result.status}
  Tool Called: {step_result.tool_called}
  Tool Result: {json.dumps(step_result.tool_result)[:1000]}
  Error: {step_result.error or 'none'}
{prev_context}

CURRENT PAGE STATE:
{dom_str}

Analyze this and produce the observation JSON."""

    def _parse_response(self, content: str) -> Dict[str, Any]:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass
        return {}

    def _build_observation(
        self, step_id: int, data: Dict[str, Any], dom_state: Dict[str, Any]
    ) -> Observation:
        page_state = data.get("page_state", {})
        return Observation(
            step_id=step_id,
            step_succeeded=data.get("step_succeeded", False),
            confidence=float(data.get("confidence", 0.7)),
            page_url=page_state.get("url", dom_state.get("url", "")),
            page_title=page_state.get("title", dom_state.get("title", "")),
            has_error=page_state.get("has_error", False),
            error_type=page_state.get("error_type", "none"),
            error_message=page_state.get("error_message", ""),
            success_indicators=page_state.get("success_indicators", []),
            failure_indicators=page_state.get("failure_indicators", []),
            action_effect=data.get("action_effect", ""),
            recommendation=data.get("recommendation", "continue"),
            notes=data.get("notes", ""),
        )

    def _classify_error(self, error: str, page_text: str, url: str) -> str:
        """Heuristic error type classification."""
        error_lower = error.lower()
        combined = error_lower + " " + page_text

        if "captcha" in combined or "recaptcha" in combined:
            return "captcha"
        if "rate limit" in combined or "too many requests" in combined or "429" in error:
            return "rate_limit"
        if "not found" in combined or "404" in error or "404" in url:
            return "not_found"
        if "500" in error or "server error" in combined or "internal error" in combined:
            return "server_error"
        if "login" in combined or "sign in" in combined or "unauthorized" in combined:
            return "login_wall"
        if "timeout" in error_lower or "timed out" in error_lower:
            return "element_not_found"
        if "element" in error_lower and (
            "not found" in error_lower or "not visible" in error_lower
        ):
            return "element_not_found"
        return "unknown"

    def _recommend_for_error(self, error_type: str) -> str:
        recommendations = {
            "captcha": "abort",
            "rate_limit": "wait",
            "not_found": "replan",
            "server_error": "retry",
            "login_wall": "replan",
            "element_not_found": "retry",
            "unexpected_redirect": "replan",
            "form_error": "retry",
            "dialog_blocking": "retry",
            "unknown": "retry",
            "none": "continue",
        }
        return recommendations.get(error_type, "retry")
