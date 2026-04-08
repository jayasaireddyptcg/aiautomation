from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from agents.planner import TaskStep
from agents.executor import StepResult
from agents.observer import Observation
from utils.logger import get_logger

log = get_logger("recovery")

RECOVERY_SYSTEM_PROMPT = """You are the Recovery Agent of an autonomous web automation system.

Your job: when a step fails, reason about WHY it failed and decide the optimal recovery action.

RECOVERY STRATEGIES:
- "retry_same": Retry the exact same step (transient errors, network glitches).
- "retry_modified": Retry with different parameters (different selector, wait longer, scroll first).
- "wait_and_retry": Wait N seconds then retry (rate limits, animations, lazy loading).
- "alternative_approach": Try a completely different way to achieve this step (e.g., keyboard shortcut instead of click).
- "skip_step": Skip this non-critical step and continue to the next.
- "replan": The situation has changed enough that a full replan is needed.
- "abort": The goal cannot be achieved (CAPTCHA, no account access, site down).

OUTPUT FORMAT (strict JSON):
{
  "strategy": "retry_same | retry_modified | wait_and_retry | alternative_approach | skip_step | replan | abort",
  "reasoning": "explanation of why this strategy was chosen",
  "wait_seconds": 0,
  "modified_step": {
    // Only if strategy is "retry_modified" or "alternative_approach"
    // A modified version of the step with updated parameters
    "action": "same or different action",
    "description": "modified description",
    "parameters": { ... },
    "expected_outcome": "..."
  },
  "replan_hint": "guidance for the Planner if strategy is replan",
  "learned_insight": "what was learned that should be stored in memory"
}"""


class RecoveryDecision(BaseModel):
    strategy: str  # retry_same | retry_modified | wait_and_retry | alternative_approach | skip_step | replan | abort
    reasoning: str = ""
    wait_seconds: float = 0.0
    modified_step: Optional[Dict[str, Any]] = None
    replan_hint: str = ""
    learned_insight: str = ""


class RecoveryAgent:
    """
    Recovery Agent: analyzes failures and determines the optimal recovery strategy.
    Implements the self-healing intelligence layer.
    """

    def __init__(self, llm):
        self.llm = llm
        self._log = get_logger("recovery")
        self._max_retries_per_step: int = 3

    async def decide(
        self,
        step: TaskStep,
        step_result: StepResult,
        observation: Observation,
        retry_count: int = 0,
        memory_hints: Optional[List[str]] = None,
        all_steps: Optional[List[TaskStep]] = None,
    ) -> RecoveryDecision:
        """
        Determine the recovery strategy for a failed step.
        """
        self._log.info(
            f"Recovery decision for step {step.step_id} "
            f"(retry #{retry_count}, error_type={observation.error_type})"
        )

        # Fast-path heuristics before calling LLM
        fast_decision = self._fast_decide(step, observation, retry_count)
        if fast_decision is not None:
            self._log.info(
                f"Fast recovery: {fast_decision.strategy} — {fast_decision.reasoning}"
            )
            return fast_decision

        # LLM-guided recovery
        return await self._llm_decide(
            step, step_result, observation, retry_count, memory_hints, all_steps
        )

    def _fast_decide(
        self,
        step: TaskStep,
        observation: Observation,
        retry_count: int,
    ) -> Optional[RecoveryDecision]:
        """Heuristic fast-path recovery decisions."""

        # Too many retries → escalate
        if retry_count >= self._max_retries_per_step:
            if not step.critical:
                return RecoveryDecision(
                    strategy="skip_step",
                    reasoning=f"Max retries ({retry_count}) reached for non-critical step.",
                )
            return RecoveryDecision(
                strategy="replan",
                reasoning=f"Max retries ({retry_count}) reached. Requesting full replan.",
                replan_hint=f"Step {step.step_id} failed {retry_count} times: {observation.error_message}",
            )

        # CAPTCHA detected → abort (human intervention needed)
        if observation.error_type == "captcha":
            return RecoveryDecision(
                strategy="abort",
                reasoning="CAPTCHA detected. Human intervention required.",
                learned_insight="CAPTCHA encountered on this site/flow.",
            )

        # Rate limit → wait and retry
        if observation.error_type == "rate_limit":
            wait_time = min(30.0 * (retry_count + 1), 120.0)
            return RecoveryDecision(
                strategy="wait_and_retry",
                reasoning=f"Rate limit detected. Waiting {wait_time}s before retry.",
                wait_seconds=wait_time,
            )

        # Element not found on first try → try modified approach
        if observation.error_type == "element_not_found" and retry_count == 0:
            return RecoveryDecision(
                strategy="retry_modified",
                reasoning="Element not found. Will try with a short wait and alternative selector.",
                wait_seconds=2.0,
                modified_step={
                    "action": step.action,
                    "description": step.description,
                    "parameters": {
                        **step.parameters,
                        "css_selector": None,  # clear any hardcoded selector
                    },
                    "expected_outcome": step.expected_outcome,
                },
                learned_insight=f"Element '{step.description}' required retry with semantic selector.",
            )

        # Server error → simple retry with backoff
        if observation.error_type == "server_error":
            wait_time = 5.0 * (retry_count + 1)
            return RecoveryDecision(
                strategy="wait_and_retry",
                reasoning=f"Server error. Waiting {wait_time}s then retrying.",
                wait_seconds=wait_time,
            )

        # Not found / login wall → replan
        if observation.error_type in ("not_found", "login_wall", "unexpected_redirect"):
            return RecoveryDecision(
                strategy="replan",
                reasoning=f"Navigation issue: {observation.error_type}. Need to replan.",
                replan_hint=f"Page state changed: {observation.error_type}. Current URL: {observation.page_url}",
            )

        # Generic retry on first attempt
        if retry_count == 0 and not step.critical is False:
            return RecoveryDecision(
                strategy="retry_same",
                reasoning="First failure. Simple retry to handle transient issues.",
                wait_seconds=1.0,
            )

        return None  # Delegate to LLM

    async def _llm_decide(
        self,
        step: TaskStep,
        step_result: StepResult,
        observation: Observation,
        retry_count: int,
        memory_hints: Optional[List[str]],
        all_steps: Optional[List[TaskStep]],
    ) -> RecoveryDecision:
        """LLM-guided recovery for complex failure scenarios."""
        prompt = self._build_recovery_prompt(
            step, step_result, observation, retry_count, memory_hints, all_steps
        )

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=RECOVERY_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            data = self._parse_response(response.content)
            decision = RecoveryDecision(
                strategy=data.get("strategy", "retry_same"),
                reasoning=data.get("reasoning", "LLM recommended this strategy."),
                wait_seconds=float(data.get("wait_seconds", 0)),
                modified_step=data.get("modified_step"),
                replan_hint=data.get("replan_hint", ""),
                learned_insight=data.get("learned_insight", ""),
            )
            self._log.info(
                f"LLM recovery decision: {decision.strategy} — {decision.reasoning[:100]}"
            )
            return decision

        except Exception as e:
            self._log.error(f"LLM recovery decision failed: {e}")
            # Safe fallback
            if retry_count < self._max_retries_per_step:
                return RecoveryDecision(
                    strategy="retry_same",
                    reasoning=f"LLM recovery failed ({e}). Falling back to simple retry.",
                    wait_seconds=2.0,
                )
            return RecoveryDecision(
                strategy="replan" if step.critical else "skip_step",
                reasoning="Max retries exceeded and LLM recovery unavailable.",
            )

    def _build_recovery_prompt(
        self,
        step: TaskStep,
        step_result: StepResult,
        observation: Observation,
        retry_count: int,
        memory_hints: Optional[List[str]],
        all_steps: Optional[List[TaskStep]],
    ) -> str:
        parts = [
            f"FAILED STEP:",
            f"  Step ID: {step.step_id}",
            f"  Action: {step.action}",
            f"  Description: {step.description}",
            f"  Parameters: {json.dumps(step.parameters)}",
            f"  Expected Outcome: {step.expected_outcome}",
            f"  Critical: {step.critical}",
            f"  Retry Count: {retry_count}",
            f"\nFAILURE DETAILS:",
            f"  Tool Status: {step_result.status}",
            f"  Error: {step_result.error or 'none'}",
            f"  Tool Result: {json.dumps(step_result.tool_result)[:500]}",
            f"\nOBSERVATION:",
            f"  Step Succeeded: {observation.step_succeeded}",
            f"  Error Type: {observation.error_type}",
            f"  Error Message: {observation.error_message}",
            f"  Current URL: {observation.page_url}",
            f"  Recommendation: {observation.recommendation}",
            f"  Failure Indicators: {observation.failure_indicators}",
        ]

        if memory_hints:
            parts.append(f"\nMEMORY HINTS:")
            for hint in memory_hints:
                parts.append(f"  - {hint}")

        if all_steps:
            remaining = [
                s for s in all_steps
                if s.step_id > step.step_id and s.status == "pending"
            ][:5]
            if remaining:
                parts.append(f"\nREMAINING STEPS (next {len(remaining)}):")
                for s in remaining:
                    parts.append(f"  - Step {s.step_id}: {s.description}")

        parts.append("\nDecide the best recovery strategy and respond with JSON.")
        return "\n".join(parts)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass
        return {}
