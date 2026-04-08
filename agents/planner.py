from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from utils.logger import get_logger

log = get_logger("planner")

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent of an autonomous web automation system.

Your job: decompose a high-level user goal into a precise, ordered list of browser actions.

RULES:
1. Think step-by-step before generating the plan.
2. Each step must be a single, atomic browser action.
3. Prefer semantic descriptions over CSS selectors (e.g., "Sign In button" not "#btn-login").
4. Include verification steps (e.g., "Verify login succeeded by checking for dashboard element").
5. Anticipate common failure points and include wait steps where needed.
6. Steps must be sequential and depend on prior steps completing successfully.
7. Use the available tools: navigate, click, type_text, fill, press_key, wait_for_element,
   wait_seconds, get_page_text, get_dom_state, element_exists, scroll, take_screenshot,
   hover, select_option, get_element_text.

OUTPUT FORMAT (strict JSON, no markdown):
{
  "goal_summary": "brief restatement of the goal",
  "estimated_steps": <number>,
  "steps": [
    {
      "step_id": 1,
      "action": "navigate | click | type_text | fill | press_key | wait_for_element | wait_seconds | get_page_text | get_dom_state | element_exists | scroll | take_screenshot | hover | select_option | get_element_text",
      "description": "human-readable description of what this step does",
      "parameters": {
        // action-specific parameters matching the tool's input schema
      },
      "expected_outcome": "what should happen if this step succeeds",
      "critical": true  // false if step is optional
    }
  ],
  "success_criteria": "how to know the overall goal was achieved"
}"""


class TaskStep(BaseModel):
    step_id: int
    action: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str = ""
    critical: bool = True
    status: str = "pending"  # pending | in_progress | success | failed | skipped
    result: Optional[Dict[str, Any]] = None
    retry_count: int = 0


class ExecutionPlan(BaseModel):
    goal: str
    goal_summary: str = ""
    estimated_steps: int = 0
    steps: List[TaskStep] = Field(default_factory=list)
    success_criteria: str = ""
    replanned: bool = False
    replan_count: int = 0


class PlannerAgent:
    """
    Planner Agent: converts a natural language goal into a structured ExecutionPlan.
    Can also replan when execution fails or conditions change.
    """

    def __init__(self, llm):
        self.llm = llm
        self._log = get_logger("planner")

    async def plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        memory_hints: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        """Generate an execution plan for the given goal."""
        self._log.info(f"Planning goal: {goal}")

        prompt = self._build_plan_prompt(goal, context, memory_hints)
        response = await self.llm.ainvoke(
            [SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )

        plan_data = self._parse_response(response.content)
        plan = self._build_plan(goal, plan_data)

        self._log.info(
            f"Plan created: {len(plan.steps)} steps for goal '{plan.goal_summary}'"
        )
        for step in plan.steps:
            self._log.debug(
                f"  Step {step.step_id}: [{step.action}] {step.description}"
            )

        return plan

    async def replan(
        self,
        original_plan: ExecutionPlan,
        failed_step: TaskStep,
        failure_reason: str,
        dom_state: Optional[Dict[str, Any]] = None,
        completed_steps: Optional[List[TaskStep]] = None,
        memory_hints: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        """Replan from a failure point."""
        self._log.info(
            f"Replanning from step {failed_step.step_id}: {failure_reason}"
        )

        prompt = self._build_replan_prompt(
            original_plan, failed_step, failure_reason, dom_state, completed_steps
        )
        response = await self.llm.ainvoke(
            [SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )

        plan_data = self._parse_response(response.content)
        new_plan = self._build_plan(original_plan.goal, plan_data)
        new_plan.replanned = True
        new_plan.replan_count = original_plan.replan_count + 1

        self._log.info(
            f"Replan #{new_plan.replan_count}: {len(new_plan.steps)} steps"
        )
        return new_plan

    def _build_plan_prompt(
        self,
        goal: str,
        context: Optional[Dict[str, Any]],
        memory_hints: Optional[List[str]],
    ) -> str:
        parts = [f"GOAL: {goal}"]

        if context:
            parts.append(f"\nCURRENT CONTEXT:\n{json.dumps(context, indent=2)}")

        if memory_hints:
            parts.append("\nMEMORY HINTS (learned from past executions):")
            for hint in memory_hints:
                parts.append(f"  - {hint}")

        parts.append("\nGenerate the execution plan as JSON.")
        return "\n".join(parts)

    def _build_replan_prompt(
        self,
        original_plan: ExecutionPlan,
        failed_step: TaskStep,
        failure_reason: str,
        dom_state: Optional[Dict[str, Any]],
        completed_steps: Optional[List[TaskStep]],
    ) -> str:
        completed_ids = [s.step_id for s in (completed_steps or []) if s.status == "success"]
        parts = [
            f"ORIGINAL GOAL: {original_plan.goal}",
            f"\nFAILED AT STEP {failed_step.step_id}: {failed_step.description}",
            f"FAILURE REASON: {failure_reason}",
            f"\nCOMPLETED STEPS: {completed_ids}",
        ]

        if dom_state:
            state_str = json.dumps(dom_state, indent=2)[:2000]
            parts.append(f"\nCURRENT PAGE STATE:\n{state_str}")

        parts.append(
            "\nGenerate a REVISED plan to complete the remaining goal. "
            "Do NOT repeat already completed steps. Start from the current page state."
        )
        return "\n".join(parts)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError as e:
            self._log.warning(f"Failed to parse plan JSON: {e}")
        return {}

    def _build_plan(self, goal: str, data: Dict[str, Any]) -> ExecutionPlan:
        """Convert raw plan dict to ExecutionPlan model."""
        steps = []
        for i, s in enumerate(data.get("steps", []), start=1):
            steps.append(
                TaskStep(
                    step_id=s.get("step_id", i),
                    action=s.get("action", "get_dom_state"),
                    description=s.get("description", ""),
                    parameters=s.get("parameters", {}),
                    expected_outcome=s.get("expected_outcome", ""),
                    critical=s.get("critical", True),
                )
            )

        return ExecutionPlan(
            goal=goal,
            goal_summary=data.get("goal_summary", goal),
            estimated_steps=data.get("estimated_steps", len(steps)),
            steps=steps,
            success_criteria=data.get("success_criteria", ""),
        )
