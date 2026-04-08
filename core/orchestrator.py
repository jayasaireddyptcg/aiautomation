from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from agents.planner import PlannerAgent, ExecutionPlan, TaskStep
from agents.executor import ExecutorAgent, StepResult
from agents.observer import ObserverAgent, Observation
from agents.recovery import RecoveryAgent, RecoveryDecision
from browser.controller import BrowserController
from memory.store import MemoryStore
from tools.browser_tools import BrowserToolkit
from config.settings import get_settings
from utils.logger import get_logger

log = get_logger("orchestrator")


class StepHistory(BaseModel):
    step: TaskStep
    result: StepResult
    observation: Observation
    recovery: Optional[RecoveryDecision] = None


class TaskResult(BaseModel):
    goal: str
    status: str  # success | failed | aborted | partial
    steps_total: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    replan_count: int = 0
    duration_seconds: float = 0.0
    final_url: str = ""
    final_title: str = ""
    success_criteria_met: bool = False
    history: List[StepHistory] = Field(default_factory=list)
    error: str = ""
    insights: List[str] = Field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""


class AgentOrchestrator:
    """
    Central orchestrator implementing the self-healing execution loop:

        Plan → Execute → Observe → (Succeed → Next) | (Fail → Recover → Retry/Replan)

    Coordinates:  Planner → Executor → Observer → Recovery Agent
    Backed by:    Memory Layer + BrowserController + LangChain Tools
    """

    def __init__(
        self,
        controller: Optional[BrowserController] = None,
        memory: Optional[MemoryStore] = None,
    ):
        settings = get_settings()
        self._settings = settings
        self._agent_cfg = settings.agent
        self._llm = settings.get_llm()

        self._controller = controller or BrowserController(llm=self._llm)
        self._memory = memory or MemoryStore(
            file_path=settings.memory.file_path,
            max_entries=settings.memory.max_entries,
        )

        self._toolkit = BrowserToolkit(self._controller)
        self._tools = self._toolkit.get_tools()

        self._planner = PlannerAgent(self._llm)
        self._executor = ExecutorAgent(self._llm, self._controller, self._tools)
        self._observer = ObserverAgent(self._llm, self._controller)
        self._recovery = RecoveryAgent(self._llm)

        self._log = get_logger("orchestrator")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, goal: str) -> TaskResult:
        """
        Execute a natural language goal end-to-end.
        Returns a TaskResult with full execution history.
        """
        started_at = datetime.utcnow().isoformat()
        start_time = asyncio.get_event_loop().time()

        self._log.info("=" * 70)
        self._log.info(f"GOAL: {goal}")
        self._log.info("=" * 70)

        result = TaskResult(goal=goal, started_at=started_at, status="in_progress")

        async with self._controller:
            try:
                # --- Phase 1: Planning ---
                site_domain = self._extract_domain(goal)
                memory_hints = self._memory.get_hints(goal, site_domain)
                if memory_hints:
                    self._log.info(f"Memory hints loaded: {len(memory_hints)}")
                    for h in memory_hints:
                        self._log.debug(f"  {h}")

                plan = await self._planner.plan(goal, memory_hints=memory_hints)
                result.steps_total = len(plan.steps)

                self._log.info(
                    f"Plan ready: {len(plan.steps)} steps — '{plan.goal_summary}'"
                )

                # --- Phase 2: Self-Healing Execution Loop ---
                await self._execute_plan(plan, result, goal, site_domain)

                # --- Phase 3: Final Verification ---
                await self._verify_success(plan, result)

            except Exception as e:
                self._log.error(f"Orchestrator fatal error: {e}", exc_info=True)
                result.status = "failed"
                result.error = str(e)

        elapsed = asyncio.get_event_loop().time() - start_time
        result.duration_seconds = round(elapsed, 2)
        result.completed_at = datetime.utcnow().isoformat()

        self._log_summary(result)
        return result

    # ------------------------------------------------------------------
    # Execution Loop
    # ------------------------------------------------------------------

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        result: TaskResult,
        goal: str,
        site_domain: str,
    ) -> None:
        """
        Main self-healing loop:
        Attempt → Observe → Detect Failure → Reason → Adapt → Retry → Learn
        """
        max_replans = self._agent_cfg.max_retries
        observations: List[Observation] = []
        step_index = 0

        while step_index < len(plan.steps):
            step = plan.steps[step_index]

            if step.status in ("success", "skipped"):
                step_index += 1
                continue

            self._log.info(
                f"[Step {step.step_id}/{len(plan.steps)}] "
                f"[{step.action.upper()}] {step.description}"
            )
            step.status = "in_progress"

            # Retry loop for this step
            retry_count = 0
            step_resolved = False

            while retry_count <= self._agent_cfg.max_retries:
                # ATTEMPT
                dom_state = await self._controller.get_dom_state()
                step_result = await self._executor.execute_step(step, dom_state)

                # OBSERVE
                observation = await self._observer.observe(
                    step, step_result, observations
                )
                observations.append(observation)
                result.history.append(
                    StepHistory(step=step, result=step_result, observation=observation)
                )

                self._log.info(
                    f"  Observation: succeeded={observation.step_succeeded} | "
                    f"confidence={observation.confidence:.2f} | "
                    f"recommendation={observation.recommendation}"
                )

                # SUCCESS PATH
                if observation.step_succeeded and observation.recommendation == "continue":
                    step.status = "success"
                    step.result = step_result.model_dump()
                    result.steps_succeeded += 1

                    # LEARN from success
                    self._memory.record_success(
                        goal_pattern=goal,
                        step_description=step.description,
                        action=step.action,
                        strategy_used=step_result.tool_called,
                        site_domain=site_domain,
                    )

                    step_resolved = True
                    break

                # FAILURE PATH
                if retry_count >= self._agent_cfg.max_retries:
                    break

                # RECOVER
                memory_hints = self._memory.get_hints(goal, site_domain, max_hints=3)
                decision = await self._recovery.decide(
                    step=step,
                    step_result=step_result,
                    observation=observation,
                    retry_count=retry_count,
                    memory_hints=memory_hints,
                    all_steps=plan.steps,
                )
                result.history[-1].recovery = decision

                self._log.warning(
                    f"  Recovery: [{decision.strategy}] {decision.reasoning[:80]}"
                )

                # Learn from failure/insight
                if observation.error_type != "none":
                    self._memory.record_failure(
                        goal_pattern=goal,
                        step_description=step.description,
                        action=step.action,
                        error_type=observation.error_type,
                        error_message=observation.error_message,
                        site_domain=site_domain,
                    )
                if decision.learned_insight:
                    self._memory.record_insight(
                        goal_pattern=goal,
                        insight=decision.learned_insight,
                        site_domain=site_domain,
                        step_description=step.description,
                    )
                    result.insights.append(decision.learned_insight)

                # APPLY RECOVERY STRATEGY
                if decision.strategy == "abort":
                    self._log.error("Abort signal from Recovery Agent.")
                    result.status = "aborted"
                    result.error = decision.reasoning
                    return

                elif decision.strategy == "skip_step":
                    step.status = "skipped"
                    result.steps_skipped += 1
                    step_resolved = True
                    break

                elif decision.strategy == "replan":
                    if plan.replan_count >= max_replans:
                        self._log.error(
                            f"Max replans ({max_replans}) reached. Aborting."
                        )
                        result.status = "failed"
                        result.error = "Max replan limit reached."
                        return

                    self._log.info("Triggering replan...")
                    completed = [s for s in plan.steps if s.status == "success"]
                    dom_state = await self._controller.get_dom_state()
                    plan = await self._planner.replan(
                        original_plan=plan,
                        failed_step=step,
                        failure_reason=decision.replan_hint or observation.error_message,
                        dom_state=dom_state,
                        completed_steps=completed,
                    )
                    result.replan_count += 1
                    step_index = 0  # Restart from beginning of new plan
                    step_resolved = True
                    break

                elif decision.strategy == "wait_and_retry":
                    wait = decision.wait_seconds or 3.0
                    self._log.info(f"  Waiting {wait}s before retry...")
                    await asyncio.sleep(wait)

                elif decision.strategy == "retry_modified" and decision.modified_step:
                    mod = decision.modified_step
                    step.action = mod.get("action", step.action)
                    step.description = mod.get("description", step.description)
                    step.parameters = mod.get("parameters", step.parameters)
                    step.expected_outcome = mod.get("expected_outcome", step.expected_outcome)
                    self._log.info(
                        f"  Modified step: [{step.action}] {step.description}"
                    )

                elif decision.strategy == "alternative_approach" and decision.modified_step:
                    mod = decision.modified_step
                    step.action = mod.get("action", step.action)
                    step.description = mod.get("description", step.description)
                    step.parameters = mod.get("parameters", step.parameters)
                    self._log.info(
                        f"  Alternative approach: [{step.action}] {step.description}"
                    )

                retry_count += 1
                step.retry_count = retry_count
                self._log.info(f"  Retrying step {step.step_id} (attempt {retry_count + 1})...")

            # Step exhausted all retries without resolution
            if not step_resolved:
                step.status = "failed"
                result.steps_failed += 1
                if step.critical:
                    self._log.error(
                        f"Critical step {step.step_id} failed after "
                        f"{retry_count} retries. Stopping."
                    )
                    result.status = "failed"
                    result.error = (
                        f"Critical step failed: {step.description}"
                    )
                    return
                else:
                    self._log.warning(
                        f"Non-critical step {step.step_id} failed. Continuing."
                    )
                    step.status = "skipped"

            step_index += 1

    # ------------------------------------------------------------------
    # Final Verification
    # ------------------------------------------------------------------

    async def _verify_success(
        self, plan: ExecutionPlan, result: TaskResult
    ) -> None:
        """Check if the overall goal was achieved."""
        try:
            dom_state = await self._controller.get_dom_state()
            result.final_url = dom_state.get("url", "")
            result.final_title = dom_state.get("title", "")
        except Exception:
            pass

        # Determine final status
        if result.status in ("failed", "aborted"):
            return

        total_non_skipped = result.steps_total - result.steps_skipped
        if total_non_skipped == 0:
            result.status = "partial"
            return

        success_rate = result.steps_succeeded / max(total_non_skipped, 1)

        if success_rate >= 0.8:
            result.status = "success"
            result.success_criteria_met = True
        elif success_rate >= 0.5:
            result.status = "partial"
        else:
            result.status = "failed"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_domain(self, goal: str) -> str:
        """Extract domain hint from a goal string for memory lookups."""
        words = goal.lower().split()
        for word in words:
            if "." in word and len(word) > 4:
                try:
                    parsed = urlparse(word if "://" in word else f"https://{word}")
                    if parsed.netloc:
                        return parsed.netloc.replace("www.", "")
                except Exception:
                    pass
        # Common site keywords
        site_keywords = {
            "gmail": "gmail.com",
            "github": "github.com",
            "google": "google.com",
            "twitter": "twitter.com",
            "linkedin": "linkedin.com",
            "amazon": "amazon.com",
            "youtube": "youtube.com",
            "facebook": "facebook.com",
        }
        for kw, domain in site_keywords.items():
            if kw in goal.lower():
                return domain
        return ""

    def _log_summary(self, result: TaskResult) -> None:
        self._log.info("=" * 70)
        self._log.info(f"TASK COMPLETE — Status: {result.status.upper()}")
        self._log.info(f"  Goal:        {result.goal[:80]}")
        self._log.info(f"  Duration:    {result.duration_seconds}s")
        self._log.info(
            f"  Steps:       {result.steps_succeeded}/{result.steps_total} succeeded "
            f"({result.steps_failed} failed, {result.steps_skipped} skipped)"
        )
        self._log.info(f"  Replans:     {result.replan_count}")
        self._log.info(f"  Final URL:   {result.final_url}")
        if result.error:
            self._log.error(f"  Error:       {result.error}")
        if result.insights:
            self._log.info(f"  Insights:    {len(result.insights)} learned")
        self._log.info("=" * 70)

    def get_memory_stats(self) -> Dict[str, Any]:
        return self._memory.get_stats()
