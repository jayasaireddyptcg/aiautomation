from __future__ import annotations

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from utils.logger import get_logger

log = get_logger("memory")


class MemoryType(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    STRATEGY = "strategy"
    SELECTOR = "selector"
    INSIGHT = "insight"


class MemoryEntry(BaseModel):
    id: str = ""
    type: MemoryType
    goal_pattern: str  # Normalized version of the goal/task context
    site_domain: str = ""  # e.g., "gmail.com", "github.com"
    step_description: str = ""
    action: str = ""
    selector_used: str = ""
    strategy_used: str = ""
    outcome: str = ""  # success | failed
    error_type: str = ""
    insight: str = ""  # Key learning
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    use_count: int = 1
    success_count: int = 0


class MemoryStore:
    """
    Persistent JSON-backed memory store.
    Records successful strategies, failure patterns, and learned insights.
    Provides relevant hints for new task executions.
    """

    def __init__(self, file_path: str = "./memory/agent_memory.json", max_entries: int = 1000):
        self.file_path = Path(file_path)
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []
        self._log = get_logger("memory")
        self._load()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load memory from disk."""
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                self._entries = [MemoryEntry(**e) for e in data.get("entries", [])]
                self._log.info(f"Loaded {len(self._entries)} memory entries from {self.file_path}")
            except Exception as e:
                self._log.warning(f"Failed to load memory: {e}. Starting fresh.")
                self._entries = []
        else:
            self._entries = []

    def _save(self) -> None:
        """Persist memory to disk."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"entries": [e.model_dump() for e in self._entries]}
            self.file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            self._log.error(f"Failed to save memory: {e}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_success(
        self,
        goal_pattern: str,
        step_description: str,
        action: str,
        strategy_used: str = "",
        selector_used: str = "",
        site_domain: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Record a successful step execution."""
        entry = self._find_existing(goal_pattern, step_description, action)
        if entry:
            entry.use_count += 1
            entry.success_count += 1
            entry.timestamp = datetime.utcnow().isoformat()
        else:
            entry = MemoryEntry(
                id=self._new_id(),
                type=MemoryType.SUCCESS,
                goal_pattern=goal_pattern,
                site_domain=site_domain,
                step_description=step_description,
                action=action,
                selector_used=selector_used,
                strategy_used=strategy_used,
                outcome="success",
                success_count=1,
                metadata=metadata or {},
            )
            self._add(entry)

        self._save()
        self._log.debug(f"Recorded success: {step_description[:60]}")
        return entry

    def record_failure(
        self,
        goal_pattern: str,
        step_description: str,
        action: str,
        error_type: str = "",
        error_message: str = "",
        site_domain: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Record a failed step execution."""
        entry = MemoryEntry(
            id=self._new_id(),
            type=MemoryType.FAILURE,
            goal_pattern=goal_pattern,
            site_domain=site_domain,
            step_description=step_description,
            action=action,
            outcome="failed",
            error_type=error_type,
            insight=error_message[:200] if error_message else "",
            metadata=metadata or {},
        )
        self._add(entry)
        self._save()
        self._log.debug(f"Recorded failure: {step_description[:60]} ({error_type})")
        return entry

    def record_insight(
        self,
        goal_pattern: str,
        insight: str,
        site_domain: str = "",
        step_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Record a learned insight (from recovery agent)."""
        entry = MemoryEntry(
            id=self._new_id(),
            type=MemoryType.INSIGHT,
            goal_pattern=goal_pattern,
            site_domain=site_domain,
            step_description=step_description,
            insight=insight,
            outcome="insight",
            metadata=metadata or {},
        )
        self._add(entry)
        self._save()
        self._log.info(f"Insight recorded: {insight[:80]}")
        return entry

    def record_selector_strategy(
        self,
        step_description: str,
        selector: str,
        strategy: str,
        site_domain: str = "",
        success: bool = True,
    ) -> None:
        """Remember which selectors worked for which elements."""
        entry = MemoryEntry(
            id=self._new_id(),
            type=MemoryType.SELECTOR,
            goal_pattern=step_description,
            site_domain=site_domain,
            step_description=step_description,
            selector_used=selector,
            strategy_used=strategy,
            outcome="success" if success else "failed",
            success_count=1 if success else 0,
        )
        self._add(entry)
        self._save()

    # ------------------------------------------------------------------
    # Read / Query
    # ------------------------------------------------------------------

    def get_hints(
        self,
        goal_pattern: str,
        site_domain: str = "",
        max_hints: int = 5,
    ) -> List[str]:
        """
        Retrieve relevant memory hints for a given goal.
        Returns human-readable strings for injection into agent prompts.
        """
        relevant = self._find_relevant(goal_pattern, site_domain)
        hints = []

        for entry in relevant[:max_hints]:
            if entry.type == MemoryType.SUCCESS:
                hints.append(
                    f"[SUCCESS] '{entry.step_description}' worked via "
                    f"{entry.strategy_used or entry.action} "
                    f"(used {entry.use_count}x, success rate: "
                    f"{entry.success_count}/{entry.use_count})"
                )
            elif entry.type == MemoryType.FAILURE:
                hints.append(
                    f"[FAILURE] '{entry.step_description}' failed with "
                    f"{entry.error_type}: {entry.insight[:100]}"
                )
            elif entry.type == MemoryType.INSIGHT:
                hints.append(f"[INSIGHT] {entry.insight}")
            elif entry.type == MemoryType.SELECTOR:
                if entry.outcome == "success":
                    hints.append(
                        f"[SELECTOR] For '{entry.step_description}', selector "
                        f"'{entry.selector_used}' worked (strategy: {entry.strategy_used})"
                    )

        return hints

    def get_successful_selector(
        self, step_description: str, site_domain: str = ""
    ) -> Optional[str]:
        """Look up a previously successful selector for an element."""
        for entry in reversed(self._entries):
            if (
                entry.type == MemoryType.SELECTOR
                and entry.outcome == "success"
                and self._text_similar(entry.step_description, step_description)
                and (not site_domain or entry.site_domain == site_domain)
            ):
                return entry.selector_used
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        total = len(self._entries)
        by_type = {}
        for e in self._entries:
            by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
        return {
            "total_entries": total,
            "by_type": by_type,
            "file_path": str(self.file_path),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        # Prune oldest entries if over limit
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def _find_existing(
        self, goal_pattern: str, step_description: str, action: str
    ) -> Optional[MemoryEntry]:
        for entry in self._entries:
            if (
                entry.action == action
                and self._text_similar(entry.step_description, step_description)
                and self._text_similar(entry.goal_pattern, goal_pattern)
            ):
                return entry
        return None

    def _find_relevant(
        self, goal_pattern: str, site_domain: str, max_results: int = 10
    ) -> List[MemoryEntry]:
        """Find entries relevant to the given goal/domain."""
        scored = []
        for entry in self._entries:
            score = 0
            if site_domain and entry.site_domain == site_domain:
                score += 3
            if self._text_similar(entry.goal_pattern, goal_pattern):
                score += 2
            if entry.type == MemoryType.SUCCESS:
                score += 1
            if entry.use_count > 1:
                score += 1
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:max_results]]

    def _text_similar(self, a: str, b: str, threshold: float = 0.4) -> bool:
        """Simple keyword-overlap similarity check."""
        if not a or not b:
            return False
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
        return overlap >= threshold

    def _new_id(self) -> str:
        return f"mem_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
