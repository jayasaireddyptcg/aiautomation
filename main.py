#!/usr/bin/env python3
"""
Universal Autonomous Web Agent
Entry point for running web automation tasks via natural language goals.

Usage:
    python main.py --goal "Log into GitHub and star the microsoft/vscode repository"
    python main.py --goal "Search for 'LangChain' on Google and return the first 5 results"
    python main.py --interactive
    python main.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config.settings import get_settings
from core.orchestrator import AgentOrchestrator, TaskResult
from utils.logger import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Universal Autonomous Web Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --goal "Go to Wikipedia and search for 'artificial intelligence'"
  python main.py --goal "Open Hacker News and find the top story"
  python main.py --demo
  python main.py --interactive
        """,
    )
    parser.add_argument(
        "--goal",
        type=str,
        help="Natural language goal for the agent to execute.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt for goals repeatedly).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in demo tasks.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "google", "anthropic"],
        help="Override LLM provider from .env.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=None,
        help="Run browser in headless mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save task result JSON to this file path.",
    )
    parser.add_argument(
        "--memory-stats",
        action="store_true",
        help="Print memory statistics and exit.",
    )
    return parser.parse_args()


async def run_goal(goal: str, output_path: str | None = None) -> TaskResult:
    """Run a single goal and return the result."""
    log = get_logger("main")
    log.info(f"Starting agent for goal: {goal}")

    orchestrator = AgentOrchestrator()
    result = await orchestrator.run(goal)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        log.info(f"Result saved to: {output_path}")

    return result


async def run_interactive() -> None:
    """Interactive mode: repeatedly prompt user for goals."""
    log = get_logger("main")
    print("\n" + "=" * 60)
    print("  Universal Autonomous Web Agent — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'memory' to show memory statistics.")
    print("=" * 60 + "\n")

    orchestrator = AgentOrchestrator()

    while True:
        try:
            goal = input("Goal > ").strip()
            if not goal:
                continue
            if goal.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if goal.lower() == "memory":
                stats = orchestrator.get_memory_stats()
                print(json.dumps(stats, indent=2))
                continue

            result = await orchestrator.run(goal)
            print(f"\n{'=' * 60}")
            print(f"Status: {result.status.upper()}")
            print(f"Steps:  {result.steps_succeeded}/{result.steps_total} succeeded")
            print(f"Time:   {result.duration_seconds}s")
            if result.final_url:
                print(f"URL:    {result.final_url}")
            if result.error:
                print(f"Error:  {result.error}")
            print("=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            log.error(f"Error: {e}")


async def run_demo() -> None:
    """Run built-in demonstration tasks."""
    log = get_logger("main")

    demo_goals = [
        "Navigate to https://httpbin.org/forms/post and fill in the customer name field with 'John Doe', then submit the form",
        "Go to https://example.com and read the page title and main heading",
        "Navigate to https://quotes.toscrape.com and find the first quote on the page",
    ]

    print("\n" + "=" * 60)
    print("  Universal Autonomous Web Agent — Demo Mode")
    print(f"  Running {len(demo_goals)} demonstration tasks")
    print("=" * 60 + "\n")

    for i, goal in enumerate(demo_goals, 1):
        print(f"\nDemo Task {i}/{len(demo_goals)}:")
        print(f"  {goal}\n")

        try:
            result = await run_goal(goal)
            status_icon = "✓" if result.status == "success" else "✗"
            print(f"\n{status_icon} [{result.status.upper()}] {result.duration_seconds}s")
        except Exception as e:
            log.error(f"Demo task {i} failed: {e}")
            print(f"\n✗ [ERROR] {e}")

        if i < len(demo_goals):
            print("\nNext task in 2 seconds...")
            await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("Demo complete.")


def print_memory_stats() -> None:
    from memory.store import MemoryStore
    settings = get_settings()
    store = MemoryStore(
        file_path=settings.memory.file_path,
        max_entries=settings.memory.max_entries,
    )
    stats = store.get_stats()
    print(json.dumps(stats, indent=2))


def print_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════╗
║         Universal Autonomous Web Agent v1.0              ║
║  Playwright + LangChain  |  Self-Healing  |  Multi-LLM  ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main() -> int:
    args = parse_args()

    # Override settings from CLI args
    import os
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.headless is not None:
        os.environ["BROWSER_HEADLESS"] = str(args.headless).lower()

    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_file=settings.log_file)

    print_banner()

    if args.memory_stats:
        print_memory_stats()
        return 0

    if args.demo:
        await run_demo()
        return 0

    if args.interactive:
        await run_interactive()
        return 0

    if args.goal:
        result = await run_goal(args.goal, args.output)
        return 0 if result.status in ("success", "partial") else 1

    # No args — show help
    print("No goal provided. Use --goal, --interactive, or --demo.\n")
    print("Quick start:")
    print('  python main.py --goal "Go to Wikipedia and search for machine learning"')
    print("  python main.py --interactive")
    print("  python main.py --demo")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
