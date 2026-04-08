"""
Example task definitions for the Universal Autonomous Web Agent.

Each task demonstrates a different workflow category:
- Web scraping / reading
- Form filling & submission
- Multi-step navigation
- Search workflows
- Compose & send (email-like)
- Login + authenticated action
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from core.orchestrator import AgentOrchestrator, TaskResult
from utils.logger import setup_logging, get_logger

setup_logging()
log = get_logger("examples")


# -----------------------------------------------------------------------
# Task definitions
# -----------------------------------------------------------------------

TASKS = {
    "web_reading": (
        "Navigate to https://example.com and read the page title and main paragraph text."
    ),

    "hn_headlines": (
        "Go to https://news.ycombinator.com and list the top 5 story titles "
        "visible on the front page."
    ),

    "wikipedia_search": (
        "Go to https://en.wikipedia.org, search for 'Large Language Models', "
        "and read the first two paragraphs of the article."
    ),

    "form_fill": (
        "Navigate to https://httpbin.org/forms/post. "
        "Fill in the customer name field with 'Alice Smith', "
        "the telephone field with '555-0123', "
        "the email field with 'alice@example.com', "
        "and submit the form. Verify the submission was successful."
    ),

    "google_search": (
        "Go to https://www.google.com, search for 'Playwright Python automation', "
        "and return the titles of the first 3 organic results."
    ),

    "github_browse": (
        "Navigate to https://github.com/microsoft/playwright-python "
        "and read the repository description and star count."
    ),

    "quotes_scrape": (
        "Go to https://quotes.toscrape.com and collect the text and author "
        "of the first 3 quotes on the page."
    ),

    "multi_step_nav": (
        "Go to https://httpbin.org, click on the 'HTTP Methods' link, "
        "then verify you can see the GET endpoint documentation."
    ),

    # Gmail-style compose (requires active Gmail session / credentials)
    "gmail_compose": (
        "Open Gmail at https://mail.google.com. "
        "Click the Compose button to open a new email. "
        "In the To field, enter 'recipient@example.com'. "
        "In the Subject field, enter 'Automated Test Email'. "
        "In the message body, type: 'Hello, this is an automated test message sent by the web agent.'. "
        "Click the Send button and verify the email was sent."
    ),

    # Login workflow (replace credentials with real ones)
    "github_login": (
        "Navigate to https://github.com/login. "
        "Enter the email 'your_email@example.com' in the email/username field. "
        "Enter the password 'your_password' in the password field. "
        "Click the Sign In button. "
        "Verify you are logged in by checking for the user avatar or dashboard."
    ),

    "duckduckgo_search": (
        "Go to https://duckduckgo.com and search for 'autonomous web agents 2024'. "
        "Read the titles and URLs of the first 5 search results."
    ),

    "wikipedia_random": (
        "Go to https://en.wikipedia.org/wiki/Special:Random to open a random Wikipedia article. "
        "Read the article title and first paragraph, then return a summary."
    ),
}


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

async def run_task(task_name: str, goal: str) -> TaskResult:
    """Execute a single named task."""
    log.info(f"Running task: {task_name}")
    log.info(f"Goal: {goal}")

    orchestrator = AgentOrchestrator()
    result = await orchestrator.run(goal)

    print(f"\n{'─' * 60}")
    print(f"Task:    {task_name}")
    print(f"Status:  {result.status.upper()}")
    print(f"Steps:   {result.steps_succeeded}/{result.steps_total} succeeded")
    print(f"Time:    {result.duration_seconds}s")
    if result.final_url:
        print(f"URL:     {result.final_url}")
    if result.final_title:
        print(f"Title:   {result.final_title}")
    if result.error:
        print(f"Error:   {result.error}")
    if result.insights:
        print(f"Learned: {len(result.insights)} insight(s)")
        for ins in result.insights:
            print(f"  • {ins}")
    print("─" * 60)

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run example web agent tasks")
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()) + ["all"],
        default="web_reading",
        help="Which example task to run (default: web_reading)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available example tasks",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable example tasks:\n")
        for name, goal in TASKS.items():
            print(f"  {name:<20} {goal[:70]}...")
        return

    if args.task == "all":
        # Run safe, non-authenticated tasks
        safe_tasks = [
            "web_reading", "quotes_scrape", "wikipedia_random", "duckduckgo_search"
        ]
        for name in safe_tasks:
            await run_task(name, TASKS[name])
            await asyncio.sleep(1)
    else:
        await run_task(args.task, TASKS[args.task])


if __name__ == "__main__":
    asyncio.run(main())
