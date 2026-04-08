# Universal Autonomous Web Agent

A production-grade autonomous browser agent that combines **Playwright** for browser control and **LangChain** for LLM-based reasoning. Executes complex web tasks from natural language goals — self-healing when UI changes or errors occur.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AgentOrchestrator                          │
│                                                                 │
│  Natural Language Goal                                          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐    Plan    ┌──────────────┐                   │
│  │   Planner   │──────────▶│ ExecutionPlan │                   │
│  │   Agent     │           │  (N steps)   │                   │
│  └─────────────┘           └──────┬───────┘                   │
│                                   │                             │
│              ┌────────────────────┘                            │
│              │  Self-Healing Loop                              │
│              │  ┌─────────────────────────────────────┐       │
│              │  │                                     │       │
│              ▼  ▼                                     │       │
│  ┌────────────────┐   StepResult   ┌───────────────┐  │       │
│  │    Executor    │──────────────▶│    Observer   │  │       │
│  │    Agent       │               │    Agent      │  │       │
│  └────────────────┘               └───────┬───────┘  │       │
│         ▲                                 │           │       │
│         │                   fail          ▼           │       │
│         │             ┌─────────────────────────┐     │       │
│         └─────────────│     Recovery Agent      │─────┘       │
│                       │  retry|replan|skip|abort│             │
│                       └─────────────────────────┘             │
│                                   │                             │
│                          ┌────────▼────────┐                   │
│                          │  Memory Layer   │                   │
│                          │ (JSON-backed)   │                   │
│                          └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Self-Healing Loop

```
Attempt → Observe → Detect Failure → Reason → Adapt → Retry → Learn
```

## Components

| Component | File | Responsibility |
|---|---|---|
| **Planner Agent** | `agents/planner.py` | Decomposes goals into ordered TaskSteps |
| **Executor Agent** | `agents/executor.py` | Executes steps via Playwright browser tools |
| **Observer Agent** | `agents/observer.py` | Monitors DOM state, classifies errors |
| **Recovery Agent** | `agents/recovery.py` | Decides retry/replan/skip/abort strategies |
| **Memory Layer** | `memory/store.py` | Persists strategies, failures, and insights |
| **Browser Controller** | `browser/controller.py` | Playwright lifecycle + all interactions |
| **Selector Engine** | `browser/selector_engine.py` | Semantic-first element resolution |
| **Browser Tools** | `tools/browser_tools.py` | LangChain tool wrappers (15 tools) |
| **Orchestrator** | `core/orchestrator.py` | Coordinates all agents + self-healing loop |

## Supported LLM Providers

| Provider | Models |
|---|---|
| **OpenAI** | `gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini` |
| **Google** | `gemini-1.5-pro`, `gemini-1.5-flash` |
| **Anthropic** | `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229` |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Playwright browsers

```bash
playwright install chromium
# or for all browsers:
playwright install
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your API keys
```

**Minimum required — pick ONE provider:**

```env
# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# OR Google
LLM_PROVIDER=google
GOOGLE_API_KEY=AI...

# OR Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Single goal

```bash
python main.py --goal "Go to Wikipedia and search for 'machine learning'"
```

### Interactive mode (multi-goal session)

```bash
python main.py --interactive
```

### Built-in demo

```bash
python main.py --demo
```

### Override provider at runtime

```bash
python main.py --goal "..." --provider anthropic
```

### Headless mode

```bash
python main.py --goal "..." --headless
```

### Save result to JSON

```bash
python main.py --goal "..." --output results/task_result.json
```

### View memory statistics

```bash
python main.py --memory-stats
```

## Example Goals

```bash
# Web navigation & reading
python main.py --goal "Go to https://news.ycombinator.com and list the top 5 headlines"

# Form filling
python main.py --goal "Navigate to https://httpbin.org/forms/post, fill in the customer name with 'Jane Smith' and submit"

# Multi-step search
python main.py --goal "Search for 'LangChain tutorials' on Google and open the first result"

# Authentication + action (requires credentials in goal or .env)
python main.py --goal "Log into GitHub at github.com with email user@example.com and password mypassword, then navigate to the repository 'myrepo'"

# Gmail-style compose (requires Gmail session)
python main.py --goal "Open Gmail, compose a new email to test@example.com with subject 'Hello' and body 'This is a test message', then send it"
```

## Programmatic API

```python
import asyncio
from core.orchestrator import AgentOrchestrator

async def main():
    orchestrator = AgentOrchestrator()
    result = await orchestrator.run(
        "Go to https://example.com and read the main heading"
    )
    print(f"Status: {result.status}")
    print(f"Steps:  {result.steps_succeeded}/{result.steps_total}")
    print(f"URL:    {result.final_url}")

asyncio.run(main())
```

## Browser Tools Available

| Tool | Description |
|---|---|
| `navigate` | Navigate to a URL |
| `click` | Click an element (semantic) |
| `type_text` | Type with human-like delay |
| `fill` | Fill field instantly |
| `press_key` | Press keyboard key |
| `wait_for_element` | Wait until element appears |
| `wait_seconds` | Fixed wait |
| `get_page_text` | Read visible page text |
| `get_element_text` | Read specific element text |
| `get_dom_state` | Full structured DOM snapshot |
| `element_exists` | Check element presence |
| `scroll` | Scroll page |
| `take_screenshot` | Capture screenshot |
| `hover` | Hover over element |
| `select_option` | Select dropdown option |

## Selector Strategy (Priority Order)

1. **Role-based** — `get_by_role("button", name="Submit")`
2. **Text-based** — `get_by_text("Sign In")`
3. **Aria-label** — `get_by_label("Email address")`
4. **Placeholder** — `get_by_placeholder("Search...")`
5. **Form label** — `get_by_label("Password")`
6. **LLM-assisted** — LLM analyzes DOM snapshot → generates CSS selector

No hardcoded selectors. All resolution is semantic and runtime-computed.

## Recovery Strategies

| Strategy | When Used |
|---|---|
| `retry_same` | Transient errors, network glitches |
| `retry_modified` | Element not found → try alternative selector |
| `wait_and_retry` | Rate limits, lazy-loading, animations |
| `alternative_approach` | Try a different interaction method |
| `skip_step` | Non-critical step fails after max retries |
| `replan` | Page state changed, redirect, login wall |
| `abort` | CAPTCHA detected, access denied |

## Memory System

The agent learns from every execution:

- **Success records** — which strategies worked for which elements/sites
- **Failure records** — what errors occur on which sites/flows
- **Insights** — synthesized learnings from Recovery Agent
- **Selector cache** — working selectors for known elements

Memory is stored in `memory/agent_memory.json` and automatically consulted at the start of each task for relevant hints.

## Project Structure

```
ai_automation/
├── main.py                    # Entry point + CLI
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py            # Pydantic settings (LLM + browser + agent)
├── agents/
│   ├── planner.py             # Planner Agent
│   ├── executor.py            # Executor Agent
│   ├── observer.py            # Observer Agent
│   └── recovery.py            # Recovery Agent
├── browser/
│   ├── controller.py          # Playwright browser controller
│   └── selector_engine.py     # Semantic selector resolution
├── tools/
│   └── browser_tools.py       # 15 LangChain browser tools
├── memory/
│   └── store.py               # JSON-backed memory store
├── core/
│   └── orchestrator.py        # Self-healing orchestration loop
├── utils/
│   └── logger.py              # Loguru structured logging
└── logs/                      # Auto-created log files
```

## Extending the Agent

### Add a new browser tool

```python
# In tools/browser_tools.py
class MyCustomInput(BaseModel):
    target: str = Field(description="...")

class MyCustomTool(BaseTool):
    name: str = "my_custom_tool"
    description: str = "..."
    args_schema: Type[BaseModel] = MyCustomInput
    controller: Any = Field(exclude=True)
    model_config = {"arbitrary_types_allowed": True}

    async def _arun(self, target: str) -> str:
        # Use self.controller to interact with browser
        ...
```

Then add to `BrowserToolkit.get_tools()`.

### Target a specific website

Create a specialized orchestrator subclass that pre-loads site-specific context:

```python
class GmailAgent(AgentOrchestrator):
    async def run(self, goal: str):
        # Pre-navigate to Gmail, inject credentials from env
        return await super().run(f"On Gmail: {goal}")
```
