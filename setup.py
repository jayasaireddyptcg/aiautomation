"""
Quick setup script for the Universal Autonomous Web Agent.
Verifies dependencies, installs Playwright browsers, and checks configuration.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ required.")
        sys.exit(1)
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor} ✓")


def install_dependencies():
    print("\nInstalling Python dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: pip install failed:\n{result.stderr}")
        sys.exit(1)
    print("  Dependencies installed ✓")


def install_playwright():
    print("\nInstalling Playwright browsers...")
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: Playwright install failed:\n{result.stderr}")
        print("  Run manually: playwright install chromium")
    else:
        print("  Playwright chromium installed ✓")


def check_env_file():
    print("\nChecking .env configuration...")
    env_path = Path(".env")
    if not env_path.exists():
        example_path = Path(".env.example")
        if example_path.exists():
            import shutil
            shutil.copy(example_path, env_path)
            print("  Created .env from .env.example")
            print("  IMPORTANT: Edit .env and set your API key(s)!")
        else:
            print("  WARNING: No .env file found. Create one from .env.example")
    else:
        print("  .env file found ✓")

    # Check for at least one API key
    from dotenv import load_dotenv
    load_dotenv()

    providers = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "google": os.getenv("GOOGLE_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
    }

    configured = [p for p, k in providers.items() if k and "your_" not in k]
    if configured:
        print(f"  LLM providers configured: {', '.join(configured)} ✓")
    else:
        print("  WARNING: No LLM API key found. Set at least one in .env")


def verify_imports():
    print("\nVerifying module imports...")
    modules = [
        ("playwright.async_api", "Playwright"),
        ("langchain_core.messages", "LangChain Core"),
        ("langchain_openai", "LangChain OpenAI"),
        ("langchain_google_genai", "LangChain Google"),
        ("langchain_anthropic", "LangChain Anthropic"),
        ("pydantic", "Pydantic"),
        ("loguru", "Loguru"),
        ("dotenv", "python-dotenv"),
    ]

    all_ok = True
    for module, label in modules:
        try:
            __import__(module)
            print(f"  {label} ✓")
        except ImportError:
            print(f"  {label} ✗ (run: pip install -r requirements.txt)")
            all_ok = False

    return all_ok


def create_directories():
    print("\nCreating required directories...")
    for d in ["logs", "memory", "screenshots", "results"]:
        Path(d).mkdir(exist_ok=True)
        print(f"  {d}/ ✓")


def print_next_steps():
    print("\n" + "=" * 60)
    print("Setup complete! Next steps:")
    print("=" * 60)
    print()
    print("1. Edit .env with your API key:")
    print("     OPENAI_API_KEY=sk-...")
    print("     LLM_PROVIDER=openai")
    print()
    print("2. Run the demo:")
    print("     python main.py --demo")
    print()
    print("3. Run your first goal:")
    print('     python main.py --goal "Go to Wikipedia and search for AI"')
    print()
    print("4. Interactive mode:")
    print("     python main.py --interactive")
    print()


def main():
    print("=" * 60)
    print("  Universal Autonomous Web Agent — Setup")
    print("=" * 60)

    check_python_version()
    install_dependencies()
    install_playwright()
    check_env_file()
    imports_ok = verify_imports()
    create_directories()

    if imports_ok:
        print_next_steps()
    else:
        print("\nSome imports failed. Run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
