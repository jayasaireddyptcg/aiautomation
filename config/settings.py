from __future__ import annotations

import os
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrowserSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BROWSER_", extra="ignore", populate_by_name=True)

    headless: bool = Field(default=False)
    slow_mo: int = Field(default=50)
    timeout: int = Field(default=30000)
    viewport_width: int = Field(default=1280)
    viewport_height: int = Field(default=720)
    # alias avoids double-prefix (BROWSER_BROWSER_TYPE → BROWSER_TYPE)
    browser_type: Literal["chromium", "firefox", "webkit"] = Field(
        default="chromium", alias="BROWSER_TYPE"
    )


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore", populate_by_name=True)

    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    max_steps: int = Field(default=20, alias="MAX_STEPS")
    step_timeout: int = Field(default=60, alias="STEP_TIMEOUT")
    recovery_max_attempts: int = Field(default=3, alias="RECOVERY_MAX_ATTEMPTS")


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMORY_", extra="ignore")

    file_path: str = Field(default="./memory/agent_memory.json")
    max_entries: int = Field(default=1000)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider
    llm_provider: Literal["openai", "google", "anthropic"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.1, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4096, alias="OPENAI_MAX_TOKENS")

    # Google
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-1.5-pro", alias="GOOGLE_MODEL")
    google_temperature: float = Field(default=0.1, alias="GOOGLE_TEMPERATURE")

    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL"
    )
    anthropic_temperature: float = Field(default=0.1, alias="ANTHROPIC_TEMPERATURE")
    anthropic_max_tokens: int = Field(default=4096, alias="ANTHROPIC_MAX_TOKENS")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/agent.log", alias="LOG_FILE")

    @field_validator("llm_provider", mode="before")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        return v.lower() if v else "openai"

    def get_llm(self):
        """Factory method: returns the configured LangChain chat model."""
        if self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider.")
            return ChatOpenAI(
                api_key=self.openai_api_key,
                model=self.openai_model,
                temperature=self.openai_temperature,
                max_tokens=self.openai_max_tokens,
            )

        elif self.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required for Google provider.")
            return ChatGoogleGenerativeAI(
                google_api_key=self.google_api_key,
                model=self.google_model,
                temperature=self.google_temperature,
            )

        elif self.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            if not self.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required for Anthropic provider."
                )
            return ChatAnthropic(
                api_key=self.anthropic_api_key,
                model=self.anthropic_model,
                temperature=self.anthropic_temperature,
                max_tokens=self.anthropic_max_tokens,
            )

        raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    @property
    def browser(self) -> BrowserSettings:
        return BrowserSettings()

    @property
    def agent(self) -> AgentSettings:
        return AgentSettings()

    @property
    def memory(self) -> MemorySettings:
        return MemorySettings()


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
