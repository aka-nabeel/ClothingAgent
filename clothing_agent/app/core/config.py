"""Unified environment configuration for the clothing agent."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """Runtime settings loaded from the workspace-level ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        env_prefix="CLOTHING_AGENT_",
        extra="ignore",
    )

    app_name: str = "Clothing AI Sales Agent"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    frontend_origin: str = "http://localhost:5173"
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    log_max_bytes: int = Field(default=5_000_000, ge=100_000)
    log_backup_count: int = Field(default=5, ge=1, le=20)
    state_dir: Path = Path("state/sessions")

    clothing_app_base_url: str = "http://127.0.0.1:8000"
    clothing_app_timeout_seconds: float = Field(default=12.0, gt=0, le=120)

    # Groq exposes an OpenAI-compatible chat-completions API, so the agent only
    # needs these provider settings.
    llm_api_base: str = "https://api.groq.com/openai/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: float = Field(default=45.0, gt=0, le=180)
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=500, ge=80, le=4000)
    allow_local_fallback: bool = True

    recent_message_limit: int = Field(default=12, ge=2, le=30)
    displayed_product_limit: int = Field(default=3, ge=1, le=6)
    maximum_clarification_questions: int = Field(default=2, ge=0, le=3)
    auto_detect_language_per_message: bool = Field(default=True)

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def empty_key_is_none(cls, value: object) -> object:
        """Treat an empty environment variable as an unconfigured API key with fallback to GROQ_API_KEY/OPENAI_API_KEY."""
        import os
        if value is None or value == "":
            fallback = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            return fallback if fallback else None
        return value


@lru_cache
def get_config() -> AgentConfig:
    """Return one cached configuration instance for the process."""

    return AgentConfig()
