"""
Configuration management for FPL Assistant using pydantic-settings.

Environment variables are loaded from .env file and validated at startup.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FPLSettings(BaseSettings):
    """FPL API credentials and settings."""

    email: str = Field(default="", description="FPL account email")
    password: SecretStr = Field(default=SecretStr(""), description="FPL account password")
    manager_id: int = Field(default=0, description="Your FPL team/manager ID")

    model_config = SettingsConfigDict(env_prefix="FPL_")


class LLMSettings(BaseSettings):
    """LLM provider API keys and configuration."""

    # API Keys
    openai_api_key: SecretStr = Field(
        default=SecretStr(""), description="OpenAI API key"
    )
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""), description="Anthropic API key"
    )
    deepseek_api_key: SecretStr = Field(
        default=SecretStr(""), description="DeepSeek API key"
    )

    # Model selection
    default_model: str = Field(
        default="gpt-4",
        description="Default model for complex explanations",
    )
    simple_model: str = Field(
        default="gpt-3.5-turbo",
        description="Model for simple queries (cost optimization)",
    )

    # Feature toggle
    enabled: bool = Field(default=True, description="Enable/disable LLM features")

    model_config = SettingsConfigDict(env_prefix="LLM_")

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key.get_secret_value())

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key.get_secret_value())

    @property
    def has_deepseek(self) -> bool:
        """Check if DeepSeek API key is configured."""
        return bool(self.deepseek_api_key.get_secret_value())

    @property
    def has_any_provider(self) -> bool:
        """Check if any LLM provider is configured."""
        return self.has_openai or self.has_anthropic or self.has_deepseek


class CacheSettings(BaseSettings):
    """Caching configuration."""

    dir: Path = Field(default=Path(".cache"), description="Cache directory path")
    bootstrap_ttl: int = Field(
        default=3600, description="Bootstrap-static cache TTL in seconds (1 hour)"
    )
    fixtures_ttl: int = Field(
        default=86400, description="Fixtures cache TTL in seconds (24 hours)"
    )

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    @field_validator("dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Convert string to Path."""
        return Path(v)


class OptimizerSettings(BaseSettings):
    """Optimization engine settings."""

    horizon: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Default planning horizon in gameweeks",
    )
    allow_hits: bool = Field(
        default=True, description="Allow point hits for additional transfers"
    )
    time_limit: int = Field(
        default=60, ge=10, le=300, description="Solver time limit in seconds"
    )

    model_config = SettingsConfigDict(env_prefix="OPTIMIZER_")


class AppSettings(BaseSettings):
    """General application settings."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    streamlit_port: int = Field(default=8501, description="Streamlit server port")

    model_config = SettingsConfigDict(env_prefix="")


class Settings(BaseSettings):
    """Main settings class combining all configuration sections."""

    fpl: FPLSettings = Field(default_factory=FPLSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    optimizer: OptimizerSettings = Field(default_factory=OptimizerSettings)
    app: AppSettings = Field(default_factory=AppSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    def validate_fpl_credentials(self) -> bool:
        """Check if FPL credentials are configured."""
        return (
            bool(self.fpl.email)
            and bool(self.fpl.password.get_secret_value())
            and self.fpl.manager_id > 0
        )

    def validate_llm_config(self) -> bool:
        """Check if LLM is properly configured."""
        if not self.llm.enabled:
            return True  # LLM disabled, no validation needed
        return self.llm.has_any_provider


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Call get_settings.cache_clear() to reload settings.

    Supports both:
    - Local: .env file
    - Streamlit Cloud: st.secrets
    """
    import os
    from dotenv import load_dotenv

    # Try to find and load .env from project root
    env_paths = [
        Path(".env"),
        Path(__file__).parent.parent.parent / ".env",  # fpl-assistant/.env
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break

    # Also check Streamlit secrets (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'fpl' in st.secrets:
            # Set environment variables from Streamlit secrets
            fpl_secrets = st.secrets.get('fpl', {})
            if 'manager_id' in fpl_secrets:
                os.environ['FPL_MANAGER_ID'] = str(fpl_secrets['manager_id'])
            if 'email' in fpl_secrets:
                os.environ['FPL_EMAIL'] = str(fpl_secrets['email'])
            if 'password' in fpl_secrets:
                os.environ['FPL_PASSWORD'] = str(fpl_secrets['password'])
    except ImportError:
        pass  # Streamlit not installed (CLI mode)
    except Exception:
        pass  # Secrets not available

    return Settings()


# Convenience function for quick access
def get_config() -> Settings:
    """Alias for get_settings()."""
    return get_settings()
