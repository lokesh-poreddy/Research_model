"""researchforge/config/env_config.py — Infrastructure and environment configuration.

Scope 2: Infrastructure environment parameters:
  - database path, API keys, cache dir, sandbox backend
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class EnvConfig:
    """Infrastructure-level configuration loaded from environment or local defaults."""
    db_path: Optional[str] = None
    cache_dir: str = ".rf_cache"
    sandbox_backend: str = "subprocess"
    openalex_email: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    github_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "EnvConfig":
        return cls(
            db_path=os.environ.get("RF_DB_PATH"),
            cache_dir=os.environ.get("RF_CACHE_DIR", ".rf_cache"),
            sandbox_backend=os.environ.get("RF_SANDBOX_BACKEND", "subprocess"),
            openalex_email=os.environ.get("OPENALEX_EMAIL"),
            semantic_scholar_api_key=os.environ.get("S2_API_KEY"),
            github_token=os.environ.get("GITHUB_TOKEN"),
        )

    def to_dict(self) -> dict:
        return {
            "db_path": self.db_path,
            "cache_dir": self.cache_dir,
            "sandbox_backend": self.sandbox_backend,
            "openalex_email": self.openalex_email,
            "semantic_scholar_api_key": "***" if self.semantic_scholar_api_key else None,
            "github_token": "***" if self.github_token else None,
        }
