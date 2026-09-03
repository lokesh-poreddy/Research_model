"""researchforge/config/__init__.py — Configuration package.

RF-1.0.0-alpha.2.1: Canonical configuration hierarchy:
  Scope 1: RF software config (RFConfig)
  Scope 2: Infrastructure environment config (EnvConfig)
  Scope 3: Research-policy config (owned by ResearchSystemGenome.execution_config / sub-configs)
  Scope 4: Experiment config (owned per-experiment by ExperimentSpec)
"""
from .rf_config import RFConfig, default_rf_config
from .env_config import EnvConfig

__all__ = [
    "RFConfig",
    "default_rf_config",
    "EnvConfig",
]
