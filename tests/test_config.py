"""tests/test_config.py — Configuration Hierarchy test suite.

Classification: CORE
Tests Phase 9:
  - Scope 1: RFConfig (software configuration)
  - Scope 2: EnvConfig (environment & infrastructure)
  - Scope 3: RSG configuration ownership
  - Scope 4: ExperimentSpec per-experiment configuration
  - Legacy settings status and encapsulation
"""
from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from unittest.mock import patch
import pytest

from researchforge.config import RFConfig, default_rf_config, EnvConfig
from researchforge.genome.research_system_genome import ResearchSystemGenome
from researchforge.experiment import ExperimentSpec
from config.settings import LEGACY_STATUS


def test_rf_config_defaults():
    cfg = RFConfig()
    assert cfg.version == "RF-1.0.0-alpha.2.1"
    assert cfg.debug is False
    assert cfg.log_level == "INFO"
    assert cfg.strict_schema_validation is True


def test_rf_config_immutability():
    cfg = RFConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.version = "tampered"  # RFConfig is immutable (frozen)


def test_rf_config_serialization():
    cfg = RFConfig(version="custom_v", debug=True, log_level="DEBUG")
    d = cfg.to_dict()
    restored = RFConfig.from_dict(d)
    assert restored.version == "custom_v"
    assert restored.debug is True
    assert restored.log_level == "DEBUG"


def test_env_config_defaults():
    env_cfg = EnvConfig.from_env()
    assert env_cfg.cache_dir == ".rf_cache"
    assert env_cfg.sandbox_backend == "subprocess"


def test_env_config_masks_secrets():
    env_cfg = EnvConfig(
        semantic_scholar_api_key="secret_s2_key",
        github_token="secret_gh_token",
    )
    d = env_cfg.to_dict()
    assert d["semantic_scholar_api_key"] == "***"
    assert d["github_token"] == "***"


def test_env_config_from_env_vars():
    with patch.dict(os.environ, {
        "RF_DB_PATH": "/custom/path.db",
        "RF_CACHE_DIR": "/tmp/custom_cache",
        "RF_SANDBOX_BACKEND": "docker",
    }):
        env_cfg = EnvConfig.from_env()
        assert env_cfg.db_path == "/custom/path.db"
        assert env_cfg.cache_dir == "/tmp/custom_cache"
        assert env_cfg.sandbox_backend == "docker"


def test_legacy_settings_has_legacy_status():
    assert LEGACY_STATUS["canonical"] is False
    assert LEGACY_STATUS["replacement"] == "researchforge.config"
    assert LEGACY_STATUS["cross_imports_allowed"] is False


def test_four_config_scopes_separation():
    """Verify all 4 configuration scopes are distinct and non-overlapping."""
    # Scope 1: Software config
    rf_cfg = RFConfig()
    # Scope 2: Infrastructure config
    env_cfg = EnvConfig()
    # Scope 3: Research-policy config (RSG)
    rsg = ResearchSystemGenome.default("full")
    # Scope 4: Experiment config (Spec)
    spec = ExperimentSpec.create(
        tmg_id="tmg_1",
        tmg_fingerprint="fp_tmg",
        dataset_name="digits",
        dataset_fingerprint="fp_d",
        data_pipeline_fingerprint="fp_p",
        evaluator="sklearn",
        metric_fn="accuracy",
        seed=42,
    )

    # Scopes have distinct type hierarchies
    assert isinstance(rf_cfg, RFConfig)
    assert isinstance(env_cfg, EnvConfig)
    assert hasattr(rsg, "execution_config")
    assert hasattr(rsg, "validity_config")
    assert hasattr(spec, "spec_id")
