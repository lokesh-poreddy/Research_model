"""researchforge/config/rf_config.py — ResearchForge software configuration.

Scope 1: Software / runtime environment parameters:
  - version string, log level, debug flag, telemetry
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RFConfig:
    """Canonical software-level configuration for ResearchForge."""
    version: str = "RF-1.0.0-alpha.2.1"
    debug: bool = False
    log_level: str = "INFO"
    strict_schema_validation: bool = True
    track_provenance: bool = True

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "debug": self.debug,
            "log_level": self.log_level,
            "strict_schema_validation": self.strict_schema_validation,
            "track_provenance": self.track_provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RFConfig":
        return cls(
            version=d.get("version", "RF-1.0.0-alpha.2.1"),
            debug=bool(d.get("debug", False)),
            log_level=d.get("log_level", "INFO"),
            strict_schema_validation=bool(d.get("strict_schema_validation", True)),
            track_provenance=bool(d.get("track_provenance", True)),
        )


# Global default software config instance
default_rf_config = RFConfig()
