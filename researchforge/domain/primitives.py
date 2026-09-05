"""Shared primitive type aliases used by domain contracts."""
from typing import NewType

EntityId = NewType("EntityId", str)
Version = NewType("Version", str)
Timestamp = NewType("Timestamp", str)
Confidence = NewType("Confidence", float)
Metric = NewType("Metric", float)
ArtifactRef = NewType("ArtifactRef", str)
