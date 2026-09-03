"""tests/test_no_cross_stack_imports.py — Architectural Boundary Invariant test.

Classification: CORE
Enforces that no file under the canonical `researchforge/` namespace ever imports
from the legacy top-level modules (agents, ecrm, evolution, rdg, policy, failure,
tools, config, benchmarks).
"""
from __future__ import annotations

import ast
from pathlib import Path
import pytest

LEGACY_NAMESPACES = frozenset([
    "agents",
    "ecrm",
    "evolution",
    "rdg",
    "policy",
    "failure",
    "tools",
    "config",
    "benchmarks",
])

# Find project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CANONICAL_ROOT = PROJECT_ROOT / "researchforge"


def test_no_legacy_imports_in_canonical_namespace():
    """No file under researchforge/ may import from legacy top-level modules."""
    violations = []
    for py_file in CANONICAL_ROOT.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    continue  # relative import within researchforge is allowed
                module = node.module
            elif isinstance(node, ast.Import):
                module = ".".join(alias.name for alias in node.names)
            else:
                continue

            if module and module.split(".")[0] in LEGACY_NAMESPACES:
                rel_path = py_file.relative_to(PROJECT_ROOT)
                violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')} imports {module!r}")

    assert not violations, (
        f"Cross-stack boundary violations found ({len(violations)}):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
