"""JSON Schema definitions for Research Development Graph (RDG) nodes and
edges, matching ResearchForge-ECRM Sec. 3.1 (RDG JSON Schema) and Sec. 3.2 of
the technical report (Neo4j node labels / relationship types), adapted for
runtime validation with the `jsonschema` package.
"""
from __future__ import annotations
from typing import Any, Dict

# Node labels, merged from both source documents (executive summary Sec 3.2
# and technical report Sec 3.1): the six core RDG objects plus the four
# supporting node types (ModelGenome, Strategy, MemoryRecord, Insight).
NODE_TYPES = [
    "Problem", "Gap", "Hypothesis", "Experiment", "Finding", "Claim",
    "ModelGenome", "Strategy", "MemoryRecord", "Insight",
]

# Edge relations, merged the same way (IDENTIFIES/MOTIVATES/TESTED_BY/
# PRODUCES/SUPPORTS/DERIVES_FROM/SAVED_AS from the exec summary, plus
# EVALUATED_AS/INFORMS/UPDATES from the technical report's ontology).
EDGE_RELATIONS = [
    "identifies", "motivates", "tested-by", "produces", "supports",
    "derives-from", "saved-as", "evaluated-as", "informs", "updates",
]

RDG_NODE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "RDG Node",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string", "enum": NODE_TYPES},
        "content": {"type": "string"},
        "timestamp": {"type": "string"},
        "attributes": {"type": "object"},
    },
    "required": ["id", "type", "content"],
}

RDG_EDGE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "RDG Edge",
    "type": "object",
    "properties": {
        "from_id": {"type": "string"},
        "to_id": {"type": "string"},
        "relation": {"type": "string", "enum": EDGE_RELATIONS},
        "properties": {"type": "object"},
    },
    "required": ["from_id", "to_id", "relation"],
}

# Typed-edge constraints: relation -> allowed {(from_type, to_type), ...}.
# Enforces the design doc's rule that "an Experiment must test its parent
# Hypothesis" etc. Checked in rdg.graph.ResearchDevelopmentGraph.add_edge().
EDGE_TYPE_CONSTRAINTS = {
    "identifies": {("Problem", "Gap")},
    "motivates": {("Gap", "Hypothesis")},
    "tested-by": {("Hypothesis", "Experiment")},
    "produces": {("Experiment", "Finding")},
    "supports": {("Finding", "Claim")},
    "derives-from": {("ModelGenome", "ModelGenome")},
    "saved-as": {("Hypothesis", "MemoryRecord"), ("Experiment", "MemoryRecord")},
    "evaluated-as": {("ModelGenome", "Finding"), ("Experiment", "Finding")},
    "informs": {("Finding", "Insight"), ("Claim", "Insight")},
    "updates": {("Insight", "Hypothesis")},
}


def validate_node(node: Dict[str, Any]) -> None:
    import jsonschema
    jsonschema.validate(node, RDG_NODE_SCHEMA)


def validate_edge(edge: Dict[str, Any]) -> None:
    import jsonschema
    jsonschema.validate(edge, RDG_EDGE_SCHEMA)
