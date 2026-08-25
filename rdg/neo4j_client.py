"""
Neo4j client for RDG persistence.
Falls back gracefully when Neo4j is unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Thin wrapper around Neo4j Python driver for RDG sync."""

    def __init__(self, uri: str, user: str, password: str):
        self._driver = None
        try:
            from neo4j import GraphDatabase  # type: ignore

            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Neo4j connected at %s", uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j unavailable: %s – running in-memory only.", exc)

    @property
    def available(self) -> bool:
        return self._driver is not None

    def _run(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(r) for r in result]

    # ── RDG Sync ──────────────────────────────────────────────────────────────

    def upsert_node(self, node_dict: Dict[str, Any]) -> None:
        """Create or update an RDG node in Neo4j."""
        ntype = node_dict["type"]
        cypher = (
            f"MERGE (n:{ntype} {{id: $id}}) "
            "SET n += $props"
        )
        props = {k: v for k, v in node_dict.items() if k not in ("type",)}
        self._run(cypher, id=node_dict["id"], props=props)

    def upsert_edge(self, edge_dict: Dict[str, Any]) -> None:
        rel = edge_dict["relation"].upper().replace("-", "_")
        cypher = (
            f"MATCH (a {{id: $from_id}}), (b {{id: $to_id}}) "
            f"MERGE (a)-[r:{rel} {{id: $eid}}]->(b) "
            f"SET r.confidence = $conf, r.timestamp = $ts"
        )
        self._run(
            cypher,
            from_id=edge_dict["from"],
            to_id=edge_dict["to"],
            eid=edge_dict["id"],
            conf=edge_dict.get("confidence", 1.0),
            ts=edge_dict.get("timestamp", ""),
        )

    def query_hypothesis_experiments(self) -> List[Dict]:
        return self._run(
            "MATCH (h:Hypothesis)-[:TESTED_BY]->(e:Experiment) "
            "RETURN h.id AS hypothesis_id, e.id AS experiment_id, e.content AS experiment"
        )

    def query_best_outcomes(self, hypothesis_id: str) -> List[Dict]:
        return self._run(
            "MATCH (h:Hypothesis {id:$hid})-[:TESTED_BY]->(e)-[:PRODUCES]->(f:Finding) "
            "RETURN f ORDER BY f.score DESC LIMIT 5",
            hid=hypothesis_id,
        )

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def __del__(self) -> None:
        self.close()
