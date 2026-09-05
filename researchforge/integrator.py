from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from researchforge.state.transition_engine import ResearchStateTransitionEngine
from researchforge.state.events import Event
from researchforge.domain.provenance import Provenance
from researchforge.vrdeg.graph import VRDEG, GraphIntegrityError
from researchforge.vrdeg.projector import VRDEGProjector, _event_edge_map
from researchforge.vrdeg.node import GraphNode, NodeType
from researchforge.vrdeg.edge import GraphEdge
from researchforge.vrdeg.edge import RelationType
from researchforge.vrdeg.graph import GraphIntegrityError as GIError


@dataclass
class IntegrationReport:
    num_events: int
    num_states: int
    num_nodes: int
    num_edges: int
    provenance_records: int
    branches: int
    failures: int
    outcomes: int
    unresolved_references: List[str]
    consistency: bool
    details: Dict[str, Any]


class IntegrationConsistencyError(Exception):
    pass


class ResearchHistoryService:
    """Integrates ResearchStateTransitionEngine with VRDEGProjector and VRDEG.

    Responsibilities:
    - Apply events through the TransitionEngine (single source of truth)
    - Project validated events into VRDEG via VRDEGProjector
    - Provide reconstruction and consistency validation helpers
    """

    def __init__(self, engine: Optional[ResearchStateTransitionEngine] = None, graph: Optional[VRDEG] = None, projector: Optional[VRDEGProjector] = None) -> None:
        self.engine = engine or ResearchStateTransitionEngine()
        self.graph = graph or VRDEG()
        self.projector = projector or VRDEGProjector(self.graph)
        self.states: List = []
        self.last_events: List[Event] = []

    def apply_events(self, initial_state, events: List[Event], provenance_map: Dict[str, Provenance] | None = None):
        state = initial_state
        self.states = [state]
        self.last_events = list(events)
        for ev in events:
            prov_obj = None
            if provenance_map and ev.provenance_id:
                prov_obj = provenance_map.get(ev.provenance_id)
            # apply transition via engine (validates)
            state = self.engine.transition(state, ev, provenance=prov_obj)
            # if provenance object provided, ensure provenance node with metadata exists
            if prov_obj:
                try:
                    pn = GraphNode(id=prov_obj.id, schema_version=prov_obj.schema_version, node_type=NodeType.PROVENANCE.value, metadata=prov_obj.to_dict())
                    self.graph.add_node(pn)
                except GraphIntegrityError:
                    pass
            # project validated event into graph
            self.projector.project_event(ev)
            self.states.append(state)
        return state

    def project_events(self, events: List[Event]):
        self.projector.project_events(events)

    def reconstruct(self, initial_state, events: List[Event], provenance_map: Dict[str, Provenance] | None = None):
        # create fresh graph and projector for a clean reconstruction
        g = VRDEG()
        p = VRDEGProjector(g)
        engine = ResearchStateTransitionEngine()
        svc = ResearchHistoryService(engine=engine, graph=g, projector=p)
        final_state = svc.apply_events(initial_state, events, provenance_map=provenance_map)
        return final_state, g

    def validate_consistency(self) -> IntegrationReport:
        unresolved = []
        # check for each event/state in memory that payload refs exist in graph
        for st in self.states:
            if getattr(st, "provenance_id", None):
                if not self.graph.get_node(st.provenance_id):
                    unresolved.append(f"state_prov:{st.id}:{st.provenance_id}")
        # examine graph for missing provenance references
        try:
            self.graph.validate_integrity()
        except GIError as e:
            unresolved.append(str(e))

        # check event-edge mappings for events we processed (requires projector mapping)
        missing = []
        for ev in getattr(self, "last_events", []):
            mapping = _event_edge_map.get(ev.event_type, [])
            for src_key, tgt_key, relation in mapping:
                src = (ev.payload or {}).get(src_key)
                tgt = (ev.payload or {}).get(tgt_key)
                if not src or not tgt:
                    continue
                edge_id = f"edge:{ev.id}:{relation}:{src}->{tgt}"
                if not self.graph.get_edge(edge_id):
                    missing.append(edge_id)
        if missing:
            unresolved.extend(missing)

        report = IntegrationReport(
            num_events=len(getattr(self, "last_events", [])),
            num_states=len(self.states),
            num_nodes=len(self.graph._nodes),
            num_edges=len(self.graph._edges),
            provenance_records=len([n for n in self.graph._nodes.values() if n.node_type == NodeType.PROVENANCE.value]),
            branches=0,
            failures=len([n for n in self.graph._nodes.values() if n.node_type == NodeType.FAILURE.value]),
            outcomes=len([n for n in self.graph._nodes.values() if n.node_type == NodeType.OUTCOME.value]),
            unresolved_references=unresolved,
            consistency=(len(unresolved) == 0),
            details={},
        )
        return report
