from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    SIGNAL = "signal"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    EXCLUSION = "exclusion"
    FAILURE = "failure"


class EdgeType(str, Enum):
    TRIGGERS = "triggers"
    CAUSES = "causes"
    EXCLUDES = "excludes"
    RESTRICTS = "restricts"
    SUPPORTS = "supports"
    INFORMS = "informs"


@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    label: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            label=data["label"],
            data=data.get("data", {}),
        )


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphEdge:
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            confidence=data.get("confidence", 1.0),
            evidence=data.get("evidence", {}),
        )


@dataclass
class ExplanationGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def edges_from(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def edges_to(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def predecessors(self, node_id: str) -> list[str]:
        return [e.source_id for e in self.edges_to(node_id)]

    def successors(self, node_id: str) -> list[str]:
        return [e.target_id for e in self.edges_from(node_id)]

    def trace_back(
        self, node_id: str, max_depth: int = 10
    ) -> list[list[tuple[str, str]]]:
        """BFS backward from node_id through incoming edges.

        Returns list of paths, where each path is a list of
        (node_id, edge_type_value) tuples from root to target.
        """
        if node_id not in self.nodes:
            return []

        # BFS queue: (current_node_id, path_so_far)
        # path_so_far = [(node_id, edge_type_that_led_here_or_empty_string)]
        paths: list[list[tuple[str, str]]] = []
        queue: list[tuple[str, list[tuple[str, str]]]] = [
            (node_id, [(node_id, "")])
        ]

        while queue:
            current_id, path = queue.pop(0)
            if len(path) > max_depth + 1:
                # reached depth limit — record this as a path
                paths.append(list(reversed(path)))
                continue

            incoming = self.edges_to(current_id)
            if not incoming:
                # root node — record path
                paths.append(list(reversed(path)))
                continue

            found_new = False
            for edge in incoming:
                if edge.source_id in [p[0] for p in path]:
                    continue  # avoid cycles
                new_path = path + [(edge.source_id, edge.edge_type.value)]
                queue.append((edge.source_id, new_path))
                found_new = True

            if not found_new:
                # all predecessors already in path (cycle) — record path as-is
                paths.append(list(reversed(path)))

        return paths

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExplanationGraph:
        graph = cls()
        for nid, ndata in data.get("nodes", {}).items():
            graph.add_node(GraphNode.from_dict(ndata))
        for edata in data.get("edges", []):
            graph.add_edge(GraphEdge.from_dict(edata))
        return graph
