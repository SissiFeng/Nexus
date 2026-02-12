"""Causal graph data structures for the Causal Discovery Engine.

Provides directed acyclic graph (DAG) primitives for representing causal
relationships between variables, with support for d-separation queries,
topological ordering, and serialization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CausalNode:
    """A node in a causal graph.

    Parameters
    ----------
    name : str
        Unique identifier for the node.
    node_type : str
        One of ``"observed"``, ``"latent"``, or ``"intervention"``.
    metadata : dict[str, Any]
        Arbitrary metadata associated with this node.
    """

    name: str
    node_type: str = "observed"  # "observed" | "latent" | "intervention"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node_type": self.node_type,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalNode:
        return cls(
            name=data["name"],
            node_type=data.get("node_type", "observed"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CausalEdge:
    """A directed edge in a causal graph.

    Parameters
    ----------
    source : str
        Name of the parent node.
    target : str
        Name of the child node.
    edge_type : str
        One of ``"causal"``, ``"confounded"``, or ``"instrumental"``.
    strength : float | None
        Estimated causal strength (e.g. partial correlation).
    """

    source: str
    target: str
    edge_type: str = "causal"  # "causal" | "confounded" | "instrumental"
    strength: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "strength": self.strength,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalEdge:
        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=data.get("edge_type", "causal"),
            strength=data.get("strength"),
        )


class CausalGraph:
    """Directed acyclic graph representing causal relationships.

    Stores nodes by name and adjacency lists for directed edges.
    Supports d-separation queries, topological sorting, and
    serialization to/from plain dicts.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, CausalNode] = {}
        # adjacency: parent -> set of children
        self._children: dict[str, set[str]] = {}
        # reverse adjacency: child -> set of parents
        self._parents: dict[str, set[str]] = {}
        self._edges: dict[tuple[str, str], CausalEdge] = {}

    # -- Node operations --------------------------------------------------------

    def add_node(self, node: CausalNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.name] = node
        self._children.setdefault(node.name, set())
        self._parents.setdefault(node.name, set())

    def get_node(self, name: str) -> CausalNode | None:
        """Return the node with *name*, or ``None``."""
        return self._nodes.get(name)

    @property
    def node_names(self) -> list[str]:
        """Sorted list of all node names."""
        return sorted(self._nodes)

    @property
    def nodes(self) -> dict[str, CausalNode]:
        return dict(self._nodes)

    # -- Edge operations --------------------------------------------------------

    def add_edge(self, edge: CausalEdge) -> None:
        """Add a directed edge from *edge.source* to *edge.target*.

        Both endpoints must already exist as nodes.
        """
        if edge.source not in self._nodes:
            raise ValueError(f"Source node '{edge.source}' not in graph")
        if edge.target not in self._nodes:
            raise ValueError(f"Target node '{edge.target}' not in graph")
        self._children[edge.source].add(edge.target)
        self._parents[edge.target].add(edge.source)
        self._edges[(edge.source, edge.target)] = edge

    def remove_edge(self, source: str, target: str) -> None:
        """Remove the directed edge from *source* to *target*."""
        self._children.get(source, set()).discard(target)
        self._parents.get(target, set()).discard(source)
        self._edges.pop((source, target), None)

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self._edges

    def get_edge(self, source: str, target: str) -> CausalEdge | None:
        return self._edges.get((source, target))

    @property
    def edges(self) -> list[CausalEdge]:
        return list(self._edges.values())

    # -- Graph queries ----------------------------------------------------------

    def parents(self, node: str) -> set[str]:
        """Return the direct parents of *node*."""
        return set(self._parents.get(node, set()))

    def children(self, node: str) -> set[str]:
        """Return the direct children of *node*."""
        return set(self._children.get(node, set()))

    def ancestors(self, node: str) -> set[str]:
        """Return all ancestors of *node* (excluding *node* itself)."""
        visited: set[str] = set()
        queue = deque(self._parents.get(node, set()))
        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                queue.extend(self._parents.get(current, set()) - visited)
        return visited

    def descendants(self, node: str) -> set[str]:
        """Return all descendants of *node* (excluding *node* itself)."""
        visited: set[str] = set()
        queue = deque(self._children.get(node, set()))
        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                queue.extend(self._children.get(current, set()) - visited)
        return visited

    # -- D-separation -----------------------------------------------------------

    def d_separated(self, x: str, y: str, z: set[str]) -> bool:
        """Test if *x* and *y* are d-separated given conditioning set *z*.

        Uses the Bayes-Ball algorithm.  Returns ``True`` when every path
        between *x* and *y* is blocked by *z*.
        """
        z_set = set(z)
        # Nodes that are ancestors of Z (needed for collider activation)
        z_ancestors: set[str] = set()
        for node in z_set:
            z_ancestors |= self.ancestors(node)

        # BFS with direction tracking: (node, came_from_child)
        # came_from_child=True means we arrived at this node from one of its children
        visited: set[tuple[str, bool]] = set()
        queue: deque[tuple[str, bool]] = deque()

        # Start from x going both directions
        for child in self._children.get(x, set()):
            queue.append((child, False))  # going to child
        for parent in self._parents.get(x, set()):
            queue.append((parent, True))  # going to parent

        reachable: set[str] = set()

        while queue:
            current, came_from_child = queue.popleft()
            if (current, came_from_child) in visited:
                continue
            visited.add((current, came_from_child))

            if current == y:
                reachable.add(y)

            # Case 1: arrived from a child (i.e. traversing an edge backwards)
            if came_from_child:
                # If current is NOT in Z: can pass through (chain/fork)
                if current not in z_set:
                    for parent in self._parents.get(current, set()):
                        queue.append((parent, True))
                    for child in self._children.get(current, set()):
                        queue.append((child, False))
                # If current IS in Z and is a collider: blocked (this direction)
                # but we don't continue from here
            else:
                # Case 2: arrived from a parent (i.e. traversing an edge forwards)
                # If current is NOT in Z: can pass through
                if current not in z_set:
                    for child in self._children.get(current, set()):
                        queue.append((child, False))
                # If current IS in Z or has an ancestor in Z: collider is activated
                if current in z_set or current in z_ancestors:
                    # Collider case: can go back to parents
                    for parent in self._parents.get(current, set()):
                        queue.append((parent, True))

        return y not in reachable

    # -- Topological sort -------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Return a topological ordering of the graph nodes.

        Raises ``ValueError`` if the graph contains a cycle.
        """
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for child in self._children.get(node, set()):
                in_degree[child] += 1

        queue = deque(n for n, d in sorted(in_degree.items()) if d == 0)
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for child in sorted(self._children.get(node, set())):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self._nodes):
            raise ValueError("Graph contains a cycle")
        return result

    # -- Serialization ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a plain dict."""
        return {
            "nodes": {name: node.to_dict() for name, node in self._nodes.items()},
            "edges": [edge.to_dict() for edge in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalGraph:
        """Deserialize a graph from a plain dict."""
        graph = cls()
        for _name, ndata in data.get("nodes", {}).items():
            graph.add_node(CausalNode.from_dict(ndata))
        for edata in data.get("edges", []):
            graph.add_edge(CausalEdge.from_dict(edata))
        return graph

    def copy(self) -> CausalGraph:
        """Return a deep copy of the graph."""
        return CausalGraph.from_dict(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"CausalGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
        )
