"""Pure views over a discovered probability tree."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from itertools import islice

from .tree import Edge, NodeId, RankedDistribution, Tree


@dataclass(frozen=True, slots=True)
class View:
    """A forest view: children of ``node`` starting at ``first_rank``."""

    node: NodeId = 0
    first_rank: int = 0
    selected_rank: int = 0


@dataclass(frozen=True, slots=True)
class Row:
    """One ephemeral render row derived from the tree."""

    parent: NodeId
    rank: int
    depth: int
    ancestor_last: tuple[bool, ...]
    is_last: bool
    token: int
    edge_nats: float
    path_nats: float
    child: NodeId | None
    expanded: bool


class BranchRole(Enum):
    """How one physical radix row leaves its branch point."""

    CONTINUATION = auto()
    SIBLING = auto()


@dataclass(frozen=True, slots=True)
class AncestorConnector:
    """One visible radix ancestor connector and its exact aggregate surprisal."""

    is_last: bool
    nats: float


@dataclass(frozen=True, slots=True)
class CompactRow:
    """One physical row for one event in the visible probability partition.

    ``edges`` is the structural radix suffix not already factored by earlier
    rows. Token ids and edge costs are derived from those edges and the tree.
    ``path_nats`` is the exact event surprisal from the view root. ``edge_nats``
    is the aggregate surprisal of the displayed branch from the view root, used
    only to shade its connector. ``ancestors`` keeps each visible ancestor
    connector's shape and aggregate surprisal as one value. ``branch_role``
    distinguishes the partition's continuing spine from an ordinary sibling;
    screen coordinates never determine that semantic role.
    """

    parent: NodeId
    rank: int
    depth: int
    ancestors: tuple[AncestorConnector, ...]
    is_last: bool
    edges: tuple[Edge, ...]
    edge_nats: float
    path_nats: float
    child: NodeId | None
    open_ended: bool = False
    forest_count: int = 0
    forest_start: int = 0
    branch_role: BranchRole = BranchRole.SIBLING

    @property
    def ancestor_last(self) -> tuple[bool, ...]:
        return tuple(connector.is_last for connector in self.ancestors)

    @property
    def ancestor_nats(self) -> tuple[float, ...]:
        return tuple(connector.nats for connector in self.ancestors)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(edge.rank for edge in self.edges)

    @property
    def forest(self) -> bool:
        return self.forest_count != 0


def row_tokens(tree: Tree, row: CompactRow) -> tuple[int, ...]:
    """Derive compact-row token ids from its structural edges."""
    return tuple(tree[edge.parent].distribution.token(edge.rank) for edge in row.edges)


@dataclass(frozen=True, slots=True)
class _ConcreteEvent:
    """One concrete cylinder in a display partition."""

    edges: tuple[Edge, ...]
    nats: float
    node: NodeId | None

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(edge.rank for edge in self.edges)


@dataclass(frozen=True, slots=True)
class _ForestEvent:
    """One aggregate sibling range in a display partition."""

    edges: tuple[Edge, ...]
    nats: float
    parent: NodeId
    start: int
    end: int
    base_nats: float

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(edge.rank for edge in self.edges)


type _PartitionEvent = _ConcreteEvent | _ForestEvent


type _SuffixMassCache = dict[NodeId, tuple[float, ...]]


def iter_rows(tree: Tree, view: View):
    """Yield the revealed discovered tree in DFS order without mutating it."""

    def visit(
        parent_id: NodeId,
        first_rank: int,
        depth: int,
        ancestor_last: tuple[bool, ...],
        parent_nats: float,
    ):
        distribution = tree[parent_id].distribution

        for rank in range(first_rank, distribution.revealed):
            child_id = tree.child(parent_id, rank)
            is_last = rank == len(distribution) - 1
            edge_nats = distribution.nats(rank)
            path_nats = parent_nats + edge_nats
            yield Row(
                parent=parent_id,
                rank=rank,
                depth=depth,
                ancestor_last=ancestor_last,
                is_last=is_last,
                token=distribution.token(rank),
                edge_nats=edge_nats,
                path_nats=path_nats,
                child=child_id,
                expanded=child_id is not None,
            )
            if child_id is not None:
                yield from visit(
                    child_id,
                    0,
                    depth + 1,
                    (*ancestor_last, is_last),
                    path_nats,
                )

    return visit(view.node, view.first_rank, 0, (), 0.0)


def rows(tree: Tree, view: View, limit: int) -> tuple[Row, ...]:
    """Return at most ``limit`` semantic rows for one frame."""
    return tuple(islice(iter_rows(tree, view), limit))


def forest_nats(
    distribution: RankedDistribution,
    start: int,
    end: int | None = None,
    *,
    parent_nats: float = 0.0,
) -> float:
    """Return surprisal of the exact aggregate sibling event ``[start, end)``."""
    stop = len(distribution) if end is None else end
    if not 0 <= start <= stop <= len(distribution):
        raise IndexError((start, stop))
    mass = distribution.mass(start, stop)
    return parent_nats - math.log(mass) if mass > 0.0 else math.inf


def _suffix_mass(
    tree: Tree,
    parent: NodeId,
    start: int,
    cache: _SuffixMassCache,
) -> float:
    """Return ``P(rank >= start)`` from one backward-built revealed suffix table."""
    distribution = tree[parent].distribution
    if not 0 <= start <= distribution.revealed:
        raise IndexError(f"suffix start {start} has not been revealed")

    suffix = cache.get(parent)
    if suffix is None:
        revealed = distribution.revealed
        masses = [0.0] * (revealed + 1)
        masses[revealed] = distribution.mass(revealed, len(distribution))
        for rank in range(revealed - 1, -1, -1):
            masses[rank] = math.fsum((distribution.probability(rank), masses[rank + 1]))
        suffix = tuple(masses)
        cache[parent] = suffix
    return suffix[start]


def partition_rows(
    tree: Tree,
    view: View,
    *,
    row_limit: int,
    first_rank: int | None = None,
) -> tuple[CompactRow, ...]:
    """Return a best-first probability partition in a leaf-only radix layout.

    Every physical row is exactly one disjoint continuation event. Starting
    from the whole visible sibling tail, each additional row refines one event
    into two disjoint events. Candidate refinements compete globally by the
    probability of their smaller result.

    A progressive replica may know only a ranked prefix concretely. Its
    unrevealed suffix remains one exact aggregate forest event; rendering never
    asks the engine to reveal it.
    """
    if row_limit <= 0:
        return ()

    root = view.node
    distribution = tree[root].distribution
    start = view.first_rank if first_rank is None else first_rank
    if not 0 <= start <= len(distribution):
        raise IndexError(start)
    if start == len(distribution):
        return ()

    initial_nats = 0.0 if start == 0 else forest_nats(distribution, start)
    suffix_masses: _SuffixMassCache = {}
    events = [
        _partition_range(
            tree,
            parent=root,
            start=start,
            end=len(distribution),
            prefix_edges=(),
            base_nats=0.0,
            suffix_masses=suffix_masses,
            range_nats=initial_nats,
        )
    ]

    while len(events) < row_limit:
        candidates: list[
            tuple[float, float, tuple[int, ...], int, tuple[_PartitionEvent, ...]]
        ] = []
        for index, event in enumerate(events):
            split = _partition_split(tree, event, suffix_masses)
            if split is None:
                continue
            smaller_piece_nats = max(part.nats for part in split)
            candidates.append(
                (
                    smaller_piece_nats,
                    event.nats,
                    _partition_order(event),
                    index,
                    split,
                )
            )

        if not candidates:
            break

        *_priority, index, split = min(candidates, key=lambda candidate: candidate[:-1])
        events[index : index + 1] = split

    events.sort(key=_partition_order)
    return _partition_layout_rows(
        tree,
        root,
        events,
        root_continuation=start == view.first_rank,
    )


def _partition_branch(
    tree: Tree,
    *,
    parent: NodeId,
    rank: int,
    prefix_edges: tuple[Edge, ...],
    base_nats: float,
) -> _ConcreteEvent:
    distribution = tree[parent].distribution
    if not 0 <= rank < distribution.revealed:
        raise IndexError(f"rank {rank} has not been revealed")
    return _ConcreteEvent(
        edges=(*prefix_edges, Edge(parent, rank)),
        nats=base_nats + distribution.nats(rank),
        node=tree.child(parent, rank),
    )


def _partition_range(
    tree: Tree,
    *,
    parent: NodeId,
    start: int,
    end: int,
    prefix_edges: tuple[Edge, ...],
    base_nats: float,
    suffix_masses: _SuffixMassCache,
    range_nats: float | None = None,
) -> _PartitionEvent:
    distribution = tree[parent].distribution
    if not 0 <= start < end <= len(distribution):
        raise IndexError((start, end))
    if end - start == 1:
        return _partition_branch(
            tree,
            parent=parent,
            rank=start,
            prefix_edges=prefix_edges,
            base_nats=base_nats,
        )
    if range_nats is None:
        if end == len(distribution) and start <= distribution.revealed:
            mass = _suffix_mass(tree, parent, start, suffix_masses)
            range_nats = base_nats - math.log(mass) if mass > 0.0 else math.inf
        else:
            range_nats = forest_nats(
                distribution,
                start,
                end,
                parent_nats=base_nats,
            )
    return _ForestEvent(
        edges=prefix_edges,
        nats=range_nats,
        parent=parent,
        start=start,
        end=end,
        base_nats=base_nats,
    )


def _partition_split(
    tree: Tree,
    event: _PartitionEvent,
    suffix_masses: _SuffixMassCache,
) -> tuple[_PartitionEvent, _PartitionEvent] | None:
    if isinstance(event, _ForestEvent):
        distribution = tree[event.parent].distribution
        if event.start >= distribution.revealed:
            return None

        first = _partition_branch(
            tree,
            parent=event.parent,
            rank=event.start,
            prefix_edges=event.edges,
            base_nats=event.base_nats,
        )
        rest = _partition_range(
            tree,
            parent=event.parent,
            start=event.start + 1,
            end=event.end,
            prefix_edges=event.edges,
            base_nats=event.base_nats,
            suffix_masses=suffix_masses,
        )
        return first, rest

    child = event.node
    if child is None:
        return None
    distribution = tree[child].distribution
    if len(distribution) < 2 or distribution.revealed == 0:
        return None

    first = _partition_branch(
        tree,
        parent=child,
        rank=0,
        prefix_edges=event.edges,
        base_nats=event.nats,
    )
    rest = _partition_range(
        tree,
        parent=child,
        start=1,
        end=len(distribution),
        prefix_edges=event.edges,
        base_nats=event.nats,
        suffix_masses=suffix_masses,
    )
    return first, rest


def _partition_order(event: _PartitionEvent) -> tuple[int, ...]:
    if isinstance(event, _ForestEvent):
        return (*event.ranks, event.start)
    return event.ranks


def _common_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    length = 0
    for left_rank, right_rank in zip(left, right, strict=False):
        if left_rank != right_rank:
            break
        length += 1
    return length


def _partition_child_key(event: _PartitionEvent, depth: int) -> int | None:
    if depth < len(event.ranks):
        return event.ranks[depth]
    return None


def _partition_children(
    events: list[_PartitionEvent],
) -> dict[tuple[int, ...], tuple[int | None, ...]]:
    children: dict[tuple[int, ...], list[int | None]] = {}
    for event in events:
        for depth in range(len(event.ranks) + 1):
            prefix = event.ranks[:depth]
            child = _partition_child_key(event, depth)
            siblings = children.setdefault(prefix, [])
            if child not in siblings:
                siblings.append(child)
    return {prefix: tuple(siblings) for prefix, siblings in children.items()}


def _partition_edge_nats(
    events: list[_PartitionEvent],
) -> dict[tuple[tuple[int, ...], int | None], float]:
    masses: dict[tuple[tuple[int, ...], int | None], list[float]] = {}
    for event in events:
        mass = math.exp(-event.nats)
        for depth in range(len(event.ranks) + 1):
            prefix = event.ranks[:depth]
            child = _partition_child_key(event, depth)
            masses.setdefault((prefix, child), []).append(mass)

    result: dict[tuple[tuple[int, ...], int | None], float] = {}
    for edge, edge_masses in masses.items():
        total = math.fsum(edge_masses)
        result[edge] = -math.log(total) if total > 0.0 else math.inf
    return result


def _partition_node(tree: Tree, root: NodeId, ranks: tuple[int, ...]) -> NodeId:
    node = root
    for rank in ranks:
        child = tree.child(node, rank)
        if child is None:
            raise ValueError("partition prefix crosses an undiscovered edge")
        node = child
    return node


def _partition_layout_rows(
    tree: Tree,
    root: NodeId,
    events: list[_PartitionEvent],
    *,
    root_continuation: bool,
) -> tuple[CompactRow, ...]:
    children = _partition_children(events)
    edge_nats = _partition_edge_nats(events)
    branch_prefixes = {prefix for prefix, siblings in children.items() if len(siblings) > 1}
    represented_roots: set[int] = set()
    rows_out: list[CompactRow] = []
    previous: _PartitionEvent | None = None

    for event in events:
        common = 0 if previous is None else _common_prefix(previous.ranks, event.ranks)
        common_ranks = event.ranks[:common]
        parent = _partition_node(tree, root, common_ranks)

        ancestor_prefixes = tuple(
            event.ranks[:depth] for depth in range(common) if event.ranks[:depth] in branch_prefixes
        )
        ancestors = tuple(
            AncestorConnector(
                is_last=event.ranks[len(prefix)] == children[prefix][-1],
                nats=edge_nats[(prefix, event.ranks[len(prefix)])],
            )
            for prefix in ancestor_prefixes
        )

        branch_prefix = common_ranks
        branch_child = _partition_child_key(event, common)
        branch_nats = edge_nats[(branch_prefix, branch_child)]
        is_last = branch_child == children[branch_prefix][-1]
        branch_role = (
            BranchRole.CONTINUATION
            if previous is None and root_continuation
            else BranchRole.SIBLING
        )

        if event.ranks:
            root_rank = event.ranks[0]
        elif isinstance(event, _ForestEvent):
            root_rank = event.start
        else:
            raise AssertionError("concrete partition event has no edge")

        representative_rank = -1
        if isinstance(event, _ConcreteEvent) and root_rank not in represented_roots:
            representative_rank = root_rank
            represented_roots.add(root_rank)

        if isinstance(event, _ForestEvent):
            forest_count = event.end - event.start
            forest_start = event.start
            open_ended = False
            child = None
        else:
            forest_count = 0
            forest_start = 0
            child = event.node
            open_ended = child is None or len(tree[child].distribution) != 0

        rows_out.append(
            CompactRow(
                parent=parent,
                rank=representative_rank,
                depth=len(ancestors),
                ancestors=ancestors,
                is_last=is_last,
                edges=event.edges[common:],
                edge_nats=branch_nats,
                path_nats=event.nats,
                child=child,
                open_ended=open_ended,
                forest_count=forest_count,
                forest_start=forest_start,
                branch_role=branch_role,
            )
        )
        previous = event

    return tuple(rows_out)


def enter(tree: Tree, view: View) -> View:
    """Focus the selected child if it has already been discovered."""
    child = tree.child(view.node, view.selected_rank)
    if child is None:
        raise ValueError("cannot enter an undiscovered child")
    return View(node=child)


def parent(tree: Tree, view: View) -> View:
    """Focus the parent and preserve the rank by which it was entered."""
    node = tree[view.node]
    if node.parent is None:
        return view
    return View(node=node.parent, first_rank=node.rank, selected_rank=node.rank)


def move(tree: Tree, view: View, delta: int) -> View:
    """Move selection within the currently revealed sibling prefix."""
    distribution = tree[view.node].distribution
    if distribution.revealed == 0:
        return view
    last = distribution.revealed - 1
    selected = min(max(view.selected_rank + delta, view.first_rank), last)
    return View(node=view.node, first_rank=view.first_rank, selected_rank=selected)
