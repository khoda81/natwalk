"""Pure views over a discovered probability tree."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice

from .tree import Distribution, NodeId, Tree


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


@dataclass(frozen=True, slots=True)
class CompactRow:
    """One physical row for one event in the visible probability partition.

    ``tokens`` is the radix suffix not already factored by earlier rows.
    ``path_nats`` is the exact event surprisal from the view root. ``edge_nats``
    is the aggregate surprisal of the displayed branch from the view root, used
    only to shade its connector. ``ancestor_nats`` gives the corresponding
    aggregate surprisal for every visible ancestor connector.
    """

    parent: NodeId
    rank: int
    depth: int
    ancestor_last: tuple[bool, ...]
    is_last: bool
    tokens: tuple[int, ...]
    edge_nats: float
    path_nats: float
    child: NodeId | None
    open_ended: bool = False
    forest_count: int = 0
    ancestor_nats: tuple[float, ...] = ()

    @property
    def forest(self) -> bool:
        return self.forest_count != 0


@dataclass(frozen=True, slots=True)
class _PartitionEvent:
    """One disjoint cylinder or sibling forest in a display partition."""

    ranks: tuple[int, ...]
    tokens: tuple[int, ...]
    nats: float
    node: NodeId | None = None
    forest_parent: NodeId | None = None
    forest_start: int = 0
    forest_end: int = 0
    forest_base_nats: float = 0.0

    @property
    def forest(self) -> bool:
        return self.forest_parent is not None


def iter_rows(tree: Tree, view: View):
    """Yield the view's discovered tree in DFS order without mutating it."""

    def visit(
        parent_id: NodeId,
        first_rank: int,
        depth: int,
        ancestor_last: tuple[bool, ...],
        parent_nats: float,
    ):
        distribution = tree[parent_id].distribution

        for rank in range(first_rank, len(distribution)):
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
                token=distribution.tokens[rank],
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
    distribution: Distribution,
    start: int,
    end: int | None = None,
    *,
    parent_nats: float = 0.0,
) -> float:
    """Return surprisal of the aggregate sibling event ``[start, end)``."""
    stop = len(distribution) if end is None else end
    if not 0 <= start <= stop <= len(distribution):
        raise IndexError((start, stop))
    mass = math.fsum(distribution.probabilities[start:stop])
    return parent_nats - math.log(mass) if mass > 0.0 else math.inf


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

    Once the event set is fixed, layout is a separate operation: shared token
    prefixes are factored like a radix trie, but internal trie nodes never get
    rows of their own. They only contribute indentation and connector state to
    the already-selected leaf rows.
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

    events = [
        _partition_range(
            tree,
            parent=root,
            start=start,
            end=len(distribution),
            prefix_ranks=(),
            prefix_tokens=(),
            base_nats=0.0,
        )
    ]

    while len(events) < row_limit:
        candidates: list[
            tuple[float, float, tuple[int, ...], int, tuple[_PartitionEvent, ...]]
        ] = []
        for index, event in enumerate(events):
            split = _partition_split(tree, event)
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
    return _partition_layout_rows(tree, root, events)


def _partition_branch(
    tree: Tree,
    *,
    parent: NodeId,
    rank: int,
    prefix_ranks: tuple[int, ...],
    prefix_tokens: tuple[int, ...],
    base_nats: float,
) -> _PartitionEvent:
    distribution = tree[parent].distribution
    return _PartitionEvent(
        ranks=(*prefix_ranks, rank),
        tokens=(*prefix_tokens, distribution.tokens[rank]),
        nats=base_nats + distribution.nats(rank),
        node=tree.child(parent, rank),
    )


def _partition_range(
    tree: Tree,
    *,
    parent: NodeId,
    start: int,
    end: int,
    prefix_ranks: tuple[int, ...],
    prefix_tokens: tuple[int, ...],
    base_nats: float,
) -> _PartitionEvent:
    if not 0 <= start < end <= len(tree[parent].distribution):
        raise IndexError((start, end))
    if end - start == 1:
        return _partition_branch(
            tree,
            parent=parent,
            rank=start,
            prefix_ranks=prefix_ranks,
            prefix_tokens=prefix_tokens,
            base_nats=base_nats,
        )
    return _PartitionEvent(
        ranks=prefix_ranks,
        tokens=prefix_tokens,
        nats=forest_nats(
            tree[parent].distribution,
            start,
            end,
            parent_nats=base_nats,
        ),
        forest_parent=parent,
        forest_start=start,
        forest_end=end,
        forest_base_nats=base_nats,
    )


def _partition_split(
    tree: Tree,
    event: _PartitionEvent,
) -> tuple[_PartitionEvent, _PartitionEvent] | None:
    if event.forest:
        parent = event.forest_parent
        if parent is None:
            raise AssertionError("forest event has no parent")
        if event.forest_end - event.forest_start < 2:
            raise AssertionError("single-rank ranges must be concrete events")

        first = _partition_branch(
            tree,
            parent=parent,
            rank=event.forest_start,
            prefix_ranks=event.ranks,
            prefix_tokens=event.tokens,
            base_nats=event.forest_base_nats,
        )
        rest = _partition_range(
            tree,
            parent=parent,
            start=event.forest_start + 1,
            end=event.forest_end,
            prefix_ranks=event.ranks,
            prefix_tokens=event.tokens,
            base_nats=event.forest_base_nats,
        )
        return first, rest

    child = event.node
    if child is None:
        return None
    distribution = tree[child].distribution
    if len(distribution) < 2:
        return None

    first = _partition_branch(
        tree,
        parent=child,
        rank=0,
        prefix_ranks=event.ranks,
        prefix_tokens=event.tokens,
        base_nats=event.nats,
    )
    rest = _partition_range(
        tree,
        parent=child,
        start=1,
        end=len(distribution),
        prefix_ranks=event.ranks,
        prefix_tokens=event.tokens,
        base_nats=event.nats,
    )
    return first, rest


def _partition_order(event: _PartitionEvent) -> tuple[int, ...]:
    if event.forest:
        return (*event.ranks, event.forest_start)
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
            event.ranks[:depth]
            for depth in range(common)
            if event.ranks[:depth] in branch_prefixes
        )
        ancestor_last = tuple(
            event.ranks[len(prefix)] == children[prefix][-1]
            for prefix in ancestor_prefixes
        )
        ancestor_nats = tuple(
            edge_nats[(prefix, event.ranks[len(prefix)])] for prefix in ancestor_prefixes
        )

        branch_prefix = common_ranks
        branch_child = _partition_child_key(event, common)
        branch_nats = edge_nats[(branch_prefix, branch_child)]
        is_last = branch_child == children[branch_prefix][-1]

        if event.ranks:
            root_rank = event.ranks[0]
        else:
            root_rank = event.forest_start

        representative_rank = -1
        if not event.forest and root_rank not in represented_roots:
            representative_rank = root_rank
            represented_roots.add(root_rank)

        if event.forest:
            forest_count = event.forest_end - event.forest_start
            open_ended = False
            child = None
        else:
            forest_count = 0
            child = event.node
            open_ended = child is None or bool(tree[child].distribution.tokens)

        rows_out.append(
            CompactRow(
                parent=parent,
                rank=representative_rank,
                depth=len(ancestor_prefixes),
                ancestor_last=ancestor_last,
                is_last=is_last,
                tokens=event.tokens[common:],
                edge_nats=branch_nats,
                path_nats=event.nats,
                child=child,
                open_ended=open_ended,
                forest_count=forest_count,
                ancestor_nats=ancestor_nats,
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
    """Move selection within the currently visible sibling tail."""
    distribution = tree[view.node].distribution
    if not distribution.tokens:
        return view
    selected = min(max(view.selected_rank + delta, view.first_rank), len(distribution) - 1)
    return View(node=view.node, first_rank=view.first_rank, selected_rank=selected)
