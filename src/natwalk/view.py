"""Pure views over a discovered probability tree."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
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
    """One weighted row in a compressed partial-trie view.

    ``tokens`` may contain a unary discovered chain. ``open_ended`` means the
    displayed path continues into tree state that was not selected for this
    frame. ``forest_count`` instead denotes an aggregate event containing that
    many omitted sibling edges; its ``path_nats`` is computed from their total
    probability mass.
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

    @property
    def forest(self) -> bool:
        return self.forest_count != 0


@dataclass(frozen=True, slots=True)
class _CompactItem:
    parent: NodeId
    rank: int
    tokens: tuple[int, ...]
    edge_nats: float
    path_nats: float
    child: NodeId | None
    open_ended: bool = False
    forest_count: int = 0
    side_forests: tuple[_CompactItem, ...] = ()


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
    """Render a best-first finite partition of continuation space.

    Every returned row is one disjoint probability event. Starting from the
    whole visible sibling tail, each additional row refines exactly one event
    into two disjoint events. Refinements compete globally by the probability
    of their smaller result, so an extremely unlikely side branch cannot steal
    a row while a more probable unresolved split is available elsewhere.

    This first layout deliberately spends no extra rows on internal trie nodes:
    concrete prefixes are stacked horizontally and sibling forests end in an
    ellipsis. Shared-prefix factoring is a separate presentation problem.
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
        candidates: list[tuple[float, float, tuple[int, ...], int, tuple[_PartitionEvent, ...]]] = []
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
    return _partition_compact_rows(tree, root, events)


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


def _partition_compact_rows(
    tree: Tree,
    root: NodeId,
    events: list[_PartitionEvent],
) -> tuple[CompactRow, ...]:
    rows_out: list[CompactRow] = []
    represented_roots: set[int] = set()

    for event in events:
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
                parent=root,
                rank=representative_rank,
                depth=0,
                ancestor_last=(),
                is_last=False,
                tokens=event.tokens,
                edge_nats=event.nats,
                path_nats=event.nats,
                child=child,
                open_ended=open_ended,
                forest_count=forest_count,
            )
        )

    if rows_out:
        rows_out[-1] = replace(rows_out[-1], is_last=True)
    return tuple(rows_out)


def compact_rows(
    tree: Tree,
    view: View,
    *,
    edge_limit: int,
    first_rank: int | None = None,
) -> tuple[CompactRow, ...]:
    """Return a relevance-pruned, unary-compressed view of the partial trie.

    The read-only selection phase is a lazy best-first walk over already-known
    distributions. It may cross an edge only when that edge's child has already
    been discovered. Missing edges remain virtual leaves. Rendering then
    compresses unary selected chains and represents omitted sibling tails by
    their aggregate probability mass.
    """
    if edge_limit <= 0:
        return ()

    root = view.node
    start = view.first_rank if first_rank is None else first_rank
    distribution = tree[root].distribution
    if not 0 <= start <= len(distribution):
        raise IndexError(start)
    if start == len(distribution):
        return ()

    shown, parent_nats = _best_edges(tree, root, start, edge_limit)

    def visit(
        parent_id: NodeId,
        depth: int,
        ancestor_last: tuple[bool, ...],
    ):
        ranks = shown.get(parent_id, ())
        if not ranks:
            return

        items: list[_CompactItem] = []
        for rank in ranks:
            branch = _compact_branch(tree, shown, parent_nats, parent_id, rank)
            items.append(replace(branch, side_forests=()))
            items.extend(branch.side_forests)

        if parent_id != root and ranks[0] == 0:
            tail = _tail_forest(
                tree[parent_id].distribution,
                ranks[-1] + 1,
                parent_nats=parent_nats[parent_id],
                parent=parent_id,
            )
            if tail is not None:
                items.append(tail)

        for index, item in enumerate(items):
            is_last = index == len(items) - 1
            yield CompactRow(
                parent=item.parent,
                rank=item.rank,
                depth=depth,
                ancestor_last=ancestor_last,
                is_last=is_last,
                tokens=item.tokens,
                edge_nats=item.edge_nats,
                path_nats=item.path_nats,
                child=item.child,
                open_ended=item.open_ended,
                forest_count=item.forest_count,
            )
            if item.child is not None and not item.open_ended and shown.get(item.child):
                yield from visit(item.child, depth + 1, (*ancestor_last, is_last))

    return tuple(visit(root, 0, ()))


def _best_edges(
    tree: Tree,
    root: NodeId,
    first_rank: int,
    limit: int,
) -> tuple[dict[NodeId, tuple[int, ...]], dict[NodeId, float]]:
    shown: dict[NodeId, list[int]] = {}
    parent_nats: dict[NodeId, float] = {root: 0.0}
    frontier: list[tuple[float, NodeId, int, float]] = []

    def push(parent: NodeId, rank: int, base_nats: float) -> None:
        distribution = tree[parent].distribution
        if rank < len(distribution):
            heapq.heappush(
                frontier,
                (base_nats + distribution.nats(rank), parent, rank, base_nats),
            )

    push(root, first_rank, 0.0)
    while frontier and sum(len(ranks) for ranks in shown.values()) < limit:
        path_nats, parent, rank, base_nats = heapq.heappop(frontier)
        ranks = shown.setdefault(parent, [])
        if rank in ranks:
            continue
        ranks.append(rank)

        push(parent, rank + 1, base_nats)
        child = tree.child(parent, rank)
        if child is not None:
            parent_nats[child] = path_nats
            push(child, 0, path_nats)

    return (
        {parent: tuple(sorted(ranks)) for parent, ranks in shown.items()},
        parent_nats,
    )


def _compact_branch(
    tree: Tree,
    shown: dict[NodeId, tuple[int, ...]],
    parent_nats: dict[NodeId, float],
    parent: NodeId,
    rank: int,
) -> _CompactItem:
    distribution = tree[parent].distribution
    first_edge_nats = distribution.nats(rank)
    path_nats = parent_nats[parent] + first_edge_nats
    tokens = [distribution.tokens[rank]]
    child = tree.child(parent, rank)
    side_forests: list[_CompactItem] = []

    if child is None:
        return _CompactItem(
            parent=parent,
            rank=rank,
            tokens=tuple(tokens),
            edge_nats=first_edge_nats,
            path_nats=path_nats,
            child=None,
            open_ended=True,
        )

    endpoint = child
    total_edge_nats = first_edge_nats
    while True:
        child_ranks = shown.get(endpoint, ())
        if len(child_ranks) != 1:
            break

        next_rank = child_ranks[0]
        endpoint_distribution = tree[endpoint].distribution
        if next_rank == 0:
            tail = _tail_forest(
                endpoint_distribution,
                1,
                parent_nats=path_nats,
                parent=parent,
                tokens=tuple(tokens),
                rank=rank,
            )
            if tail is not None:
                side_forests.append(tail)

        next_edge_nats = endpoint_distribution.nats(next_rank)
        total_edge_nats += next_edge_nats
        path_nats += next_edge_nats
        tokens.append(endpoint_distribution.tokens[next_rank])

        next_child = tree.child(endpoint, next_rank)
        if next_child is None:
            return _CompactItem(
                parent=parent,
                rank=rank,
                tokens=tuple(tokens),
                edge_nats=total_edge_nats,
                path_nats=path_nats,
                child=None,
                open_ended=True,
                side_forests=tuple(side_forests),
            )
        endpoint = next_child

    open_ended = bool(tree[endpoint].distribution.tokens and not shown.get(endpoint))
    return _CompactItem(
        parent=parent,
        rank=rank,
        tokens=tuple(tokens),
        edge_nats=total_edge_nats,
        path_nats=path_nats,
        child=endpoint,
        open_ended=open_ended,
        side_forests=tuple(side_forests),
    )


def _tail_forest(
    distribution: Distribution,
    start: int,
    *,
    parent_nats: float,
    parent: NodeId,
    tokens: tuple[int, ...] = (),
    rank: int = -1,
) -> _CompactItem | None:
    if start >= len(distribution):
        return None
    return _CompactItem(
        parent=parent,
        rank=rank,
        tokens=tokens,
        edge_nats=forest_nats(distribution, start),
        path_nats=forest_nats(distribution, start, parent_nats=parent_nats),
        child=None,
        forest_count=len(distribution) - start,
    )


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
