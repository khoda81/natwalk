"""Pure views over a discovered probability tree."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

from .tree import NodeId, Tree


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


def iter_rows(tree: Tree, view: View):
    """Yield the view's known tree in DFS order without mutating it."""

    def visit(
        parent_id: NodeId,
        first_rank: int,
        depth: int,
        ancestor_last: tuple[bool, ...],
        parent_nats: float,
    ):
        parent = tree[parent_id]
        distribution = parent.distribution
        if distribution is None:
            return

        for rank in range(first_rank, len(distribution)):
            child_id = parent.children.get(rank)
            child = tree[child_id] if child_id is not None else None
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
                expanded=child is not None and child.distribution is not None,
            )
            if child is not None and child.distribution is not None:
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


def enter(tree: Tree, view: View) -> View:
    """Focus the selected child, materializing only its node identity."""
    child = tree.child(view.node, view.selected_rank)
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
    if distribution is None or not distribution.tokens:
        return view
    selected = min(max(view.selected_rank + delta, view.first_rank), len(distribution) - 1)
    return View(node=view.node, first_rank=view.first_rank, selected_rank=selected)
