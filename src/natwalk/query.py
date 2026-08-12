"""Read-only completion queries over discovered probability-tree knowledge."""

from __future__ import annotations

from dataclasses import dataclass

from .tree import NodeId, Tree


@dataclass(frozen=True, slots=True)
class Suggestion:
    """Stable endpoint edge of one suggested continuation.

    ``parent`` is always a discovered node. ``rank`` identifies its outgoing
    edge whether or not that edge's child distribution has been discovered yet.
    No token sequence is cached: text, cost, and highlighted ancestry are all
    derived from the current tree.
    """

    parent: NodeId
    rank: int


def suggestion_edges(
    tree: Tree,
    root: NodeId,
    suggestion: Suggestion,
) -> tuple[Suggestion, ...]:
    """Return the structural edge path from ``root`` through ``suggestion``."""
    edges = [suggestion]
    current = suggestion.parent
    while current != root:
        node = tree[current]
        if node.parent is None:
            raise ValueError("suggestion is not a descendant of root")
        edges.append(Suggestion(node.parent, node.rank))
        current = node.parent
    edges.reverse()
    return tuple(edges)


def suggestion_tokens(
    tree: Tree,
    root: NodeId,
    suggestion: Suggestion | None,
) -> tuple[int, ...]:
    """Derive suggestion tokens from structural tree edges."""
    if suggestion is None:
        return ()
    return tuple(tree[edge.parent].distribution.token(edge.rank) for edge in suggestion_edges(tree, root, suggestion))


def suggestion_nats(tree: Tree, root: NodeId, suggestion: Suggestion) -> float:
    """Derive cumulative surprisal of one suggestion from ``root``."""
    return sum(tree[edge.parent].distribution.nats(edge.rank) for edge in suggestion_edges(tree, root, suggestion))


def suggestion_complete(
    tree: Tree,
    root: NodeId,
    suggestion: Suggestion | None,
    *,
    max_nats: float,
    max_tokens: int,
) -> bool:
    """Whether known state proves that the suggestion should end here."""
    if suggestion is None:
        return True

    edges = suggestion_edges(tree, root, suggestion)
    child = tree.child(suggestion.parent, suggestion.rank)
    if child is None:
        return False
    if len(edges) >= max_tokens:
        return False

    distribution = tree[child].distribution
    if len(distribution) == 0:
        return True
    if distribution.revealed == 0:
        return False
    return suggestion_nats(tree, root, suggestion) + distribution.nats(0) > max_nats


def _extends(
    tree: Tree,
    root: NodeId,
    candidate: Suggestion,
    prefix: Suggestion,
) -> bool:
    """Whether ``candidate`` lies on or below ``prefix`` from ``root``."""
    return prefix in suggestion_edges(tree, root, candidate)


def normalize_suggestion(
    tree: Tree,
    root: NodeId,
    current: Suggestion | None,
    *,
    max_nats: float,
    max_tokens: int = 256,
    limit: int = 64,
) -> Suggestion | None:
    """Keep a structural selection stable as tree/query state changes.

    Exact endpoints survive unchanged. If new tree knowledge extends the
    selected branch, the first maximal completion on that same branch is used.
    If a smaller budget truncates the branch, its deepest valid ancestor is
    retained. Only when the old branch is no longer applicable do we fall back
    to the first current completion (or greedy endpoint when no alternatives
    are enumerated).
    """
    candidates = completions(
        tree,
        root,
        max_nats=max_nats,
        max_tokens=max_tokens,
        limit=limit,
    )
    if not candidates:
        return greedy(tree, root, max_nats=max_nats, max_tokens=max_tokens)
    if current is None:
        return candidates[0]
    if current in candidates:
        return current

    try:
        for candidate in candidates:
            if _extends(tree, root, candidate, current):
                return candidate

        old_edges = suggestion_edges(tree, root, current)
    except ValueError:
        return candidates[0]

    old_positions = {edge: index for index, edge in enumerate(old_edges)}
    ancestors = [candidate for candidate in candidates if candidate in old_positions]
    if ancestors:
        return max(ancestors, key=old_positions.__getitem__)
    return candidates[0]


def cycle_suggestion(
    tree: Tree,
    root: NodeId,
    current: Suggestion | None,
    step: int,
    *,
    max_nats: float,
    max_tokens: int = 256,
    limit: int = 64,
) -> Suggestion | None:
    """Move to the next/previous current completion endpoint."""
    candidates = completions(
        tree,
        root,
        max_nats=max_nats,
        max_tokens=max_tokens,
        limit=limit,
    )
    if not candidates:
        return greedy(tree, root, max_nats=max_nats, max_tokens=max_tokens)

    selected = normalize_suggestion(
        tree,
        root,
        current,
        max_nats=max_nats,
        max_tokens=max_tokens,
        limit=limit,
    )
    try:
        index = candidates.index(selected)
    except ValueError:
        index = 0
    return candidates[(index + step) % len(candidates)]


def greedy(
    tree: Tree,
    root: NodeId,
    *,
    max_nats: float,
    max_tokens: int = 256,
) -> Suggestion | None:
    """Return the known rank-zero endpoint inside ``max_nats``."""
    node_id = root
    endpoint: Suggestion | None = None
    used = 0.0

    for _ in range(max_tokens):
        distribution = tree[node_id].distribution
        if len(distribution) == 0 or distribution.revealed == 0:
            return endpoint

        cost = distribution.nats(0)
        if used + cost > max_nats:
            return endpoint

        endpoint = Suggestion(node_id, 0)
        used += cost
        child_id = tree.child(node_id, 0)
        if child_id is None:
            return endpoint
        node_id = child_id

    return endpoint


def completions(
    tree: Tree,
    root: NodeId,
    *,
    max_nats: float,
    max_tokens: int = 256,
    limit: int = 64,
) -> tuple[Suggestion, ...]:
    """Return maximal known endpoint edges in probability-ranked DFS order."""
    if limit <= 0 or max_tokens <= 0:
        return ()

    out: list[Suggestion] = []

    def visit(
        node_id: NodeId,
        endpoint: Suggestion | None,
        depth: int,
        used: float,
    ) -> None:
        if len(out) >= limit:
            return

        distribution = tree[node_id].distribution
        if len(distribution) == 0:
            if endpoint is not None:
                out.append(endpoint)
            return

        extended = False
        for rank in range(distribution.revealed):
            cost = distribution.nats(rank)
            next_used = used + cost
            if next_used > max_nats:
                break

            extended = True
            candidate = Suggestion(node_id, rank)
            child_id = tree.child(node_id, rank)
            if depth + 1 >= max_tokens or child_id is None:
                out.append(candidate)
            else:
                before = len(out)
                visit(child_id, candidate, depth + 1, next_used)
                if len(out) == before:
                    out.append(candidate)
            if len(out) >= limit:
                return

        if endpoint is not None and not extended:
            out.append(endpoint)

    visit(root, None, 0, 0.0)
    return tuple(out[:limit])
