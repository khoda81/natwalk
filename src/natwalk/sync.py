"""Progressive synchronization for append-only discovered trees."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass

from .tree import NodeId, RankedDistribution, Tree

_INITIAL_REVEAL = 128


class ReplicaDistribution:
    """Exact ranked prefix, sparse exact pins, and aggregate unrevealed tail."""

    __slots__ = (
        "_size",
        "_tokens",
        "_probabilities",
        "_pins",
        "_tail_probability",
    )

    def __init__(
        self,
        size: int,
        tokens: tuple[int, ...],
        probabilities: tuple[float, ...],
        tail_probability: float,
    ) -> None:
        if size < 0:
            raise ValueError("distribution size cannot be negative")
        if len(tokens) != len(probabilities):
            raise ValueError("revealed token/probability lengths differ")
        if len(tokens) > size:
            raise ValueError("revealed prefix exceeds distribution size")
        if tail_probability < 0.0:
            raise ValueError("distribution tail probability cannot be negative")
        if len(tokens) == size and tail_probability != 0.0:
            raise ValueError("fully revealed distribution must have zero tail mass")

        self._size = size
        self._tokens = array("I", tokens)
        self._probabilities = array("d", probabilities)
        self._pins: dict[int, tuple[int, float]] = {}
        self._tail_probability = float(tail_probability)

    def __len__(self) -> int:
        return self._size

    @property
    def revealed(self) -> int:
        return len(self._tokens)

    @property
    def storage_bytes(self) -> int:
        return (
            self._tokens.itemsize * len(self._tokens)
            + self._probabilities.itemsize * len(self._probabilities)
            + 12 * len(self._pins)
            + 8
        )

    @property
    def tail_probability(self) -> float:
        return self._tail_probability

    def token(self, rank: int) -> int:
        if 0 <= rank < self.revealed:
            return int(self._tokens[rank])
        pinned = self._pins.get(rank)
        if pinned is not None:
            return pinned[0]
        raise IndexError(f"rank {rank} has not been revealed or pinned")

    def probability(self, rank: int) -> float:
        if 0 <= rank < self.revealed:
            return float(self._probabilities[rank])
        pinned = self._pins.get(rank)
        if pinned is not None:
            return pinned[1]
        raise IndexError(f"rank {rank} has not been revealed or pinned")

    def mass(self, start: int, end: int) -> float:
        start = min(max(start, 0), len(self))
        end = min(max(end, 0), len(self))
        if start >= end:
            return 0.0
        if end <= self.revealed:
            return math.fsum(self._probabilities[start:end])
        if end == len(self) and start <= self.revealed:
            return (
                math.fsum(self._probabilities[start : self.revealed])
                + self._tail_probability
            )

        probabilities: list[float] = []
        for rank in range(start, end):
            if rank < self.revealed:
                probabilities.append(float(self._probabilities[rank]))
                continue
            pinned = self._pins.get(rank)
            if pinned is None:
                raise IndexError("partial unrevealed range mass is not available")
            probabilities.append(pinned[1])
        return math.fsum(probabilities)

    def rank(self, token: int) -> int:
        try:
            return self._tokens.index(token)
        except ValueError:
            for rank, (pinned_token, _probability) in self._pins.items():
                if pinned_token == token:
                    return rank
            if self.revealed < len(self):
                raise ValueError(f"token {token} has not been revealed or pinned") from None
            raise

    def nats(self, rank: int) -> float:
        probability = self.probability(rank)
        return -math.log(probability) if probability != 0.0 else math.inf

    def pin(self, rank: int, token: int, probability: float) -> None:
        """Make one exact out-of-prefix rank available without revealing its neighbors."""
        if not 0 <= rank < len(self):
            raise IndexError(rank)
        token = int(token)
        probability = float(probability)
        if rank < self.revealed:
            if self.token(rank) != token or self.probability(rank) != probability:
                raise ValueError(f"conflicting pinned contents at rank {rank}")
            return

        existing = self._pins.get(rank)
        if existing is not None:
            if existing != (token, probability):
                raise ValueError(f"conflicting pinned contents at rank {rank}")
            return
        self._pins[rank] = (token, probability)

    def reveal(
        self,
        start: int,
        tokens: tuple[int, ...],
        probabilities: tuple[float, ...],
        tail_probability: float,
    ) -> None:
        """Extend the concrete prefix, verifying overlap, pins, and mass conservation."""
        if len(tokens) != len(probabilities):
            raise ValueError("revealed token/probability lengths differ")
        end = start + len(tokens)
        if not 0 <= start <= end <= len(self):
            raise IndexError((start, end))
        if start > self.revealed:
            raise ValueError(
                f"missing distribution reveal before rank {start}; expected {self.revealed}"
            )

        overlap = min(self.revealed - start, len(tokens))
        for offset in range(overlap):
            rank = start + offset
            if self.token(rank) != tokens[offset]:
                raise ValueError(f"conflicting token at revealed rank {rank}")
            if self.probability(rank) != probabilities[offset]:
                raise ValueError(f"conflicting probability at revealed rank {rank}")

        if end <= self.revealed:
            return

        for rank, (pinned_token, pinned_probability) in tuple(self._pins.items()):
            if not self.revealed <= rank < end:
                continue
            offset = rank - start
            if tokens[offset] != pinned_token or probabilities[offset] != pinned_probability:
                raise ValueError(f"revealed prefix conflicts with pinned rank {rank}")

        new_probabilities = probabilities[overlap:]
        expected_tail = math.fsum(new_probabilities) + tail_probability
        if not math.isclose(
            expected_tail,
            self._tail_probability,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("distribution reveal does not conserve tail probability mass")

        previous_revealed = self.revealed
        self._tokens.extend(tokens[overlap:])
        self._probabilities.extend(new_probabilities)
        self._tail_probability = float(tail_probability)
        for rank in tuple(self._pins):
            if previous_revealed <= rank < self.revealed:
                del self._pins[rank]
        if self.revealed == len(self) and self._tail_probability != 0.0:
            raise ValueError("fully revealed distribution must have zero tail mass")


@dataclass(frozen=True, slots=True)
class NodeUpdate:
    """Absolute node metadata plus a small exact ranked prefix."""

    node: NodeId
    parent: NodeId | None
    rank: int
    size: int
    tokens: tuple[int, ...]
    probabilities: tuple[float, ...]
    tail_probability: float


@dataclass(frozen=True, slots=True)
class RevealUpdate:
    """Extend one replica node's concrete ranked prefix."""

    node: NodeId
    start: int
    tokens: tuple[int, ...]
    probabilities: tuple[float, ...]
    tail_probability: float


@dataclass(frozen=True, slots=True)
class RankUpdate:
    """Pin one exact rank needed by a discovered edge outside the visible prefix."""

    node: NodeId
    rank: int
    token: int
    probability: float


type TreeUpdate = NodeUpdate | RevealUpdate | RankUpdate


def _snapshot(
    distribution: RankedDistribution,
    start: int,
    stop: int,
) -> tuple[tuple[int, ...], tuple[float, ...], float]:
    if not 0 <= start <= stop <= len(distribution):
        raise IndexError((start, stop))
    if stop > distribution.revealed:
        raise IndexError(f"rank {stop - 1} has not been revealed")

    tokens = tuple(distribution.token(rank) for rank in range(start, stop))
    probabilities = tuple(distribution.probability(rank) for rank in range(start, stop))
    tail_probability = distribution.mass(stop, len(distribution))
    return tokens, probabilities, tail_probability


def pin(tree: Tree, node: NodeId, rank: int) -> RankUpdate:
    """Return one exact sparse rank dependency for a replica tree edge."""
    distribution = tree[node].distribution
    return RankUpdate(
        node=node,
        rank=rank,
        token=distribution.token(rank),
        probability=distribution.probability(rank),
    )


def updates(
    tree: Tree,
    *,
    start: NodeId = 0,
    initial_reveal: int = _INITIAL_REVEAL,
) -> tuple[TreeUpdate, ...]:
    """Return an append-log suffix plus sparse rank dependencies for its edges."""
    if not 0 <= start <= len(tree.nodes):
        raise IndexError(start)
    if initial_reveal <= 0:
        raise ValueError("initial_reveal must be positive")

    result: list[TreeUpdate] = []
    for node_id, node in enumerate(tree.nodes[start:], start=start):
        if node.parent is not None and node.rank >= initial_reveal:
            result.append(pin(tree, node.parent, node.rank))

        distribution = node.distribution
        stop = min(len(distribution), initial_reveal, distribution.revealed)
        tokens, probabilities, tail_probability = _snapshot(distribution, 0, stop)
        result.append(
            NodeUpdate(
                node=node_id,
                parent=node.parent,
                rank=node.rank,
                size=len(distribution),
                tokens=tokens,
                probabilities=probabilities,
                tail_probability=tail_probability,
            )
        )
    return tuple(result)


def reveal(
    tree: Tree,
    node: NodeId,
    start: int,
    stop: int,
) -> RevealUpdate:
    """Reveal one contiguous authoritative ranked page for an existing node."""
    distribution = tree[node].distribution
    stop = min(stop, distribution.revealed)
    tokens, probabilities, tail_probability = _snapshot(distribution, start, stop)
    return RevealUpdate(
        node=node,
        start=start,
        tokens=tokens,
        probabilities=probabilities,
        tail_probability=tail_probability,
    )


class TreeReplica:
    """Client tree reconstructed from progressive idempotent updates."""

    def __init__(self) -> None:
        self.tree: Tree | None = None

    @property
    def next_node(self) -> NodeId:
        return 0 if self.tree is None else len(self.tree.nodes)

    def apply(self, update: TreeUpdate) -> None:
        if isinstance(update, RevealUpdate):
            self._apply_reveal(update)
            return
        if isinstance(update, RankUpdate):
            self._apply_rank(update)
            return
        if update.node == 0:
            self._apply_root(update)
            return

        tree = self.tree
        if tree is None:
            raise ValueError("tree sync must start with root node 0")

        if update.node < len(tree.nodes):
            self._verify_existing(update)
            return
        if update.node > len(tree.nodes):
            raise ValueError(
                f"missing tree updates before node {update.node}; expected {len(tree.nodes)}"
            )
        if update.parent is None:
            raise ValueError("non-root tree update has no parent")

        parent_distribution = tree[update.parent].distribution
        try:
            parent_distribution.token(update.rank)
            parent_distribution.probability(update.rank)
        except IndexError as error:
            raise ValueError(
                f"tree update edge ({update.parent}, {update.rank}) is not revealed or pinned"
            ) from error

        child = tree.put_child(update.parent, update.rank, self._distribution(update))
        if child != update.node:
            raise ValueError(f"tree update {update.node} conflicts with existing child id {child}")

    def apply_many(self, batch: tuple[TreeUpdate, ...]) -> None:
        for update in batch:
            self.apply(update)

    @staticmethod
    def _distribution(update: NodeUpdate) -> ReplicaDistribution:
        return ReplicaDistribution(
            update.size,
            update.tokens,
            update.probabilities,
            update.tail_probability,
        )

    def _apply_root(self, update: NodeUpdate) -> None:
        if update.parent is not None or update.rank != -1:
            raise ValueError("root update must have parent=None and rank=-1")
        if self.tree is None:
            self.tree = Tree(self._distribution(update))
            return
        self._verify_existing(update)

    def _verify_existing(self, update: NodeUpdate) -> None:
        assert self.tree is not None
        node = self.tree[update.node]
        distribution = node.distribution
        if node.parent != update.parent or node.rank != update.rank:
            raise ValueError(f"conflicting contents for tree node {update.node}")
        if len(distribution) != update.size:
            raise ValueError(f"conflicting contents for tree node {update.node}")
        if distribution.revealed < len(update.tokens):
            raise ValueError(f"tree node {update.node} lost revealed ranks")
        for rank, (token, probability) in enumerate(
            zip(update.tokens, update.probabilities, strict=True)
        ):
            if distribution.token(rank) != token or distribution.probability(rank) != probability:
                raise ValueError(f"conflicting contents for tree node {update.node}")
        if not math.isclose(
            distribution.mass(len(update.tokens), len(distribution)),
            update.tail_probability,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"conflicting probability mass for tree node {update.node}")
        if update.parent is not None and self.tree.child(update.parent, update.rank) != update.node:
            raise ValueError(f"conflicting edge for tree node {update.node}")

    def _apply_reveal(self, update: RevealUpdate) -> None:
        tree = self.tree
        if tree is None or not 0 <= update.node < len(tree.nodes):
            raise ValueError(f"cannot reveal unknown tree node {update.node}")
        distribution = tree[update.node].distribution
        if not isinstance(distribution, ReplicaDistribution):
            raise TypeError("tree replica contains an authoritative distribution")

        before = distribution.storage_bytes
        distribution.reveal(
            update.start,
            update.tokens,
            update.probabilities,
            update.tail_probability,
        )
        tree.account_distribution_growth(update.node, before)

    def _apply_rank(self, update: RankUpdate) -> None:
        tree = self.tree
        if tree is None or not 0 <= update.node < len(tree.nodes):
            raise ValueError(f"cannot pin rank on unknown tree node {update.node}")
        distribution = tree[update.node].distribution
        if not isinstance(distribution, ReplicaDistribution):
            raise TypeError("tree replica contains an authoritative distribution")

        before = distribution.storage_bytes
        distribution.pin(update.rank, update.token, update.probability)
        tree.account_distribution_growth(update.node, before)
