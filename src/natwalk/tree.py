"""Persistent append-only probability tree."""

from __future__ import annotations

import math
import operator
import sys
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from struct import Struct
from types import MappingProxyType
from typing import Protocol, runtime_checkable

type NodeId = int


@runtime_checkable
class RankedDistribution(Protocol):
    """Probability-ranked next-symbol distribution.

    Backends own the concrete representation. Natwalk only requires random
    access by probability rank, exact aggregate range mass, reverse token
    lookup for explicit navigation, and a retained-storage estimate.

    ``mass(start, end)`` is the probability of the clipped half-open rank range
    ``[start, end) ∩ [0, len(self))``. Empty or reversed ranges have zero mass;
    negative endpoints are coordinates outside the domain, not Python-style
    indices from the end.
    """

    def __len__(self) -> int: ...

    @property
    def revealed(self) -> int:
        """Concrete ranked prefix currently available to the consumer."""
        ...

    @property
    def storage_bytes(self) -> int:
        """Approximate retained payload owned by this distribution."""
        ...

    def token(self, rank: int) -> int: ...

    def probability(self, rank: int) -> float: ...

    def mass(self, start: int, end: int) -> float: ...

    def rank(self, token: int) -> int: ...

    def nats(self, rank: int) -> float: ...


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class _PackedUInt32(Sequence[int]):
    """Immutable packed unsigned-32 sequence used for token ids."""

    _data: bytes
    _STRUCT = Struct("<I")
    _ITEM_SIZE = _STRUCT.size

    @classmethod
    def pack(cls, values: Sequence[int]) -> _PackedUInt32:
        if isinstance(values, cls):
            return values
        packed = array("I", (operator.index(value) for value in values))
        if packed.itemsize != cls._ITEM_SIZE:
            raise RuntimeError("platform uint size is not 32 bits")
        if sys.byteorder != "little":
            packed.byteswap()
        return cls(packed.tobytes())

    def __len__(self) -> int:
        return len(self._data) // self._ITEM_SIZE

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return type(self)(
                    self._data[start * self._ITEM_SIZE : stop * self._ITEM_SIZE]
                )
            return tuple(self[position] for position in range(start, stop, step))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._STRUCT.unpack_from(self._data, index * self._ITEM_SIZE)[0]

    def __iter__(self) -> Iterator[int]:
        for (value,) in self._STRUCT.iter_unpack(self._data):
            yield value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self._data == other._data
        if isinstance(other, Sequence):
            return len(self) == len(other) and all(
                left == right for left, right in zip(self, other, strict=True)
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._data)

    def index(self, value: int, start: int = 0, stop: int | None = None) -> int:
        if stop is None:
            stop = len(self)
        start, stop, _ = slice(start, stop).indices(len(self))
        for position in range(start, stop):
            if self[position] == value:
                return position
        raise ValueError(f"{value!r} is not in sequence")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(len={len(self)})"


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class _PackedFloat64(Sequence[float]):
    """Immutable packed IEEE-754 binary64 sequence used for probabilities."""

    _data: bytes
    _STRUCT = Struct("<d")
    _ITEM_SIZE = _STRUCT.size

    @classmethod
    def pack(cls, values: Sequence[float]) -> _PackedFloat64:
        if isinstance(values, cls):
            return values
        packed = array("d", (float(value) for value in values))
        if packed.itemsize != cls._ITEM_SIZE:
            raise RuntimeError("platform double size is not 64 bits")
        if sys.byteorder != "little":
            packed.byteswap()
        return cls(packed.tobytes())

    def __len__(self) -> int:
        return len(self._data) // self._ITEM_SIZE

    def __getitem__(self, index: int | slice) -> float | Sequence[float]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return type(self)(
                    self._data[start * self._ITEM_SIZE : stop * self._ITEM_SIZE]
                )
            return tuple(self[position] for position in range(start, stop, step))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._STRUCT.unpack_from(self._data, index * self._ITEM_SIZE)[0]

    def __iter__(self) -> Iterator[float]:
        for (value,) in self._STRUCT.iter_unpack(self._data):
            yield value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self._data == other._data
        if isinstance(other, Sequence):
            return len(self) == len(other) and all(
                left == right for left, right in zip(self, other, strict=True)
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(len={len(self)})"


@dataclass(frozen=True, slots=True)
class Distribution:
    """Dependency-free complete ranked distribution.

    This is Natwalk's default adapter for ordinary ``Sequence[float]`` model
    outputs. Backends may instead supply any object implementing
    :class:`RankedDistribution` and retain their own native representation.
    """

    tokens: Sequence[int]
    probabilities: Sequence[float]

    def __post_init__(self) -> None:
        tokens = _PackedUInt32.pack(self.tokens)
        probabilities = _PackedFloat64.pack(self.probabilities)
        if len(tokens) != len(probabilities):
            raise ValueError("distribution token/probability lengths differ")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "probabilities", probabilities)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def revealed(self) -> int:
        return len(self)

    @property
    def storage_bytes(self) -> int:
        return len(self) * (_PackedUInt32._ITEM_SIZE + _PackedFloat64._ITEM_SIZE)

    def token(self, rank: int) -> int:
        return self.tokens[rank]

    def probability(self, rank: int) -> float:
        return self.probabilities[rank]

    def mass(self, start: int, end: int) -> float:
        start = min(max(start, 0), len(self))
        end = min(max(end, 0), len(self))
        if start >= end:
            return 0.0
        return math.fsum(self.probabilities[start:end])

    def rank(self, token: int) -> int:
        return self.tokens.index(token)

    def nats(self, rank: int) -> float:
        probability = self.probability(rank)
        return -math.log(probability) if probability != 0.0 else math.inf


@dataclass(frozen=True, slots=True)
class Node:
    """One complete immutable node in the discovered tree."""

    parent: NodeId | None
    rank: int
    distribution: RankedDistribution
    _children: dict[int, NodeId] = field(default_factory=dict, compare=False, repr=False)

    @property
    def children(self) -> Mapping[int, NodeId]:
        return MappingProxyType(self._children)


class Tree:
    """Arena-backed append-only probability trie.

    A node is published only after its authoritative distribution is known.
    Child insertion is idempotent: repeating the same write returns the existing
    node; attempting to give an existing edge a different distribution is an
    invariant violation.
    """

    def __init__(self, root_distribution: RankedDistribution) -> None:
        self.nodes: list[Node] = [
            Node(parent=None, rank=-1, distribution=root_distribution)
        ]
        self._storage_bytes = root_distribution.storage_bytes

    @property
    def root(self) -> NodeId:
        return 0

    @property
    def storage_bytes(self) -> int:
        return self._storage_bytes

    def __getitem__(self, node: NodeId) -> Node:
        return self.nodes[node]

    def child(self, parent_id: NodeId, rank: int) -> NodeId | None:
        return self.nodes[parent_id]._children.get(rank)

    def put_child(
        self,
        parent_id: NodeId,
        rank: int,
        distribution: RankedDistribution,
    ) -> NodeId:
        parent = self.nodes[parent_id]
        if not 0 <= rank < len(parent.distribution):
            raise IndexError(rank)

        existing = parent._children.get(rank)
        if existing is not None:
            node = self.nodes[existing]
            if node.distribution != distribution:
                raise ValueError(f"conflicting distribution for child ({parent_id}, {rank})")
            return existing

        child_id = len(self.nodes)
        child = Node(parent=parent_id, rank=rank, distribution=distribution)
        self.nodes.append(child)
        self._storage_bytes += distribution.storage_bytes
        parent._children[rank] = child_id
        return child_id

    def account_distribution_growth(self, node_id: NodeId, previous_bytes: int) -> None:
        """Account for progressive storage growth of one already-published node."""
        distribution = self.nodes[node_id].distribution
        if distribution.storage_bytes < previous_bytes:
            raise ValueError("published distribution storage cannot shrink")
        self._storage_bytes += distribution.storage_bytes - previous_bytes

    def token(self, node_id: NodeId) -> int:
        node = self.nodes[node_id]
        if node.parent is None:
            raise ValueError("root has no token")
        return self.nodes[node.parent].distribution.token(node.rank)

    def edge_nats(self, node_id: NodeId) -> float:
        node = self.nodes[node_id]
        if node.parent is None:
            return 0.0
        return self.nodes[node.parent].distribution.nats(node.rank)

    def path(self, node_id: NodeId) -> tuple[int, ...]:
        return self.path_from(self.root, node_id)

    def path_from(self, ancestor: NodeId, node_id: NodeId) -> tuple[int, ...]:
        """Return tokens from ``ancestor`` (exclusive) to ``node_id`` (inclusive)."""
        tokens: list[int] = []
        current = node_id
        while current != ancestor:
            node = self.nodes[current]
            if node.parent is None:
                raise ValueError("node is not a descendant of ancestor")
            tokens.append(self.token(current))
            current = node.parent
        tokens.reverse()
        return tuple(tokens)

    def path_nats(self, node_id: NodeId, *, ancestor: NodeId = 0) -> float:
        """Derive cumulative surprisal from ``ancestor`` to ``node_id``."""
        total = 0.0
        current = node_id
        while current != ancestor:
            node = self.nodes[current]
            if node.parent is None:
                raise ValueError("node is not a descendant of ancestor")
            total += self.edge_nats(current)
            current = node.parent
        return total
