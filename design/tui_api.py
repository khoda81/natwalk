"""Proposed boundary for the shared natwalk probability-tree UI.

This file is intentionally an API sketch, not an implementation. Nothing in the
package imports it. The goal is to agree on ownership and invariants before
migrating either example.

Model-specific boundary
=======================

A backend provides the existing ``Cursor`` plus a token-description function.
It knows about GGUFs, audio chunks, MIDI programs, tokenizers, and model state.
The shared UI does not.

Core/UI boundary
================

The UI consumes an immutable probability-tree snapshot. Search materialization
and known conditional distributions are deliberately separate: inspecting a
prefix may populate ``distributions`` without making that prefix part of the
Dijkstra search tree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from natwalk import Cursor, ExplorerStats, NavigationSnapshot, RankedDistribution

TokenPath: TypeAlias = tuple[int, ...]
DescribeToken: TypeAlias = Callable[[int], str]


@dataclass(frozen=True, slots=True)
class SearchEdge:
    """One path materialized by Dijkstra.

    ``path`` is non-empty. The token and parent are therefore derived as
    ``path[-1]`` and ``path[:-1]`` rather than duplicated here.
    """

    path: TokenPath
    rank: int
    edge_nats: float
    path_nats: float


@dataclass(frozen=True, slots=True)
class ProbabilityTreeSnapshot:
    """Immutable knowledge shared by search, inspection, and the view layer.

    ``materialized`` answers which edges Dijkstra has admitted to its search
    tree. ``distributions`` answers which complete conditional distributions
    are already known, regardless of whether search or human inspection caused
    them to be computed. Keeping those facts separate is the key invariant.
    """

    materialized: Mapping[TokenPath, SearchEdge]
    distributions: Mapping[TokenPath, RankedDistribution]
    ended: frozenset[TokenPath]
    pending_distributions: frozenset[TokenPath]


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Everything the shared UI may read from natwalk core state."""

    navigation: NavigationSnapshot
    tree: ProbabilityTreeSnapshot
    search: ExplorerStats
    choices: int


class NatwalkSession(Protocol):
    """Command/query boundary between the shared app and natwalk core.

    A concrete session will own ``Navigator``, Dijkstra, the shared distribution
    cache, and inspection requests. The UI never reaches into explorer private
    fields.
    """

    def snapshot(self) -> SessionSnapshot: ...

    def choose(self, bucket: int) -> tuple[int, ...]: ...

    def accept(self, tokens: TokenPath) -> None: ...

    def undo(self) -> bool: ...

    def request_distribution(self, path: TokenPath) -> None: ...


@dataclass(frozen=True, slots=True)
class AppState:
    """Minimal human-interface state.

    Viewport position is intentionally absent. Given ``selected_rank`` and the
    current terminal height, the visible rank range is derived. We only store
    state that cannot be reconstructed.
    """

    focus_path: TokenPath = ()
    selected_rank: int = 0
    pending_focus: TokenPath | None = None
    budget_nats: float = 1.5
    completion_index: int = 0
    debug: bool = False


# Semantic actions are deliberately independent of terminal key sequences. A
# terminal frontend can map arrows/bytes to these; another frontend need not.
@dataclass(frozen=True, slots=True)
class MoveRank:
    delta: int


@dataclass(frozen=True, slots=True)
class EnterSelected:
    pass


@dataclass(frozen=True, slots=True)
class FocusParent:
    pass


@dataclass(frozen=True, slots=True)
class ChooseBucket:
    bucket: int


@dataclass(frozen=True, slots=True)
class AcceptCompletion:
    pass


@dataclass(frozen=True, slots=True)
class Undo:
    pass


@dataclass(frozen=True, slots=True)
class AdjustBudget:
    delta_nats: float


@dataclass(frozen=True, slots=True)
class CycleCompletion:
    delta: int


@dataclass(frozen=True, slots=True)
class ToggleDebug:
    pass


Action: TypeAlias = (
    MoveRank
    | EnterSelected
    | FocusParent
    | ChooseBucket
    | AcceptCompletion
    | Undo
    | AdjustBudget
    | CycleCompletion
    | ToggleDebug
)


# Pure state transitions may ask the imperative shell to perform one of these
# effects. Model/search mutation therefore stays out of the reducer.
@dataclass(frozen=True, slots=True)
class RequestDistribution:
    path: TokenPath


@dataclass(frozen=True, slots=True)
class ApplyBucket:
    bucket: int


@dataclass(frozen=True, slots=True)
class ApplyCompletion:
    tokens: TokenPath


@dataclass(frozen=True, slots=True)
class ApplyUndo:
    pass


Effect: TypeAlias = RequestDistribution | ApplyBucket | ApplyCompletion | ApplyUndo


@dataclass(frozen=True, slots=True)
class Transition:
    state: AppState
    effects: tuple[Effect, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenSegment:
    """One token inside a rendered branch or compressed corridor."""

    path: TokenPath
    token: int
    edge_nats: float


@dataclass(frozen=True, slots=True)
class BranchRow:
    """One visible branch row; ``segments`` is non-empty.

    A corridor is simply a branch row containing multiple token segments. Local
    probability luminance therefore falls out naturally from each segment's
    ``edge_nats`` while ``total_nats`` remains the corridor's aggregate cost.
    """

    segments: tuple[TokenSegment, ...]
    depth: int
    ancestor_last: tuple[bool, ...]
    is_last: bool
    total_nats: float


@dataclass(frozen=True, slots=True)
class EllipsisRow:
    """Exact probability mass represented by unshown alternatives."""

    parent_path: TokenPath
    depth: int
    ancestor_last: tuple[bool, ...]
    is_last: bool
    hidden_count: int
    mass_nats: float


TreeRow: TypeAlias = BranchRow | EllipsisRow


@dataclass(frozen=True, slots=True)
class TreeViewport:
    """Only rows that physically fit in the current elastic tree region."""

    rows: tuple[TreeRow, ...]
    rows_above: int
    rows_below: int
    selected_path: TokenPath | None


class BuildTreeView(Protocol):
    """Shape core state into visible semantic rows, with no ANSI or terminal IO."""

    def __call__(
        self,
        snapshot: SessionSnapshot,
        state: AppState,
        *,
        height: int,
    ) -> TreeViewport: ...


class RunTui(Protocol):
    """Intended final executable boundary used by every backend example."""

    def __call__(
        self,
        cursor: Cursor,
        describe_token: DescribeToken,
        *,
        title: str,
    ) -> None: ...
