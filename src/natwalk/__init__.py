"""Navigate autoregressive distributions in fixed-information steps."""

from .core import (
    Cursor,
    ExplorerStats,
    GreedySuggestion,
    NavigationSnapshot,
    Navigator,
    Preview,
    RankedDistribution,
    RewindableCursor,
    State,
    TokenTreeExplorer,
    TreeEntry,
    TreeExplorer,
)
from .suggestions import accept_completion, cached_budget_completions

__all__ = [
    "Cursor",
    "ExplorerStats",
    "GreedySuggestion",
    "NavigationSnapshot",
    "Navigator",
    "Preview",
    "RankedDistribution",
    "RewindableCursor",
    "State",
    "TokenTreeExplorer",
    "TreeEntry",
    "TreeExplorer",
    "accept_completion",
    "cached_budget_completions",
]
__version__ = "0.2.0"
