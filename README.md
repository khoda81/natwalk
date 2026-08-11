# natwalk

**Navigate autoregressive distributions in information space.**

`natwalk` is an interaction primitive for causal probabilistic models. A user supplies fixed-information choices while a background uniform-cost search discovers likely model continuations. Browsing the discovered tree is inspection only: it never changes the probability state being searched.

The core intentionally has no top-k/top-p truncation.

## Model interface

A backend is one rewindable causal cursor:

```python
from collections.abc import Sequence


class Cursor:
    def predict(self) -> Sequence[float]:
        """Return the complete normalized next-symbol distribution.

        Return an empty sequence at end of generation.
        """

    def observe(self, token: int) -> None:
        """Advance the causal state by one token."""

    def checkpoint(self) -> object:
        """Return a cheap rewind point."""

    def restore(self, checkpoint: object) -> None:
        """Restore a previous rewind point."""
```

There is no clone fallback and no required public prefix/ended state. Large transformer backends can keep one mutable KV allocation and checkpoint only the logical controls needed to rewind it.

## One probability tree

`Tree` is the single representation of discovered model knowledge.

An expanded node stores its complete probability-ranked distribution. Its vocabulary children are virtual: a concrete `Node` is allocated only when that child acquires a known subtree or somebody needs its identity. This lets the renderer inspect rank 50,000 without constructing 50,000 Python tree nodes.

A concrete node stores only:

```text
parent node id
rank within the parent's distribution
optional known distribution
map of concrete child ranks -> node ids
```

Tokens, edge surprisal, paths, depth, and cumulative path cost are derived.

## Dijkstra in information distance

For a token edge with conditional probability `p`:

```text
edge cost = -ln p
```

`Search` performs synchronous uniform-cost search from the committed root. Every expanded parent's children are already ordered from highest to lowest probability, therefore their edge costs are nondecreasing.

Instead of putting the whole vocabulary on the global heap, each expanded parent contributes only the head of its sorted sibling stream. When `(parent, rank)` is popped:

```text
queue (parent, rank + 1)
expand child(parent, rank)
queue (child, 0)
```

The heap is therefore a k-way merge of sorted child streams and produces the same expansion order as eager full-frontier Dijkstra. The test suite checks this against an intentionally stupid eager implementation over randomized finite probability trees.

`Search.step()` is entirely synchronous. `SearchWorker` is only a thread/lock wrapper around the same method; concurrency does not have separate search semantics.

## Session and committed state

`Session` owns:

```text
model cursor
committed trie root
checkpoint at that root
probability tree
Dijkstra frontier
```

Search speculation always restores the committed checkpoint, replays from the committed root to the candidate node, predicts, and restores the checkpoint again.

`Session.commit(tokens)` is the one token-commit path. Advancing the committed root resets the Dijkstra frontier relative to the new root but keeps already discovered trie knowledge.

Inspection is separate: `Session.inspect(node)` may cache a descendant distribution without changing either the committed cursor or the search frontier.

## Fixed-information navigation

`Navigation` owns the pending arithmetic interval and user-action undo history.

With `K` choices, one action divides the current interval into `K` equal-width pieces, corresponding conceptually to

$$
\ln K \text{ nats} = \log_2 K \text{ bits}.
$$

Tokens are committed only when the whole selected interval lies inside one token cell. If a choice forces no token, the committed root, model cursor, probability tree, and Dijkstra frontier are unchanged.

That separation is important: arithmetic narrowing is interaction state, not a search filter.

Explicit completion acceptance commits a path through the same `Session.commit()` primitive and clears the pending arithmetic interval. Undo restores model/session state only when the action actually changed the committed root; undoing arithmetic-only narrowing leaves whatever search progress happened meanwhile intact.

The current interval implementation uses Python floating point. The information semantics are exact conceptually, but the current arithmetic representation is not an integer arithmetic coder.

## View state

Tree browsing is represented as a small zipper-like `View`:

```python
View(
    node=x,
    first_rank=k,
    selected_rank=i,
)
```

Its meaning is approximately **"show children of node `x` starting at sibling rank `k`"**.

The renderer derives frame rows directly from the probability tree. Viewport clipping, sibling tails, indentation, and future corridor compression are rendering decisions rather than search state.

Browsing or materializing a child identity does not enqueue it for Dijkstra.

## Completion queries

Suggestions are read-only queries over known trie state:

```python
from natwalk import completions, greedy

suggestion = greedy(tree, root, max_nats=1.5)
options = completions(tree, root, max_nats=1.5)
```

If the search has not discovered enough of a path, a suggestion is marked incomplete. The query layer never performs model evaluation or mutates the tree.

## Minimal example

```python
from natwalk import Navigation, Session


class TableCursor:
    def __init__(self):
        self.prefix = ()
        self.table = {
            (): (0.7, 0.3),
            (0,): (0.8, 0.2),
        }

    def predict(self):
        return self.table.get(self.prefix, ())

    def observe(self, token):
        self.prefix = (*self.prefix, token)

    def checkpoint(self):
        return self.prefix

    def restore(self, checkpoint):
        self.prefix = checkpoint


session = Session(TableCursor())
navigation = Navigation(session, choices=2)

session.search.step()  # discover one Dijkstra node
forced = navigation.choose(0)
navigation.undo()
```

## Terminal UI

The generic terminal UI is part of the package rather than an example-specific module:

```python
from natwalk.tui import run_tui

run_tui(
    cursor,
    describe_token,
    title="my model · natwalk",
)
```

It depends only on the natwalk cursor contract. Model-specific adapters belong with the model project so natwalk does not acquire heavyweight backend dependencies such as PyTorch or MuScriptor.

The renderer clips by terminal display cells and guarantees each emitted row fits the reported terminal width instead of relying on Python string length.

## llama.cpp demo

`examples/llm_app.py` can resolve an existing Ollama model directly to its GGUF blob:

```bash
uv run examples/llm_app.py \
  --model ministral-3:14b \
  "The most surprising thing about information theory is"
```

The demo contains only llama.cpp-specific model adaptation and imports the shared UI from `natwalk.tui`.

MuScriptor-specific code is intentionally not carried in this repository; its natwalk command should live in MuScriptor itself and import the same `run_tui` entry point.
