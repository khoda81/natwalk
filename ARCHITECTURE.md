# Architecture

## 1. Exact navigator

The core owns only four semantic pieces of state:

```text
causal model cursor
arithmetic interval [lo, hi)
action count
committed-path surprisal
```

A user action narrows `[lo, hi)` by an exact factor of `K`. Tokens are committed only while that whole interval lies inside one model child interval.

The model backend owns tokenization, model state, and the complete next-token probability vector.

## 2. Why preview generation is currently slow

A visible menu needs several representative continuations. Each representative is a separate walker through the autoregressive tree, which means repeated model inference and branching KV state.

Generating all visible walkers synchronously puts model latency directly on the interaction path. That is unnecessary: most of the tree can be explored before the user acts.

## 3. Probability-first prefetch worker

The intended runtime has one inference worker maintaining a frontier of speculative nodes.

Each node stores at least:

```text
parent
selected token / path suffix
arithmetic interval
log probability mass
forced prefix
preview metadata
backend cursor state (or a way to reconstruct it)
generation epoch
```

The worker keeps a max-priority queue ordered by probability mass (equivalently minimum surprisal):

```text
priority(node) = log P(node)
```

It repeatedly expands the highest-mass unresolved node first. This naturally spends compute on the continuations most likely to appear in the next menu.

## 4. Selection and pruning

When the user selects an exact bucket interval `R`:

1. Intersect the speculative tree with `R`.
2. Drop every node whose arithmetic region is disjoint from `R`.
3. Renormalize retained descendant intervals into the selected region.
4. Advance the live exact navigator and bump a generation epoch.
5. Reuse already-expanded descendants whenever possible.
6. Resume probability-first expansion under the retained subtree only.

The worker must never make semantic decisions for the exact navigator. Prefetch results are caches; the interval arithmetic remains the source of truth.

## 5. KV-state memory is the hard part

Naively cloning a full preallocated transformer KV cache for every speculative node is fast to implement and terrible at scale, especially on a small GPU.

Useful backend strategies, in increasing sophistication:

1. **Active-spine GPU cache** — keep GPU state only for the live path and immediate visible walkers; reconstruct deeper nodes on demand.
2. **CPU-offloaded branch state** — move inactive KV states to pinned host memory and restore likely branches.
3. **Parent + suffix replay** — store probabilities/path suffixes and replay from the nearest cached ancestor instead of storing every KV tensor.
4. **Copy-on-write/persistent KV pages** — share immutable prefix pages between walkers and allocate only divergent suffix pages. This is the ideal backend API for large speculative trees.

The generic natwalk core should not require any one strategy. Cursor cloning is the semantic interface; backends can optimize it independently.

## 6. UI rendering

Raw model tokens are useful for debugging but not for humans. Presentation should be a separate layer:

```text
model token path
    ↓
domain event decoder
    ↓
renderable continuation
```

For MuScriptor that likely means decoding preview paths into notes/instruments/timing and rendering a compact piano roll or staff-like view. The exact arithmetic interval remains attached to each rendered option so presentation never changes probability semantics.

## 7. Future backend API

The minimal API is:

```python
predict() -> complete distribution
observe(token)
clone()
```

For efficient background expansion, backends may additionally expose capabilities such as:

```python
fork_many(tokens)
snapshot()
restore(snapshot)
replay(tokens)
batch_predict(cursors)
```

These are performance extensions only. They must preserve the same exact distribution and causal state semantics as the minimal cursor.
