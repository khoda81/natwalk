# Architecture

## Exact layer

`Navigator` is the source of truth. It owns:

```text
causal cursor
arithmetic interval [lo, hi)
equal-information action count
committed-path surprisal
undo history
```

A K-way action narrows the arithmetic interval by exactly `1/K`. Tokens are committed only when the entire selected interval lies in one token cell.

Explicit greedy autocomplete is separate from arithmetic navigation: it commits a model-greedy path up to a bit budget and resets the unresolved interval.

## Search layer

`TokenTreeExplorer` is a cache/search engine over the model's token-prefix tree.

For token edge probability `p`:

```text
edge cost = -ln(p)
path cost = sum(edge costs) = -ln P(prefix)
```

The worker uses a global min-heap and expands the lowest information-cost prefix first: uniform-cost search / Dijkstra.

This is intentionally different from the first prototype, which prefetched the equal-mass *interaction tree* by action depth.

### Lazy sibling streams

A model prediction gives the whole vocabulary distribution at once, sorted by probability.

Putting all vocabulary children onto the heap is exact but wasteful. Instead, each expanded parent contributes only its cheapest unseen child to the heap. When that child is popped, the next sibling is queued.

This is a k-way merge of sorted child streams and produces the same Dijkstra order while keeping the frontier compact.

### Hidden residual

Unmaterialized children are retained as exact aggregate mass:

```text
hidden_mass = sum(probability of hidden relevant children)
ellipsis_cost = -ln(hidden_mass)
```

Rendering may show only a viewport of the discovered tree; search completeness and render completeness are separate concerns.

## Current-root KV safety

The MuScriptor backend uses a preallocated append-only KV cache. Its cheap checkpoint stores only logical state and streaming offsets.

Speculation is safe only when it starts from the **current committed root**: speculative writes then occur strictly after the live prefix and can be discarded by restoring offsets.

Searching from an older ancestor while a newer live prefix exists could overwrite KV slots needed by that live prefix. Therefore when a user action commits new tokens (or undo moves the root), the active search is re-rooted.

Persistent tree metadata across roots can be added later with persistent/copy-on-write KV pages or replay from independent branch state, but live-state correctness takes priority.

## Undo

For rewindable backends, each user action stores a lightweight cursor checkpoint plus arithmetic state. Older checkpoints remain valid because later search only writes after the then-current prefix.

Undo restores the previous checkpoint and re-roots Dijkstra search there.

## Resource growth

There is no semantic depth limit.

The current implementation limits *materialized tree nodes* with `max_nodes`. Expanded nodes retain their ranked distributions so hidden residuals and lazy siblings remain exact.

Future memory strategies:

1. compact ranked distributions,
2. eviction/reconstruction of cold expanded nodes,
3. CPU-offloaded branch metadata,
4. batched replay,
5. persistent/copy-on-write KV pages.

## UI separation

The UI deliberately separates three things:

```text
tree browsing      -> inspection only; no probability commitment
0/1 (or K digits) -> exact equal-information arithmetic navigation
Space              -> explicit greedy continuation under a bit budget
```

This prevents a visual "move to sibling" from pretending to cost one bit when the sibling may actually carry very different surprisal.
