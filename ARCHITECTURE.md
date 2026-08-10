# Architecture

natwalk is split by ownership. Each piece of mutable state has one reason to exist and one owner.

## Sources of truth

```text
model execution
    Cursor + committed checkpoint

committed generation state
    Session.root

known probability structure
    Tree

search scheduling
    Search.frontier

pending user information + action history
    Navigation

inspection state
    View
```

The design goal is not merely separation into classes. It is to avoid keeping two mutable representations of the same fact.

## Cursor

The model boundary is deliberately small:

```text
predict() -> complete normalized distribution
observe(token)
checkpoint()
restore(checkpoint)
```

An empty distribution means terminal.

There is no `clone()` capability and no required `prefix` or `ended` field. Transformer implementations are expected to retain one mutable KV allocation and make checkpoint/restore cheap.

## Probability tree

`Tree` is an arena of concrete nodes. Node identity is an integer.

A node contains:

```text
parent: NodeId | None
rank: int
known distribution: Distribution | None
concrete children: rank -> NodeId
```

An expanded node's `Distribution` contains the complete next-symbol probabilities in descending probability order.

All vocabulary children therefore exist virtually as `(parent, rank)` even if no `Node` object has been allocated for them. A concrete child node is created only when a subtree or stable identity is needed.

The tree does **not** store:

```text
token                derived from parent distribution + rank
edge surprisal       derived as -ln(probability)
path                  derived through parent links
depth                 derived through parent links
cumulative path cost  accumulated by the consumer
expanded flag         distribution is None vs known
ended flag            known distribution is empty
render rows           derived per frame
```

## Search

`Search` is synchronous uniform-cost search over the token-prefix tree.

For an expanded parent `x`, its child probabilities satisfy

```text
p(x, 0) >= p(x, 1) >= p(x, 2) >= ...
```

so child edge costs satisfy

```text
-ln p(x, 0) <= -ln p(x, 1) <= -ln p(x, 2) <= ...
```

Each expanded parent is therefore a sorted stream of Dijkstra candidates.

The global frontier contains only the head of every active stream:

```text
Candidate(path_nats, parent, rank)
```

When a candidate is popped:

```text
1. queue the next sibling from the same parent
2. materialize the popped child identity
3. evaluate that child if its distribution is unknown
4. queue rank zero of the child's distribution
```

This is a k-way merge of sorted child streams. Tests compare it with eager Dijkstra that inserts every child and assert equal pop paths and costs over randomized finite trees.

### Rerooting

Search cost is relative to the current committed root. When the committed root changes, `Search.reset(root)` discards only the scheduling frontier and seeds a new root-relative frontier.

The probability tree is retained. If nodes under the new root were already discovered, their distributions are reused when Dijkstra reaches them.

There is no arithmetic interval in Search and no concept of viewport relevance.

## Session

`Session` joins one cursor to one tree/search pair.

Its semantic state is:

```text
cursor
committed root
cursor checkpoint at committed root
tree
search frontier
```

### Search evaluation

To evaluate a candidate node:

```text
restore committed checkpoint
replay tokens from committed root to candidate
predict complete distribution
restore committed checkpoint
```

The committed cursor is unchanged by speculation.

### Inspection

`Session.inspect(node)` uses the same replay mechanism to fill a node's distribution cache, but does not add or remove anything from the search frontier.

This lets the human inspect arbitrary known/virtual branches independently of Dijkstra scheduling.

### Commit

`Session.commit(tokens)` is the only operation that advances committed model state.

For each token it advances the trie root and cursor. At the end it takes one new checkpoint and resets search relative to the new root.

Session itself does not define user actions or own undo history.

## Navigation

`Navigation` owns two things:

```text
pending arithmetic interval [lo, hi)
user-action undo history
```

One K-way choice narrows the interval to one of K equal-width pieces. After narrowing, the current root distribution is examined. If both interval endpoints fall in the same token cell, that token is forced, the interval is renormalized within that cell, and the token is committed. This repeats while additional tokens are forced.

If no token is forced:

```text
Session.root       unchanged
Cursor             unchanged
Tree               unchanged
Search.frontier    unchanged
```

That invariant is tested directly.

### Undo

History records the previous arithmetic state and the previous committed Session checkpoint.

If an action did not change `Session.root`, undo restores only arithmetic state. Search may have progressed while the arithmetic interval was narrowed and that progress is retained.

If an action committed tokens, undo restores the previous committed cursor/root and reroots search there while retaining tree knowledge.

### Explicit acceptance

Greedy or alternate completion acceptance is not a separate mutation mechanism. A query produces a token path and `Navigation.accept(path)` commits it through the same `Session.commit()` operation.

## Queries

`greedy()` and `completions()` are read-only traversals over the known trie.

They never:

```text
call the model
acquire model execution state
materialize virtual children
change the search frontier
```

If a path reaches unknown trie state, the returned suggestion is marked incomplete.

## View and rendering

`View` is intentionally tiny:

```text
node
first_rank
selected_rank
```

It represents the sibling-tail forest beginning at `children(node)[first_rank:]` plus a selected sibling.

Rendering derives rows from `Tree + View + frame dimensions`. Virtual children can be rendered directly from the parent's complete distribution without allocating tree nodes.

Browsing may materialize a selected child's identity and `Session.inspect()` may fill its distribution, but neither operation changes Dijkstra scheduling.

The committed root is the lower navigation boundary for a live session. Historical ancestors may remain cached in the persistent tree after commits/undo, but they are not reinterpreted as descendants of the current model checkpoint.

## Background execution

`SearchWorker` wraps synchronous `Search.step()` with one condition/lock and one thread.

It owns no semantic search state. In particular it has no:

```text
second frontier
pending-node graph
generation counter
range relevance cache
```

Foreground UI code briefly acquires the same lock to inspect or mutate Session/Navigation state, then releases it while waiting for user input so background search can continue.

## Numerical representation

The arithmetic navigation interval currently uses Python `float`.

The conceptual K-way information accounting is exact, but token-cell containment and renormalization inherit binary64 numerical behavior. Replacing this with an integer/range-coded representation is independent of the search/tree architecture and remains future work.

## Performance work intentionally deferred

The ownership model is designed so these optimizations can be added without changing search semantics:

1. compact/array-backed ranked distributions for very large vocabularies,
2. faster token->rank lookup for path acceptance,
3. checkpointing selected hot trie nodes rather than replaying every path from the root,
4. persistent/copy-on-write KV pages,
5. batched candidate evaluation,
6. resource limits/eviction for long-running background search.

The synchronous eager-vs-lazy equivalence tests remain the reference correctness oracle underneath those optimizations.
