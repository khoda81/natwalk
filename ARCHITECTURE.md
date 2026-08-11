# Architecture

Natwalk is organized around ownership: each mutable fact has one source of truth.

## Sources of truth

```text
model execution            Cursor + committed checkpoint
causal generation state    Session.root
known probability state    Tree
search scheduling          Search.frontier
interactive command order  Engine command queue + history
client inspection state    View
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

Natwalk trusts this contract. It does not silently renormalize model output. There is no clone fallback and no required public prefix/ended state. Transformer implementations can retain one mutable KV allocation and checkpoint only the logical controls needed to rewind it.

## Distribution and Tree

A `Distribution` contains the complete next-symbol probabilities in descending probability order.

`Tree` is an arena-backed append-only trie. A node contains:

```text
parent: NodeId | None
rank: int
distribution: Distribution
children: rank -> NodeId
```

A node is published only after its complete distribution is known. There is no incomplete-node state.

All vocabulary children exist virtually as `(parent, rank)`. `tree.child(parent, rank)` is a read-only lookup; `tree.put_child(parent, rank, distribution)` is the publication boundary. Repeating an identical publication is an idempotent no-op. Publishing conflicting contents for an existing edge is an invariant violation.

The tree does not duplicate facts that can be derived cheaply:

```text
token                 parent distribution + rank
edge surprisal        -ln(probability)
path                   parent links
cumulative path cost   accumulated from edge costs
terminal state         empty distribution
```

## Search

`Search` is synchronous uniform-cost search over virtual child edges.

For a parent `x`, ranked probabilities satisfy

```text
p(x, 0) >= p(x, 1) >= p(x, 2) >= ...
```

therefore sibling edge costs are already nondecreasing:

```text
-ln p(x, 0) <= -ln p(x, 1) <= -ln p(x, 2) <= ...
```

The global heap needs only one candidate from each active sibling stream:

```text
Candidate(path_nats, parent, rank)
```

Advancing one candidate does exactly this:

```text
1. pop the lowest-cost (parent, rank)
2. queue (parent, rank + 1)
3. reuse the child if already discovered, otherwise evaluate and publish it
4. queue rank 0 of that child's distribution
```

This is a k-way merge of sorted sibling streams. Randomized tests compare it with eager Dijkstra that inserts every child and assert equal pop paths and costs.

`Search.step()` performs exactly one transition. `Search.discover()` performs the same transitions but fast-forwards already-known edges until it publishes one genuinely new node or exhausts the frontier. The latter avoids replay churn after rerooting while preserving the same search order.

Search has no viewport relevance and no arithmetic interval. Background exploration depends only on the causal root and cumulative surprisal.

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

### Speculative evaluation

To evaluate a virtual child `(parent, rank)`:

```text
restore committed checkpoint
replay known tokens from committed root to parent
observe the selected token
predict the complete child distribution
restore committed checkpoint
```

Speculation therefore never changes committed model state.

### Commit and restore

`Session.commit(tokens)` is the mutation path for causal advancement. Existing discovered children are reused; unknown children are predicted and published complete. If the root changes, Session checkpoints the new causal state and resets search relative to that root.

`Session.restore(checkpoint)` restores cursor/root state and resets search there without deleting any discovered tree knowledge.

`Session.inspect_child(parent, rank)` may explicitly discover one child without changing the committed root.

## Optional fixed-information Navigation

`Navigation` is a library layer above `Session`, not the TUI's state model.

It owns an equal-width arithmetic interval and action history. A K-way choice narrows the interval to one of K equal pieces. Tokens are committed only when the entire selected interval lies inside one token cell; otherwise the Session root, cursor, tree, and search frontier remain unchanged.

Explicit acceptance commits through the same `Session.commit()` path. Undo restores Session state only when the corresponding action changed the causal root.

## Process engine

The interactive app isolates model execution in a spawned process.

The child process owns:

```text
Session
Cursor
Search
causal checkpoint history
```

Commands are ordered:

```text
Advance(command_id, tokens)
Rewind(command_id)
Stop()
```

`Advance` may contain multiple tokens but stores one checkpoint per token, so `Rewind` always moves one trie edge.

The worker drains queued commands FIFO before returning to background search. This gives the command queue a simple authoritative meaning: after all queued commands complete, the last command target is the causal truth.

## Tree synchronization

The UI does not share the worker's mutable tree object. It reconstructs an append-only `TreeReplica` from `NodeUpdate` records.

`NodeId` doubles as append-log position. A replica therefore needs only its next unseen node id to resume synchronization.

Applying updates is strict:

```text
old duplicate      verify exact contents, then no-op
next expected id   append
future id          fail: missing update gap
conflict           fail
```

This gives reconnect/replay idempotence without a second revision counter.

## Optimistic causal navigation

The TUI has one logical causal cursor: the last queued navigation target.

For an already-discovered child, RIGHT can move the visible cursor immediately and enqueue the authoritative `Advance` behind earlier commands. Several known moves may therefore be queued without waiting for the worker.

LEFT/Backspace likewise queues one-edge `Rewind` and moves the visible cursor to the known parent immediately.

An undiscovered child is a natural queue barrier: the command is sent, but the UI cannot move into a node whose distribution/identity is not yet in the replica.

Intermediate worker state never snaps the UI backward. When the queue drains, the visible cursor and authoritative engine root must converge exactly; disagreement is an error, not something to reconcile silently.

## View

Client-local browsing state is intentionally tiny:

```python
View(
    node=x,
    first_rank=k,
    selected_rank=i,
)
```

It identifies the causal/view root plus sibling-tail selection. It does not schedule search.

## Probability partition

The renderer first decides **which disjoint events deserve rows**, then separately decides how to draw their shared tree geometry.

With a row budget `N`, `partition_rows()` starts from one event representing the visible sibling tail. Each refinement replaces one event with two disjoint events, increasing the row count by exactly one. Candidate refinements currently compete by the probability of their smaller result, preventing extremely unlikely deviations from consuming rows while more probable unresolved alternatives remain elsewhere.

The partition conserves visible probability mass. No renderer-created side branch receives a free row.

## Leaf-only radix layout

Once the event set is fixed, selected paths are ordered in trie order and shared prefixes are factored like a radix/Patricia trie.

The hard layout rule is:

> one semantic probability event = one physical row.

Internal trie structure can consume columns but never additional rows. Long unary paths collapse horizontally. Branches attach at the exact display column of the token boundary where selected events diverge.

Structural connector brightness uses aggregate probability mass of the displayed events beneath that branch. Right-hand nat values remain the exact cumulative surprisal of each row event from the visible causal cursor.

Terminal widths are computed from unpainted Unicode display cells; ANSI styling is applied only after geometry is known.

## Queries

`greedy()` and `completions()` are read-only traversals over already-discovered tree state. They never call the model, materialize unknown children, or mutate search scheduling. If known state ends before a requested completion does, the returned suggestion is marked incomplete.
