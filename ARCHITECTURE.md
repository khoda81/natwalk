# Architecture

## 1. Exact navigator

The semantic state is deliberately small:

```text
causal model cursor
arithmetic interval [lo, hi)
action count
committed-path surprisal
```

A user action narrows `[lo, hi)` by an exact factor of `K`. Tokens are committed only while that whole interval lies inside one model child interval.

The backend owns tokenization, model state, and the complete next-token probability vector.

## 2. Branching API

The minimal backend API is:

```python
predict() -> complete distribution
observe(token)
clone()
```

For models with large KV caches, `clone()` is semantically convenient and often operationally terrible. Backends can additionally expose:

```python
checkpoint() -> opaque small state
restore(checkpoint)
```

`Navigator` prefers checkpoint/restore whenever available.

A preallocated append-only transformer cache is particularly friendly to this: a speculative branch writes only after the current offsets. Rewinding those offsets makes the suffix invisible again, so the next branch can overwrite it without copying the immutable prefix.

## 3. Background tree explorer

`TreeExplorer` owns a single background inference thread around an exact `Navigator`.

It precomputes previews for future **action paths** such as:

```text
(0,)
(1,)
...
(4, 0)
(4, 1)
...
```

A path means “apply all earlier bucket choices exactly, then preview the final bucket.” This is useful because a cached path `(a, b)` becomes the immediately useful preview `(b,)` if the user chooses `a`.

For a `K`-ary interaction, every action region at depth `d` has exact mass

$$
K^{-d}.
$$

So probability-mass-first scheduling over action regions is simply shallow-first. Equal-mass paths are tie-broken lexicographically, which explores modeward buckets before the residual bucket.

Only paths ending in visible buckets need a rendered preview. Residual buckets still appear inside path prefixes so their descendants are prefetched.

## 4. Selection, pruning, and rebasing

Suppose the cache contains:

```text
(0,)      preview A
(1,)      preview B
(0, 0)    preview AA
(0, 1)    preview AB
(1, 0)    preview BA
...
```

If the user selects bucket `0`:

1. The exact navigator commits that equal-information action.
2. The generation epoch increments.
3. Every cached path not beginning with `0` is discarded.
4. Descendant paths are rebased:

```text
(0, 0) -> (0,)
(0, 1) -> (1,)
```

5. Stale in-flight work from the old epoch is ignored when it completes.
6. Missing descendants are scheduled under the new root.

The worker never decides semantics. Its previews are disposable caches; the arithmetic interval is always authoritative.

## 5. Concurrency model

Model inference and live `choose()` calls share one compute lock. This intentionally serializes accelerator access for stateful backends.

The worker may temporarily walk and rewind the real cursor, so the UI must not inspect that mutable cursor directly while inference is happening. `TreeExplorer` therefore publishes an immutable `NavigationSnapshot` containing only:

```text
committed prefix
ended flag
[lo, hi)
action count
path surprisal
```

The UI can repaint from the snapshot and cached previews without touching speculative model state.

## 6. MuScriptor backend

MuScriptor's streaming transformer preallocates KV tensors and controls visible history using offsets. The natwalk example checkpoints:

```text
prefix / ended
current input token
first-step flag
cached probability vector
streaming offset/control state
```

It deliberately does **not** copy tensors named `cache`.

Before a checkpoint, `predict()` ensures the current conditioning/input has been written into KV. A speculative branch then advances offsets and writes later slots. Restoring the saved offsets hides that suffix; the next branch overwrites it.

This replaces repeated large KV tensor clones with tiny control-state copies.

## 7. Memory strategy

The current explorer caches preview metadata, not one full backend cursor per tree node. That keeps the generic tree cheap and avoids turning speculative depth into VRAM growth.

More advanced backends can later add:

```text
persistent / paged KV
batched branch prediction
CPU-offloaded checkpoints
parent + suffix replay
fork_many(tokens)
```

Those are performance extensions only. They must preserve the exact same complete distribution and causal semantics.

## 8. UI rendering

Raw model tokens are still only a debugging representation. Presentation should remain separate:

```text
model token path
    ↓
domain event decoder
    ↓
renderable continuation
```

For MuScriptor, the next useful layer is a compact piano-roll/staff-like rendering of each preview while retaining the exact arithmetic interval behind the option.
