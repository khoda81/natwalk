# natwalk

**Navigate autoregressive distributions in fixed-information steps.**

`natwalk` is an interaction primitive for generative models where the unit of progress is **information**, not tokens, characters, seconds, or pixels.

Given a complete autoregressive distribution and `K` visible choices, each action selects one of `K` equal-width arithmetic-code intervals. Every action therefore contributes exactly

$$
\ln K \text{ nats} = \log_2 K \text{ bits}.
$$

For the default `K = 5`, one action is exactly `ln(5) = 1.609438` nats (`2.321928` bits).

A confident model can commit a long continuation after one action. An uncertain model may need several actions before even the next token is determined. **Length adapts to surprisal.**

## The UI idea

At each step, natwalk lays out the model's *entire* next-token distribution in descending probability order and partitions the currently unresolved arithmetic interval into equal-cost buckets:

```text
[1] high-probability continuation
[2] another continuation
[3] another continuation
[4] another continuation
[5] …   residual / none of the above
```

The last bucket is not top-k truncation. It is an exact equal-mass residual interval. Selecting `…` zooms into that region and repartitions it again.

There are two kinds of displayed continuation:

- **FORCES** — every code point in the selected interval agrees on these tokens, so they are safe to commit.
- **preview** — one representative arithmetic-code point from the bucket, useful for display but never silently treated as the whole bucket.

This distinction keeps navigation complete: no probability mass is thrown away for UI convenience.

## Core invariant

Let the unresolved arithmetic interval be `[l, h)` and let the user choose bucket `j` out of `K`. The new interval is

$$
\left[
 l + \frac{j}{K}(h-l),
 l + \frac{j+1}{K}(h-l)
\right).
$$

If that whole interval lies inside one model token's arithmetic interval `[a, a+p)`, that token is forced. Commit it and renormalize:

$$
[l,h) \leftarrow
\left[
\frac{l-a}{p},
\frac{h-a}{p}
\right).
$$

Repeat until the unresolved interval straddles multiple next-token cells.

The committed model path has surprisal

$$
-\sum_t \ln p(y_t \mid y_{<t}, x),
$$

while the information supplied by the user is exactly

$$
N_{actions}\ln K.
$$

Those are close but not identical in finite discrete trees because base-`K` bucket boundaries need not align with model-token interval boundaries—the usual arithmetic-coding alignment overhead.

## Backend API

The core is model-agnostic. A backend only needs a clonable causal cursor:

```python
class Cursor:
    prefix: tuple[int, ...]
    ended: bool

    def predict(self) -> Sequence[float]:
        """The COMPLETE next-token distribution."""

    def observe(self, token: int) -> None:
        """Commit one symbol and advance model state."""

    def clone(self) -> "Cursor":
        """Fork the causal state for another walker."""
```

Then:

```python
from natwalk import Navigator

nav = Navigator(cursor, choices=5)

preview = nav.preview(0)  # does not mutate state
forced = nav.choose(0)  # exactly ln(5) nats of user information
```

`predict()` must expose the **complete normalized distribution**. natwalk deliberately has no top-k or top-p approximation in its core semantics.

## MuScriptor demo

The first backend is [MuScriptor](https://github.com/muscriptor/muscriptor), using its full 1,393-symbol transcription distribution and streaming KV state.

If you already have a MuScriptor checkout/environment:

```bash
# inside the MuScriptor virtualenv
uv pip install -e /path/to/natwalk

uv run /path/to/natwalk/examples/muscriptor_cli.py \
  "samples/Laura Marling - What He Wrote.mp3" \
  --model medium \
  --device cuda \
  --chunk 0
```

The demo currently navigates one 5-second chunk. Chunk 0 matches MuScriptor's normal start-of-stream conditioning. Cross-chunk tie/prelude state is not wired into natwalk yet.

> The adapter currently uses MuScriptor private APIs (`_compute_logits`, `_build_conditions`, streaming model state) because the public transcription API does not expose the complete distribution or a clonable cursor.

## Why `natwalk`?

Ordinary autocomplete asks: *how many tokens should I suggest?*

natwalk asks: *how much information should one action specify?*

That makes the same interaction meaningful for text, code, music transcription, speech, image tokens, or any other causal probabilistic model.

## Current status

Early prototype. The exact arithmetic navigation is working; the MuScriptor CLI is the first live demo.

Next engineering target: **probability-first speculative expansion**. A background worker should keep expanding the highest-mass unresolved descendants, cache their model states/previews, and prune everything outside the selected interval after each user action. See [`ARCHITECTURE.md`](ARCHITECTURE.md).