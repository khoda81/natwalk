# natwalk

**Navigate autoregressive distributions in information space.**

`natwalk` is an interaction primitive for generative models where progress is measured in **nats/bits**, not tokens, characters, seconds, or pixels.

A backend exposes the complete next-token distribution and a causal cursor. natwalk can then:

- narrow the distribution by exact equal-information arithmetic-code actions,
- search the model's token tree with Dijkstra / uniform-cost search in surprisal,
- expose hidden residual probability as an exact `…` branch,
- accept a greedy continuation up to a fixed information budget,
- undo user actions without changing the probability semantics.

No top-k or top-p truncation is part of the core.

## Exact arithmetic navigation

With `K` choices, one user action selects one of `K` equal-width subintervals of the current arithmetic-code interval and therefore supplies exactly

$$
\ln K \text{ nats} = \log_2 K \text{ bits}.
$$

For binary navigation (`K = 2`), every `0`/`1` action is exactly one bit.

If the selected interval lies completely inside one model token interval, that token is forced and committed. The interval is renormalized inside that token and the process repeats.

## Token-tree search

The model tree uses token surprisal as edge cost:

$$
c(y_t) = -\ln p(y_t \mid y_{<t}).
$$

Therefore a prefix has cumulative cost

$$
g(y_{1:n}) = -\ln P(y_{1:n}).
$$

`TokenTreeExplorer` expands the smallest cumulative cost first: **Dijkstra in information distance**, not BFS by token depth and not beam search.

One model call exposes the complete ranked distribution at a prefix. Children are materialized lazily. Only the cheapest unseen sibling of each expanded parent needs to sit on the global heap; when it is popped, the next sibling is exposed. This is equivalent to inserting the whole vocabulary into the heap without the vocabulary-sized frontier blow-up.

There is no search depth limit. `max_nodes` is only a resource cap.

### Exact ellipsis

If some children are not materialized/displayed, `…` represents their aggregate probability mass:

$$
p_{\ldots} = \sum_{i \in \text{hidden}} p_i,
\qquad
c_{\ldots} = -\ln p_{\ldots}.
$$

So the ellipsis is a real probability region, not UI hand-waving.

## Budgeted greedy autocomplete

Space can accept the longest greedy continuation whose cumulative surprisal fits a budget `B`:

$$
\sum_t -\log_2 p(y_t \mid y_{<t}) \le B.
$$

A confident model may fit many tokens into two bits. An uncertain model may fit none.

```python
from natwalk import Navigator

nav = Navigator(cursor, choices=2)

suggestion = nav.greedy_suggestion(max_bits=2.0)
accepted = nav.accept_greedy(max_bits=2.0)
nav.undo()
```

Greedy acceptance is an explicit autocomplete action. It resets any partially narrowed arithmetic interval to `[0, 1)`.

## Backend API

The minimal backend is:

```python
class Cursor:
    prefix: tuple[int, ...]
    ended: bool

    def predict(self) -> Sequence[float]:
        """Return the COMPLETE normalized next-token distribution."""

    def observe(self, token: int) -> None:
        """Commit one token and advance causal state."""

    def clone(self) -> "Cursor":
        """Fork the causal state."""
```

Large transformer backends should usually also implement:

```python
def checkpoint(self) -> object: ...
def restore(self, checkpoint: object) -> None: ...
```

natwalk then speculates by rewinding lightweight cursor controls rather than cloning bulk KV tensors.

## MuScriptor demo

The first live backend is [MuScriptor](https://github.com/muscriptor/muscriptor), using its complete 1,393-symbol transcription distribution.

With sibling checkouts at `~/Projects/muscriptor` and `~/Projects/natwalk`:

```bash
uv run \
  --project ~/Projects/muscriptor \
  --with-editable ~/Projects/natwalk \
  -- python ~/Projects/natwalk/examples/muscriptor_cli.py \
  "samples/Laura Marling - What He Wrote.mp3" \
  --model medium \
  --device cuda \
  --chunk 0
```

The CLI defaults to binary navigation and a 2-bit Space suggestion:

```text
Budget: 2.00 bit    binary action: 1.00 bit

Committed:
tie · t=0.23s · program(acoustic_guitar) · note_on

Suggestion [1.87/2.00 bit]:
A2 · t=0.45s · note_off · A2 · note_on · E3

tree: 84 nodes · 31 model expansions · 12 frontier · ⟳

  ▶ ├┬ A2                              0.09 nat
  │  ├┬ t=0.45s                        0.13 nat
  │  │  ├┬ note_off                    0.18 nat
  │  │  │  └─ …  +1389 hidden          0.12 nat
  ├─ E3                                1.31 nat
  ├─ program(clean_electric_guitar)    1.72 nat
  └─ …  +1388 hidden                   2.04 nat
```

Controls:

```text
0 / 1       choose an exact binary arithmetic half
Space       accept the highlighted greedy suggestion
Backspace   undo the previous user action
[ / ]       decrease / increase suggestion budget
↑ / ↓       browse the rendered tree
← / →       collapse / expand a rendered subtree
d           toggle debug interval/cost details
q           quit
```

MuScriptor `shift` tokens are rendered as `t=...s`: they are absolute 100 Hz positions inside the current 5-second chunk, not time deltas.

The demo currently operates on one 5-second MuScriptor chunk. Cross-chunk tie/prelude state is not wired into natwalk yet.

## Why `natwalk`?

Ordinary autocomplete asks:

> how many tokens should I suggest?

natwalk asks:

> how much information should this interaction specify?

That same primitive can apply to text, code, music, speech, image tokens, or any other causal probabilistic model.
