# MIL Math Archetypes

A math-only notebook for deep multiple-instance learning and slide-level
aggregation.

This repository intentionally contains no training scripts, no datasets, no
checkpoints, and no model zoo. It studies the mathematical shape of aggregation
archetypes:

- mean-pooling MIL
- attention-based MIL
- transformer-style MIL
- state-space MIL
- graph MIL

The goal is to make aggregator design legible. Each note answers the same
questions:

- What is the forward map from patch embeddings to bag prediction?
- What statistic survives the aggregation?
- What loss supervises the bag?
- What inductive bias is being assumed?
- What failure mode follows directly from the math?

## Reading Order

1. [Common notation](docs/00_common_notation.md)
2. [Mean-pooling MIL](docs/01_mean_pooling_mil.md)
3. [Attention MIL](docs/02_attention_mil.md)
4. [Transformer MIL](docs/03_transformer_mil.md)
5. [State-space MIL](docs/04_state_space_mil.md)
6. [Graph MIL](docs/05_graph_mil.md)
7. [Unifying view](docs/06_unifying_view.md)
8. [Design axes](docs/07_design_axes.md)

## One-Line Taxonomy

Let a slide be a bag of patch embeddings:

```math
H=\{h_1,\ldots,h_n\},\qquad h_i\in\mathbb{R}^d.
```

Every MIL aggregator can be written schematically as:

```math
\widetilde H
=
\mathcal C(H;G,S),
\qquad
z
=
\mathcal R(\widetilde H),
\qquad
p
=
\mathrm{softmax}(Wz+b).
```

Here:

- `C` is the context operator.
- `G` is optional geometry, such as grid coordinates or graph edges.
- `S` is optional sequence order.
- `R` is the bag readout.

The archetypes differ mostly in the choice of `C` and `R`.

| Archetype | Context operator | Readout | Surviving statistic |
| --- | --- | --- | --- |
| Mean MIL | identity or per-patch MLP | arithmetic mean | first moment |
| ABMIL | identity or per-patch MLP | learned probability measure | weighted first moment |
| Transformer MIL | dense token mixing | class token or attention pooling | pairwise correlations |
| State-space MIL | selective recurrent scan | attention or terminal/readout pooling | compressed ordered memory |
| Graph MIL | sparse message passing | global pooling | local topology and contextualized nodes |

## Why This Exists

Most MIL papers present their architectures as code blocks and benchmark tables.
For designing a new aggregator, the useful object is usually smaller:

```math
\text{which interactions are allowed,}
\qquad
\text{which statistics survive,}
\qquad
\text{which invariances are enforced or broken.}
```

This repo keeps those choices explicit.

## References

See [references/papers.md](references/papers.md).

## GitHub Math Compatibility

GitHub's Markdown renderer rejects some LaTeX macros inside math blocks. In
particular, avoid:

```tex
\operatorname{softmax}
```

Use this style instead:

```tex
\mathrm{softmax}
```

Also avoid repository-wide custom macros such as `\newcommand`, since GitHub
does not consistently preserve them across Markdown files.
