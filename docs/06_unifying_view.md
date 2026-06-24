# Unifying View

All archetypes can be seen as choosing an interaction operator and then a bag
readout.

```math
H
\xrightarrow{\mathcal C}
\widetilde H
\xrightarrow{\mathcal R}
z
\xrightarrow{\mathrm{head}}
p.
```

## Interaction Operators

Mean MIL:

```math
\widetilde h_i=\phi(h_i).
```

Attention MIL:

```math
\widetilde h_i=\phi(h_i),
\qquad
z=\sum_i a_i\widetilde h_i.
```

Transformer MIL:

```math
\widetilde h_i
=
\sum_j A_{ij}Vh_j.
```

State-space MIL:

```math
s_t=\bar A_t s_{t-1}+\bar B_t h_t,
\qquad
\widetilde h_t=C_t s_t.
```

Graph MIL:

```math
\widetilde h_i
=
\gamma
\left(
h_i,
\rho(\{\phi(h_i,h_j):j\in\mathcal N(i)\})
\right).
```

## Readout Operators

Uniform mean:

```math
z=\frac{1}{n}\sum_i\widetilde h_i.
```

Attention expectation:

```math
z=\sum_i a_i\widetilde h_i,
\qquad
a_i\ge0,
\qquad
\sum_i a_i=1.
```

Class token:

```math
z=\widetilde h_{\mathrm{cls}}.
```

Set statistics:

```math
z=
\left[
\mathrm{mean}_i\widetilde h_i
\Vert
\mathrm{max}_i\widetilde h_i
\Vert
\mathrm{var}_i\widetilde h_i
\right].
```

## What Each Archetype Preserves

| Archetype | Preserved object | Lost or weakened object |
| --- | --- | --- |
| Mean MIL | first moment | rare patch evidence |
| Attention MIL | weighted first moment | pairwise and spatial context |
| Transformer MIL | dense pairwise correlations | exact geometry unless encoded |
| State-space MIL | ordered compressed memory | permutation invariance |
| Graph MIL | local topology | all-to-all global interaction |

## The Design Equation

A new aggregator should state:

```math
\boxed{
\text{surviving statistic}
=
\mathcal R(\mathcal C(H;G,S)).
}
```

If the statistic is not clear, the architecture is probably not clear either.
