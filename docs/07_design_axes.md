# Design Axes

This note is a checklist for inventing a new MIL aggregator without losing the
mathematical plot.

## Axis 1: Invariance

Fully permutation invariant:

```math
f(H)=f(\pi H).
```

Geometry-conditioned:

```math
f(H,G)=f(\pi H,\pi G).
```

Sequence-conditioned:

```math
f(H,S)\ne f(\pi H,S).
```

Question: should patch order matter, and if yes, where did the order come from?

## Axis 2: Interaction Range

No interaction:

```math
\widetilde h_i=\phi(h_i).
```

Sparse local interaction:

```math
\widetilde h_i=\phi(h_i,\{h_j:j\in\mathcal N(i)\}).
```

Dense global interaction:

```math
\widetilde h_i=\phi(h_i,\{h_j:j=1,\ldots,n\}).
```

Compressed global interaction:

```math
s_t=\Psi(s_{t-1},h_t),
\qquad
\widetilde h_t=\Omega(s_t,h_t).
```

## Axis 3: Weight Normalization

Uniform:

```math
a_i=\frac{1}{n}.
```

Softmax:

```math
a_i=\frac{\exp(\ell_i)}{\sum_j\exp(\ell_j)}.
```

Independent sigmoid:

```math
g_i=\sigma(\ell_i),
\qquad
z=\frac{\sum_i g_i h_i}{\epsilon+\sum_i g_i}.
```

Softmax conserves mass. Sigmoid gates do not. This matters for burden.

## Axis 4: Bag Statistics

First moment:

```math
\mu=\frac{1}{n}\sum_i h_i.
```

Second moment:

```math
\Sigma=\frac{1}{n}\sum_i(h_i-\mu)(h_i-\mu)^\top.
```

Extreme evidence:

```math
m=\max_i s_i.
```

Burden:

```math
b=\sum_i \mathbf 1[s_i>\tau].
```

Heterogeneity:

```math
\mathrm{Var}_i(s_i).
```

Question: does the task depend on presence, amount, distribution, or context?

## Axis 5: Failure Mode to Design Against

Mean pooling fails when:

```math
k/n\ll1.
```

Attention fails when one normalized map must encode multiple phenotypes:

```math
\sum_i a_i=1.
```

Transformers fail when pairwise freedom exceeds supervision:

```math
n^2\text{ interactions} \gg \text{labeled bags}.
```

State-space models fail when serialization is arbitrary:

```math
S\text{ is not a property of the tissue}.
```

Graph models fail when topology is wrong or too local:

```math
\mathcal N(i)\text{ misses the relevant context}.
```

## A Good Next Aggregator Template

One mathematically clean next step after mean pooling is gated burden-aware MIL:

```math
g_i=\sigma(w_g^\top\phi(h_i)),
```

```math
z_{\mathrm{mean}}
=
\frac{\sum_i g_i\phi(h_i)}
{\epsilon+\sum_i g_i},
```

```math
z_{\mathrm{burden}}
=
\frac{1}{n}\sum_i g_i,
```

```math
z
=
[z_{\mathrm{mean}}\Vert z_{\mathrm{burden}}],
\qquad
p=\mathrm{softmax}(Wz+b).
```

This changes only the readout, keeps permutation invariance, and makes disease
burden explicit instead of hiding it inside a normalized attention vector.
