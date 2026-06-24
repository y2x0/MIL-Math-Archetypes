# Common Notation

A whole-slide image is represented as a bag of patch embeddings:

```math
X=\{x_1,\ldots,x_n\},
\qquad
x_i\in\mathbb{R}^{D_{\mathrm{in}}}.
```

A frozen or trainable patch encoder gives:

```math
h_i=f_\psi(x_i)\in\mathbb{R}^{d}.
```

Stack the patch embeddings row-wise:

```math
H=
\begin{bmatrix}
h_1^\top\\
\vdots\\
h_n^\top
\end{bmatrix}
\in\mathbb{R}^{n\times d}.
```

The bag label is:

```math
y\in\{0,1\}^C,
\qquad
\sum_{c=1}^{C}y_c=1.
```

The final classifier produces:

```math
\eta=W_o z+b_o\in\mathbb{R}^{C},
\qquad
p_c=\frac{\exp(\eta_c)}
{\sum_{r=1}^{C}\exp(\eta_r)}.
```

The standard bag-level cross-entropy loss is:

```math
\mathcal L_{\mathrm{CE}}
=
-\sum_{c=1}^{C}y_c\log p_c.
```

For binary classification, one may equivalently use one logit:

```math
\theta(X)=P(Y=1\mid X)=\sigma(\eta),
```

with:

```math
\mathcal L_{\mathrm{BCE}}
=
-y\log\theta(X)-(1-y)\log(1-\theta(X)).
```

## Four Mathematical Slots

It helps to decompose an aggregator into four slots:

```math
\widetilde H = \mathcal C(H;G,S),
```

```math
z=\mathcal R(\widetilde H),
```

```math
\eta=W_o z+b_o,
```

```math
p=\mathrm{softmax}(\eta).
```

The slots are:

- `H`: patch embeddings.
- `G`: geometry, such as coordinates, grid, or graph.
- `S`: sequence order.
- `C`: context operator that updates patch embeddings.
- `R`: readout operator that turns a set or sequence into one bag vector.

Mean pooling makes `C` trivial. Attention changes `R`. Transformers change `C`
through dense pairwise interaction. State-space models change `C` through an
ordered recurrence. Graph models change `C` through sparse local neighborhoods.

## Invariance Vocabulary

A bag readout is permutation invariant when:

```math
\mathcal R(h_{\pi(1)},\ldots,h_{\pi(n)})
=
\mathcal R(h_1,\ldots,h_n)
```

for every permutation \(\pi\).

Mean pooling and attention pooling are invariant if their scores are computed
independently per instance. Transformers with positional encodings,
state-space scans, and graph models generally break or condition this
invariance by introducing order or geometry.
