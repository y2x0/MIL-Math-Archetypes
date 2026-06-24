# Attention MIL

Attention MIL keeps the bag unordered but replaces uniform patch weights with a
learned probability measure over patches.

## Forward Map

Start with patch embeddings:

```math
h_i\in\mathbb{R}^{d}.
```

A standard attention scorer computes:

```math
\ell_i=w^\top\tanh(Vh_i),
```

and a gated scorer computes:

```math
\ell_i
=
w^\top
\left[
\tanh(Vh_i)
\odot
\sigma(Uh_i)
\right].
```

Normalize over the bag:

```math
a_i
=
\frac{\exp(\ell_i)}
{\sum_{j=1}^{n}\exp(\ell_j)},
\qquad
\sum_i a_i=1.
```

Read out the bag:

```math
z=\sum_{i=1}^{n}a_i h_i.
```

Classify:

```math
\eta=W_o z+b_o,
\qquad
p=\mathrm{softmax}(\eta).
```

## Measure-Theoretic View

Attention learns a discrete probability measure:

```math
\mu_X(i)=a_i.
```

The bag vector is an expectation:

```math
z
=
\mathbb E_{i\sim\mu_X}[h_i].
```

Mean pooling is the special case:

```math
\mu_X(i)=\frac{1}{n}.
```

## Loss

```math
\mathcal L_{\mathrm{CE}}
=
-\sum_c y_c\log p_c.
```

## Mathematical Innovation

Attention MIL replaces a fixed empirical measure with a learned one:

```math
\frac{1}{n}
\quad\longrightarrow\quad
a_i(H;\theta).
```

The model still returns a convex combination:

```math
z\in\mathrm{conv}\{h_1,\ldots,h_n\}.
```

## Inductive Bias

- The bag is unordered.
- A scalar relevance score is enough per patch.
- Important patches can be selected without modeling patch-patch interaction.
- The final bag vector is a weighted first moment.

## Failure Mode

Attention weights are not signed evidence. With a linear head:

```math
\eta_c
=
b_c+\sum_i a_i\,w_c^\top h_i.
```

A high \(a_i\) only means the patch strongly affects \(z\). It does not by
itself tell whether the patch supports or opposes class \(c\).

Softmax competition also makes the model conserve attention mass:

```math
\sum_i a_i=1.
```

This can underrepresent multi-focal or diffuse patterns.
