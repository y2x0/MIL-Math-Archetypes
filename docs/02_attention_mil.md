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

Here $w\in\mathbb{R}^{d'}$ is the score direction and $V,U\in\mathbb{R}^{d'\times d}$.
The non-gated form $\ell_i=w^\top\tanh(Vh_i)$ passes every patch through the same
bounded feature map. The gate $\sigma(Uh_i)\in(0,1)^{d'}$ adds a
*data-dependent* mask: each coordinate of the feature map can be suppressed or
kept depending on the patch itself, which lets a single scorer express
"feature $j$ only matters when feature $k$ is also present." Formally the score
becomes bilinear in the two projections rather than linear in one, giving the
attention head strictly more capacity at negligible cost.

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

## Gradient of the Attention Weights

It is worth seeing exactly how supervision reaches the attention scores, because
this is what mean pooling cannot do. With a linear head and cross-entropy, the
error at the bag vector is

```math
\frac{\partial\mathcal L}{\partial z}=W_o^\top(p-y).
```

Since $z=\sum_i a_i h_i$, the sensitivity to a single weight is
$\partial z/\partial a_i=h_i$, and the softmax Jacobian is
$\partial a_i/\partial \ell_j=a_i(\delta_{ij}-a_j)$. Composing them, the gradient
on the raw score of patch $j$ is

```math
\frac{\partial\mathcal L}{\partial \ell_j}
=
a_j\,\Big\langle W_o^\top(p-y),\ h_j-z\Big\rangle.
```

Read this term by term. The factor $a_j$ means patches that already receive
little attention receive little gradient (rich-get-richer). The factor
$h_j-z$ is the patch's deviation from the current bag mean: attention grows on
patches whose embedding points in the loss-reducing direction *relative to the
current readout*, and shrinks on the rest. Unlike mean pooling — where every
patch got the identical signal $\tfrac1n W_o^\top(p-y)$ — the signal here is
patch-specific, which is precisely the mechanism that lets ABMIL find a rare
positive phenotype.

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

Finally, the scores themselves are not identifiable: softmax is invariant to a
constant shift,

```math
a_i(\ell_1+c,\ldots,\ell_n+c)=a_i(\ell_1,\ldots,\ell_n),
```

so only score *differences* are learned. Reading absolute attention values as an
importance scale is therefore not justified by the math — only the induced
ranking and relative magnitudes are meaningful.
