# Transformer MIL

Transformer MIL changes the context operator. Instead of pooling independent
patch embeddings, it lets every token mix with every other token before the bag
readout.

## Forward Map

Let:

```math
X^0=[h_{\mathrm{cls}};h_1;\ldots;h_n]\in\mathbb{R}^{(n+1)\times d}.
```

For layer \(\ell\):

```math
Q^\ell=X^{\ell-1}W_Q^\ell,
\qquad
K^\ell=X^{\ell-1}W_K^\ell,
\qquad
V^\ell=X^{\ell-1}W_V^\ell.
```

Self-attention is:

```math
A^\ell
=
\mathrm{softmax}
\left(
\frac{Q^\ell(K^\ell)^\top}{\sqrt{d_k}}
\right).
```

The token update is:

```math
Y^\ell=A^\ell V^\ell.
```

With residual and normalization:

```math
X^\ell
=
X^{\ell-1}
+
\mathrm{MSA}(\mathrm{LN}(X^{\ell-1})).
```

After \(L\) layers, use the class token:

```math
z=X^L_{\mathrm{cls}},
\qquad
p=\mathrm{softmax}(W_o z+b_o).
```

## Nyström Approximation

Full attention uses:

```math
A=\mathrm{softmax}(QK^\top/\sqrt{d_k})
\in\mathbb{R}^{n\times n}.
```

Nyström attention approximates \(A\) using \(m\) landmark queries and keys:

```math
\widehat A
=
\mathrm{softmax}(Q\widetilde K^\top/\sqrt{d_k})
\left[
\mathrm{softmax}(\widetilde Q\widetilde K^\top/\sqrt{d_k})
\right]^+
\mathrm{softmax}(\widetilde QK^\top/\sqrt{d_k}).
```

Then:

```math
\widehat Y=\widehat A V.
```

The assumption is that the attention matrix is well represented through a small
landmark set.

## Positional Context

If a model restores tokens to a grid and applies positional convolution:

```math
H_{\mathrm{grid}}\in\mathbb{R}^{q\times q\times d},
```

then a multiscale positional update has the form:

```math
H'
=
H_{\mathrm{grid}}
+
\mathrm{DWConv}_{3\times3}(H_{\mathrm{grid}})
+
\mathrm{DWConv}_{5\times5}(H_{\mathrm{grid}})
+
\mathrm{DWConv}_{7\times7}(H_{\mathrm{grid}}).
```

This injects local geometry into an otherwise sequence-level transformer.

## Loss

```math
\mathcal L_{\mathrm{CE}}
=
-\sum_c y_c\log p_c.
```

## Mathematical Innovation

Transformer MIL replaces first-moment pooling with dense pairwise context:

```math
h_i
\mapsto
\widetilde h_i
=
\sum_j A_{ij}Vh_j.
```

The fundamental object is the learned \(n\times n\) interaction matrix.

## Inductive Bias

- Patch-patch correlations matter.
- A class token can collect global information.
- Positional encodings or positional convolutions can make token geometry
  meaningful.
- Attention matrices may be approximately low rank.

## Failure Mode

The model can overfit because it learns many pairwise interactions:

```math
O(n^2)
```

for full attention. Approximations reduce compute but introduce approximation
error:

```math
A\approx\widehat A.
```

If token order or grid restoration is false, the positional assumptions become
false too.
