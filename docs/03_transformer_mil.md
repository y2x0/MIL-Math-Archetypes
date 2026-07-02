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
\right),
```

where the softmax is applied row-wise, so $A^\ell\in\mathbb{R}^{(n+1)\times(n+1)}$
with each row a probability distribution over tokens: $A^\ell_{ij}\ge0$,
$\sum_j A^\ell_{ij}=1$. Row $i$ is the set of weights token $i$ places on all
tokens when forming its update.

The scaling by $\sqrt{d_k}$ is not cosmetic. If the entries of $Q,K$ are roughly
independent with unit variance, then $\langle q_i,k_j\rangle$ has variance $d_k$,
so without scaling the logits grow with dimension and softmax saturates into a
near one-hot map with vanishing gradient. Dividing by $\sqrt{d_k}$ normalizes the
logit variance back to $O(1)$ and keeps attention trainable:

```math
\mathrm{Var}\big(\langle q_i,k_j\rangle\big)=d_k
\ \Longrightarrow\
\mathrm{Var}\!\left(\frac{\langle q_i,k_j\rangle}{\sqrt{d_k}}\right)=1.
```

The token update is a weighted sum of value vectors:

```math
Y^\ell=A^\ell V^\ell,
\qquad
\widetilde h_i=\sum_{j}A^\ell_{ij}\,(V^\ell)_j.
```

### Multi-Head Attention

A single attention map can only express one relation. Multi-head attention runs
$H$ maps in parallel on projected subspaces of width $d_k=d/H$ and concatenates:

```math
\mathrm{head}_r=\mathrm{softmax}\!\left(\frac{Q_r K_r^\top}{\sqrt{d_k}}\right)V_r,
\qquad
\mathrm{MSA}=\big[\mathrm{head}_1\Vert\cdots\Vert\mathrm{head}_H\big]W_O.
```

Different heads can specialize to different patch-patch relations (e.g. one
tracks stroma-tumor adjacency, another tracks similar morphology) at the same
$O(n^2)$ cost, since the per-head width shrinks as $H$ grows.

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

The fundamental object is the learned \(n\times n\) interaction matrix. Mean
pooling and ABMIL both produce a bag vector that is a convex combination of the
*inputs*; a transformer instead rewrites each token as a convex combination of
all others before any readout, so the surviving statistic is the set of pairwise
correlations $\{\langle q_i,k_j\rangle\}$ rather than a single moment.

### Why Permutation Invariance Is Preserved (or Broken)

Without positional encodings, self-attention is permutation *equivariant*:
permuting the input tokens by $\pi$ permutes the outputs the same way,

```math
\mathrm{MSA}(\Pi X)=\Pi\,\mathrm{MSA}(X),
```

for any permutation matrix $\Pi$, because $A$ is built from content-only inner
products. A permutation-invariant readout (a shared class token, or attention
pooling) then makes the whole bag prediction permutation invariant — matching
the unordered nature of a slide. Adding positional encodings or the grid
convolution above deliberately breaks this, trading invariance for geometry.

## Inductive Bias

- Patch-patch correlations matter.
- A class token can collect global information.
- Positional encodings or positional convolutions can make token geometry
  meaningful.
- Attention matrices may be approximately low rank.

## Failure Mode

The model can overfit because it learns many pairwise interactions. For a bag of
$n$ patches the attention matrix has

```math
O(n^2)
```

entries — with $n$ in the thousands for a whole slide, that is both a memory
cost ($n^2$ scores per head per layer) and a statistical cost: the number of
free pairwise interactions vastly exceeds the number of labeled bags,

```math
n^2\ \text{interactions}\ \gg\ \#\{\text{labeled slides}\},
```

so the model can fit spurious patch-patch relations. This is why Transformer MIL
in pathology almost always uses a low-rank or landmark approximation
$A\approx\widehat A$ (the Nyström form above), which reduces cost to $O(nm)$ for
$m\ll n$ landmarks but introduces approximation error when $A$ is not actually
low rank. And if token order or grid restoration is false, the positional
assumptions become false too.
