# DSMIL: Dual-Stream MIL

DSMIL runs two readouts in parallel and averages them. The first stream is a
max-pooling instance classifier that selects a single **critical instance**; the
second stream scores every patch by its similarity to that critical instance and
pools an attention-weighted bag embedding. Combining a max statistic with a
weighted first moment lets the model react to rare strong evidence while still
using the whole bag. CVPR 2021.

Setup: a bag of $n$ patch embeddings

```math
H=\{h_1,\ldots,h_n\},\qquad h_i\in\mathbb{R}^{d},
```

and $C$ classes. Under the standard MIL assumption a bag is positive if *any*
instance is positive, which is exactly a max over instance scores — this is the
assumption DSMIL's first stream encodes directly.

## Stream 1: Critical Instance by Max Pooling

Give every patch an instance score with a shared linear classifier
$W_0\in\mathbb{R}^{C\times d}$:

```math
c_i=W_0\,h_i\in\mathbb{R}^{C}.
```

Reduce the bag by taking, per class, the max over instances; the **critical
instance** $m$ is the patch achieving the largest score of any class:

```math
(s_{\max})_c=\max_{i}\,(c_i)_c,
\qquad
m=\arg\max_{i}\,\max_{c}\,(c_i)_c.
```

This stream by itself is max-pooling MIL: it keeps the single most extreme piece
of evidence and discards the rest.

**Gradient (why max pooling is sparse).** The max is piecewise linear, so its
subgradient routes the entire learning signal to one patch per class:

```math
\frac{\partial (s_{\max})_c}{\partial c_i}
=
\mathbf 1\!\big[i=\arg\max_{j}(c_j)_c\big].
```

Only the arg-max patch updates $W_0$ on a given step. That is efficient but
high-variance, which is the structural reason DSMIL adds a second, dense stream.

## Stream 2: Attention to the Critical Instance

Project every patch into a query and an information (value) vector:

```math
q_i=W_q\,h_i\in\mathbb{R}^{d_q},
\qquad
v_i=W_v\,h_i\in\mathbb{R}^{d_v},
\qquad
W_q\in\mathbb{R}^{d_q\times d},\ W_v\in\mathbb{R}^{d_v\times d}.
```

Let $q_m$ be the query of the critical instance from Stream 1. Score each patch
by the inner product between its query and the critical instance's query, then
normalize across the bag:

```math
U(h_i,h_m)=\langle q_i,\,q_m\rangle,
\qquad
a_i=\frac{\exp\!\big(\langle q_i,q_m\rangle\big)}
{\sum_{j=1}^{n}\exp\!\big(\langle q_j,q_m\rangle\big)},
\qquad
\sum_i a_i=1.
```

This is scaled-dot-product attention with a single fixed query — the critical
instance — rather than one query per token. The bag embedding is the
attention-weighted sum of value vectors, scored by a bag classifier
$W_b\in\mathbb{R}^{C\times d_v}$:

```math
b=\sum_{i=1}^{n}a_i\,v_i\in\mathbb{R}^{d_v},
\qquad
s_{\mathrm{bag}}=W_b\,b\in\mathbb{R}^{C}.
```

Because $q_m$ is itself in the sum, $a_m$ is typically the largest weight
($\langle q_m,q_m\rangle=\lVert q_m\rVert^2$), so the bag embedding is anchored
at the critical instance and blends in patches whose queries align with it.

**Gradient into attention.** With
$\partial\mathcal L/\partial b=W_b^\top(\partial\mathcal L/\partial s_{\mathrm{bag}})$
and the softmax Jacobian $\partial a_i/\partial U_j=a_i(\delta_{ij}-a_j)$ where
$U_j=\langle q_j,q_m\rangle$,

```math
\frac{\partial \mathcal L}{\partial U_j}
=
a_j\Big\langle W_b^\top\tfrac{\partial\mathcal L}{\partial s_{\mathrm{bag}}},\ v_j-b\Big\rangle.
```

Attention increases on patches whose value $v_j$ deviates from the current bag
embedding $b$ in the loss-reducing direction — the same "deviation from the
mean" signal as ABMIL, but with weights tied to similarity with $q_m$.

## Combined Prediction and Loss

The final logit averages the two streams:

```math
s=\tfrac12\big(s_{\max}+s_{\mathrm{bag}}\big),
\qquad
p=\mathrm{softmax}(s).
```

Both streams are supervised so that each is individually calibrated, not just
their average:

```math
\mathcal L
=
\tfrac12\,\mathcal L_{\mathrm{CE}}(s_{\max},Y)
+
\tfrac12\,\mathcal L_{\mathrm{CE}}(s_{\mathrm{bag}},Y),
\qquad
\mathcal L_{\mathrm{CE}}(s,Y)=-\log\mathrm{softmax}(s)_Y.
```

## Mathematical Innovation

DSMIL fuses two different surviving statistics in one readout:

```math
\underbrace{(s_{\max})_c=\max_i (c_i)_c}_{\text{extreme evidence}}
\qquad
\underbrace{s_{\mathrm{bag}}=W_b\sum_i a_i v_i}_{\text{weighted first moment}}.
```

The key difference from ABMIL is where the attention query comes from. ABMIL
learns a free query $w$ and scores $a_i\propto\exp(w^\top\tanh(Vh_i))$. DSMIL
sets the query to the data-dependent critical instance:

```math
a_i=\mathrm{softmax}_i\,\langle q_i,\,q_m\rangle,
\qquad
m=\arg\max_i\max_c(c_i)_c.
```

So the first-moment stream is computed *relative to* the extreme-evidence stream
rather than independently. This couples presence detection (does any patch look
diseased?) with amount estimation (how much of the bag agrees with it?).

## Complexity

Stream 1 is $O(ndC)$ and the arg-max is $O(nC)$. Stream 2 costs $O(nd d_q)$ for
queries, $O(nd_q)$ for the $n$ dot products against the single $q_m$, and
$O(nd_v)$ for the weighted sum — all linear in $n$. There is no $n\times n$
attention matrix, unlike Transformer MIL. Memory is $O(n\max(d,d_q,d_v))$.

## Inductive Bias

- The bag is unordered; both streams are permutation invariant.
- The standard MIL assumption holds: one most-suspicious patch can make the bag
  positive (Stream 1 is literally a max).
- Patches whose queries align with the critical instance carry corroborating
  evidence (Stream 2).
- Averaging a max statistic and a mean statistic is more robust than committing
  to either alone.

## Failure Mode

The entire second stream is anchored to one arg-max choice:

```math
m=\arg\max_i\,\max_c\,(c_i)_c.
```

If the critical instance is a false positive (staining artifact, edge patch),
$q_m$ is corrupted and every attention weight $a_i$ is scored against noise. The
arg-max is also non-smooth: near-ties make $m$ jump between patches across
steps, so $a_i$ (and its gradient) can change discontinuously. And when disease
is diffuse rather than focal, a single critical instance underrepresents the
bag — the classic max-pooling limitation, only partly mitigated by averaging in
the bag stream.
