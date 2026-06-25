# RRT-MIL: Re-embedded Regional Transformer

RRT-MIL argues that global self-attention over thousands of patch tokens causes
feature homogenization, degrading downstream MIL performance. It inserts a
re-embedding stage between the frozen encoder and any ABMIL-style aggregator,
using local windowed attention followed by cross-region attention. CVPR 2024.

## Re-Embedding Framework

Standard MIL:

```math
\hat{Y} = C(A(F(X))).
```

RRT-MIL inserts a re-embedding module $R$:

```math
\hat{Y} = C(A(R(F(X)))).
```

$R$ is the Re-embedded Regional Transformer (R2T). It takes frozen features
$H \in \mathbb{R}^{I \times D}$ and outputs context-enriched features
$Z \in \mathbb{R}^{I \times D}$ of the same shape — a drop-in module for any
existing aggregator.

## R2T Forward Pass

Two sequential blocks with residual connections:

```math
\hat{Z} = \operatorname{R-MSA}(\operatorname{LN}(H)) + H,
```

```math
Z = \operatorname{CR-MSA}(\operatorname{LN}(\hat{Z})) + \hat{Z}.
```

## Regional Multi-head Self-Attention (R-MSA)

The $I$ patch tokens are reshaped into a 2D grid
$\lceil\sqrt{I}\rceil \times \lceil\sqrt{I}\rceil$, then partitioned into $L^2$
non-overlapping $M \times M$ windows:

```math
H \;\in\; \mathbb{R}^{I \times D}
\;\to\;
\{H^l\}_{l=1}^{L^2},
\quad
H^l \in \mathbb{R}^{M^2 \times D},
\quad
L \cdot M = \lceil\sqrt{I}\rceil.
```

Attention score for query $i$ and key $j$ within window $l$:

```math
e_{ij}^l
=
\frac{(H_i^l W^Q)(H_j^l W^K)^\top}{\sqrt{D}}.
```

The Embedded Position Encoding Generator (EPEG) injects relative positional
information via a 1-D convolution over the score vector before softmax:

```math
\alpha_{ij}^l
=
\operatorname{SoftMax}_j\!\left(
e_{ij}^l + \operatorname{Conv}_{1\text{-}D}(e_{ij}^l)
\right).
```

Attended output for instance $i$ in window $l$:

```math
\operatorname{R-SA}^l_i
=
\sum_{j=1}^{M^2}
\alpha_{ij}^l\,(H_j^l W^V).
```

Complexity is $O(L^2 M^4)$ instead of $O(I^2)$ for global attention.

## Cross-Region Multi-head Self-Attention (CR-MSA)

Each window is compressed to $K$ representative tokens via learned pooling:

```math
W_a^l
=
\operatorname{SoftMax}_{m=1}^{M^2}\!\left(\hat{Z}_m^l \,\Phi\right)
\in \mathbb{R}^{M^2 \times K},
\qquad
R^l = (W_a^l)^\top \hat{Z}^l \in \mathbb{R}^{K \times D},
```

where $\Phi \in \mathbb{R}^{D \times K}$ is learnable. The $L^2$ region
representatives are stacked and processed with global MSA:

```math
\hat{R} = S(R) \in \mathbb{R}^{L^2 K \times D}.
```

Global context is distributed back to each instance:

```math
W_d^l
=
\operatorname{MinMax}_{m=1}^{M^2}\!\left(\hat{Z}_m^l \,\Phi\right)
\in \mathbb{R}^{M^2 \times K},
\qquad
Z^l = (W_d^l)\,\hat{R}^l\,\hat{W}_d^l,
```

where $\hat{W}_d^l$ uses SoftMax normalization across the $K$ dimension.

## Integration with ABMIL

Re-embedded features $Z$ feed directly into ABMIL's gated attention pool.
The full model trains end-to-end:

```math
\arg\min_{\theta_R, \theta_A, \theta_C}
\mathcal{L}_{\mathrm{CE}}(Y,\, C(A(R(H)))).
```

## Key Innovation

R2T first enriches each token within a local region (fine-grained tissue
context), then propagates compressed cross-region summaries back to each
instance (global architectural context) — all without a full $I \times I$
attention matrix.

## Inductive Biases

- Patch order, when reshaped to 2D, encodes meaningful spatial locality.
- Fine-grained features are better recovered from local than global attention.
- $K$ prototype vectors per window are sufficient to summarize local diversity.
- Region-to-region relationships can be captured at a coarser scale.

## Failure Modes

- The 2D reshape assumes patches are extracted in a consistent scanline order.
- Window boundaries may split spatially continuous tissue structures.
- EPEG is 1-D over the flattened score vector, not 2-D over spatial positions.
- Adding R2T to a pretrained MIL pipeline requires retraining the aggregator.
