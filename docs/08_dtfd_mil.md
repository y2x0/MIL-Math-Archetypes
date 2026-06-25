# DTFD-MIL: Double-Tier Feature Distillation

DTFD-MIL addresses a structural asymmetry in WSI MIL: the number of bags
(slides) is small, but each bag contains thousands of instances. It introduces
pseudo-bag augmentation and a two-tier MIL hierarchy to make small cohorts
tractable. CVPR 2022.

## Pseudo-Bag Construction

A slide $n$ with instance set $\mathbf{X}_n = \{x_k\}_{k=1}^{K_n}$ is
randomly partitioned into $M$ disjoint pseudo-bags:

```math
\mathbf{X}_n = \{\mathbf{X}_{nm}\}_{m=1}^{M},
\qquad
\bigcup_{m=1}^{M} \mathbf{X}_{nm} = \mathbf{X}_n,
\qquad
\mathbf{X}_{nm} \cap \mathbf{X}_{nm'} = \emptyset.
```

Each pseudo-bag inherits the slide label: $Y_{nm} = Y_n$. This virtually
multiplies the number of training bags from $N$ to $MN$.

## Instance Probability Derivation

Under a standard ABMIL forward pass, the class activation for instance $k$ and
class $c$ is derived via a Grad-CAM-style linearization:

```math
L_k^c
=
\sum_{d=1}^{D}
\beta_d^c \,\hat{h}_{k,d},
\qquad
\beta_d^c
=
\frac{1}{K}\sum_{i=1}^{K}
\frac{\partial s_c}{\partial \hat{h}_{k,d}},
```

where $\hat{h}_k = a_k K h_k$ is the attention-weighted feature and $s_c$ is
the pre-softmax logit. The instance-level class probability is:

```math
p_k^c
=
\frac{\exp(L_k^c)}{\sum_{t=1}^{C}\exp(L_k^t)}.
```

## Tier-1: MIL on Pseudo-Bags

A MIL model $T_1$ processes each pseudo-bag and produces a prediction:

```math
y_{nm} = T_1\!\left(\{h_k = H(x_k) \mid x_k \in \mathbf{X}_{nm}\}\right).
```

Tier-1 loss (binary cross-entropy over all $MN$ pseudo-bags):

```math
\mathcal{L}_1
=
-\frac{1}{MN}
\sum_{n,m}
\left[
Y_{nm}\log y_{nm}
+
(1-Y_{nm})\log(1-y_{nm})
\right].
```

## Tier-2: MIL on Distilled Pseudo-Bag Features

Each pseudo-bag is summarized as a distilled feature $\hat{f}_{nm}$ (e.g., the
ABMIL bag embedding from Tier-1). A second MIL model $T_2$ pools over the $M$
summaries to produce the final slide prediction:

```math
\hat{y}_n = T_2\!\left(\{\hat{f}_{nm} \mid m=1,\ldots,M\}\right).
```

Tier-2 loss (binary cross-entropy at the slide level):

```math
\mathcal{L}_2
=
-\frac{1}{N}
\sum_{n}
\left[
Y_n\log\hat{y}_n
+
(1-Y_n)\log(1-\hat{y}_n)
\right].
```

## Overall Objective

```math
\mathcal{L}
=
\arg\min_{\theta_1}\,\mathcal{L}_1
\;+\;
\arg\min_{\theta_2}\,\mathcal{L}_2.
```

The two tiers have separate parameter sets and are optimized independently.

## Key Innovation

Standard MIL treats one slide as one bag. DTFD-MIL treats one slide as a
bag-of-bags:

```math
\text{instance}
\xrightarrow{T_1}
\text{pseudo-bag summary}
\xrightarrow{T_2}
\text{slide prediction}.
```

Random splits at each epoch also act as data augmentation — the model sees many
distinct subsets of each slide.

## Inductive Biases

- A slide's label is consistent across all of its pseudo-bags.
- Pseudo-bags are exchangeable at Tier-2.
- Two-tier pooling can represent higher-order statistics than a single attention
  pool.

## Failure Modes

- Random splits ignore spatial or functional tissue structure.
- Tier-1 receives fewer instances per bag, which can destabilize attention for
  very small pseudo-bags.
- Two separately optimized tiers can diverge without an explicit alignment loss.
- The instance-probability derivation relies on a Grad-CAM linearity
  approximation.
