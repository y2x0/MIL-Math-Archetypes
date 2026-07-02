# CLAM: Clustering-Constrained Attention MIL

CLAM keeps the ABMIL forward map but adds supervision at the *instance* level.
It uses a shared attention backbone with one class-specific attention branch per
class, and it constrains the learned attention with an auxiliary instance
clustering objective. The clustering loss forces the most- and least-attended
patches to be linearly separable, which regularizes the attention measure on
small cohorts. Nature Biomedical Engineering 2021.

Throughout, a slide is a bag of $n$ patch embeddings

```math
H=\{h_1,\ldots,h_n\},\qquad h_i\in\mathbb{R}^{d},
```

and $C$ is the number of classes. If you have read
[attention MIL](02_attention_mil.md), CLAM is that model plus two changes:
attention becomes class-specific, and a second loss is attached to individual
patches.

## Step 1: Shared Gated-Attention Backbone

A first linear layer projects each patch into the attention space:

```math
g_i=\tanh\!\big(V h_i\big)\odot\sigma\!\big(U h_i\big)\in\mathbb{R}^{d'},
\qquad
V,U\in\mathbb{R}^{d'\times d}.
```

The two paths do different jobs. The $\tanh(V h_i)$ path is a bounded feature
map in $(-1,1)^{d'}$. The $\sigma(U h_i)$ path is a per-coordinate gate in
$(0,1)^{d'}$ that can switch features on or off depending on the patch, which
lets the scorer model non-linear "this feature only matters in this context"
effects. Their elementwise product $g_i$ is the gated representation. Setting
$U=0$ (so $\sigma(Uh_i)=\tfrac12$) recovers the plain, non-gated attention of
ABMIL up to a constant.

## Step 2: Class-Specific Attention Scores

CLAM attaches one linear scorer per class. Collect the $C$ scorers as rows of a
matrix $W_a\in\mathbb{R}^{C\times d'}$, so class $c$ has weight
$w_c\in\mathbb{R}^{d'}$:

```math
\ell_{i,c}=w_c^\top g_i\in\mathbb{R}.
```

Normalize each class score across the bag with a softmax, giving one probability
measure over patches per class:

```math
a_{i,c}
=
\frac{\exp(\ell_{i,c})}{\sum_{j=1}^{n}\exp(\ell_{j,c})},
\qquad
a_{i,c}\ge0,
\qquad
\sum_{i=1}^{n}a_{i,c}=1.
```

Two variants:

- **`CLAM-SB` (single branch).** One shared attention map, $a_i=a_{i,c}$ for all
  $c$ (i.e. $W_a$ has a single row). Used when classes are mutually exclusive.
- **`CLAM-MB` (multi branch).** The full class-specific $a_{i,c}$ above, one map
  per class.

## Step 3: Per-Class Readout and Classification

Each class reads out its own slide vector as the attention-weighted mean of the
*raw* patch embeddings, then scores it with a class-specific classifier vector
$W_{b,c}\in\mathbb{R}^{d}$:

```math
z_c=\sum_{i=1}^{n}a_{i,c}\,h_i\in\mathbb{R}^{d},
\qquad
s_c=W_{b,c}\,z_c\in\mathbb{R},
\qquad
p=\mathrm{softmax}(s)\in\Delta^{C-1}.
```

Because $\sum_i a_{i,c}=1$ and $a_{i,c}\ge0$, each $z_c$ is a convex combination
of the patch embeddings:

```math
z_c\in\mathrm{conv}\{h_1,\ldots,h_n\},
\qquad
z_c=\mathbb E_{i\sim a_{\cdot,c}}[h_i].
```

So the slide vector for class $c$ is the expected patch embedding under the
class-$c$ attention measure. This is the ABMIL readout, replicated per class.

## Step 4: Instance Clustering Constraint

The novel part. CLAM attaches a binary linear classifier per class,

```math
\theta_c(h_i)=W_{\mathrm{inst},c}\,h_i\in\mathbb{R}^{2},
\qquad
W_{\mathrm{inst},c}\in\mathbb{R}^{2\times d},
```

whose two logits mean "not evidence for $c$" vs "evidence for $c$." The
supervision for this classifier is not given — it is *manufactured from the
attention ranking*, which is why the method is "clustering-constrained": patches
are clustered into evidence / non-evidence by how much attention they receive.

**Pseudo-label assignment.** Fix a cluster size $k$ (a hyperparameter, e.g.
$k=8$). For class $c$, sort patches by $a_{i,c}$ and take the extremes:

```math
\mathcal T_c^{+}=\{\,i : a_{i,c}\text{ is among the top }k\,\},
\qquad
\mathcal T_c^{-}=\{\,i : a_{i,c}\text{ is among the bottom }k\,\}.
```

For a bag with ground-truth label $Y$:

- **In-class branch $c=Y$.** Top-$k$ patches are positive evidence
  ($\text{target}=1$), bottom-$k$ are negative ($\text{target}=0$).
- **Out-of-class branches $c\ne Y$ (mutually-exclusive setting).** The top-$k$
  patches of a *wrong* class should not look like evidence for it, so they are
  labeled negative ($\text{target}=0$).

Compactly, the target for the pseudo-labeled pair $(i,c)$ is

```math
t_{i,c}=
\begin{cases}
1, & c=Y,\ i\in\mathcal T_Y^{+},\\[2pt]
0, & c=Y,\ i\in\mathcal T_Y^{-},\\[2pt]
0, & c\ne Y,\ i\in\mathcal T_c^{+}.
\end{cases}
```

Let $\mathcal B$ be the set of all pseudo-labeled $(i,c)$ pairs for the bag;
$|\mathcal B|=2k$ for `CLAM-SB` and up to $2k+(C-1)k$ for `CLAM-MB`.

**Instance loss.** CLAM uses the smooth top-1 SVM loss. Write the binary margin
for pair $(i,c)$ using the signed label $\tilde t_{i,c}=2t_{i,c}-1\in\{-1,+1\}$
and the score gap $\delta_{i,c}=\big(\theta_c(h_i)\big)_{1}-\big(\theta_c(h_i)\big)_{0}$
between the positive and negative logit. A standard hinge loss is

```math
\ell^{\mathrm{hinge}}_{i,c}=\max\!\big(0,\;1-\tilde t_{i,c}\,\delta_{i,c}\big),
```

and its smoothed (temperature-$\tau$) version, which CLAM actually optimizes, is
the softplus relaxation

```math
\ell^{\mathrm{smooth}}_{i,c}
=
\tau\,\log\!\Big(1+\exp\!\big((1-\tilde t_{i,c}\,\delta_{i,c})/\tau\big)\Big)
\;\xrightarrow[\tau\to0]{}\;
\ell^{\mathrm{hinge}}_{i,c}.
```

A cross-entropy head is an equally valid instance objective and is common in
re-implementations:

```math
\ell^{\mathrm{CE}}_{i,c}=\mathrm{BCE}\big(\theta_c(h_i),\,t_{i,c}\big).
```

The clustering loss averages over the pseudo-labeled set:

```math
\mathcal L_{\mathrm{inst}}
=
\frac{1}{|\mathcal B|}\sum_{(i,c)\in\mathcal B}\ell^{\mathrm{smooth}}_{i,c}.
```

## Step 5: Total Objective

```math
\mathcal L
=
c_1\,\mathcal L_{\mathrm{CE}}(p,Y)
+
c_2\,\mathcal L_{\mathrm{inst}},
\qquad
c_1+c_2=1,\ \ c_1,c_2\ge0,
```

with the bag term

```math
\mathcal L_{\mathrm{CE}}(p,Y)=-\sum_{c=1}^{C}\mathbf 1[c=Y]\log p_c=-\log p_Y.
```

The bag term shapes the slide prediction; the instance term shapes the geometry
of the embedding space so that attention-selected patches are separable.

## Gradient View: How the Two Losses Interact

**Bag loss into attention.** The slide score is linear in $z_c$, and $z_c$ is
linear in the weights $a_{\cdot,c}$, so with $\partial\mathcal L_{\mathrm{CE}}/\partial s_c=p_c-\mathbf 1[c=Y]$
and $\partial s_c/\partial a_{i,c}=W_{b,c}h_i$,

```math
\frac{\partial\mathcal L_{\mathrm{CE}}}{\partial a_{i,c}}
=
(p_c-\mathbf 1[c=Y])\,W_{b,c}\,h_i.
```

Pushing this through the softmax Jacobian
$\partial a_{i,c}/\partial \ell_{j,c}=a_{i,c}(\delta_{ij}-a_{j,c})$ gives the
signal on the raw score of patch $j$ in branch $c$:

```math
\frac{\partial\mathcal L_{\mathrm{CE}}}{\partial \ell_{j,c}}
=
(p_c-\mathbf 1[c=Y])\;a_{j,c}\;\big(W_{b,c}h_j-W_{b,c}z_c\big).
```

The factor $W_{b,c}(h_j-z_c)$ is how much patch $j$'s contribution deviates from
the current bag mean: attention rises on patches that pull the class score in
the direction that reduces the bag loss, weighted by how much attention they
already hold. This is the ABMIL learning signal — CLAM does not change it.

**Instance loss into embeddings.** The clustering term acts directly on $h_i$
(not through the softmax). For the smooth-SVM branch,

```math
\frac{\partial \ell^{\mathrm{smooth}}_{i,c}}{\partial \delta_{i,c}}
=
-\,\tilde t_{i,c}\,\sigma\!\big((1-\tilde t_{i,c}\,\delta_{i,c})/\tau\big),
```

so top-$k$ patches are pushed to increase the positive-vs-negative logit gap and
bottom-$k$ patches to decrease it — a margin on the *support* of the attention
measure. The two gradients cooperate: the bag loss decides which patches get
attention, and the instance loss makes those same patches linearly separable
from the ignored ones.

## Mathematical Innovation

ABMIL supervises only the expectation $z_c=\mathbb E_{i\sim a_{\cdot,c}}[h_i]$.
CLAM adds a constraint on the *support* of that measure — where $a$ is large vs
small must lie on opposite sides of a linear boundary:

```math
W_{\mathrm{inst},c}\,h_i\ \text{scores positive for}\ i\in\mathcal T_c^{+},
\qquad
\text{negative for}\ i\in\mathcal T_c^{-}.
```

This is extra supervision, not a new context operator: `C` is still per-patch,
and the readout `R` is still a weighted first moment.

## Complexity

Per slide, attention scoring and readout are $O(nd d')$ and $O(ndC)$; the
instance loss touches only $2k$ (or $O(Ck)$) patches, so it adds
$O(Ckd)$ — negligible next to the backbone. Memory is $O(nd)$; there is no
$n\times n$ interaction matrix.

## Inductive Bias

- The bag is unordered; attention is computed independently per patch.
- Each class has its own relevant-patch measure (multi-branch).
- The most- and least-attended patches are reliable enough to act as
  pseudo-labeled instances.
- Instance-level separability is a useful prior when labeled bags are few.

## Failure Mode

The pseudo-labels are only as good as the current attention map. Early in
training, ranking by $a_{i,c}$ is near-uniform, so the manufactured targets can
reinforce a wrong support set:

```math
a_{i,c}\approx\tfrac1n
\ \Longrightarrow\
\mathcal T_c^{+}\ \text{is essentially random.}
```

The mutually-exclusive labeling assumes one dominant class per slide, which is
wrong for co-occurring phenotypes. And $k$ is fixed: if true disease burden is
much smaller or larger than $k$, the top-$k$ set mixes positive and negative
evidence, so $\mathcal L_{\mathrm{inst}}$ is trained on contradictory targets.
