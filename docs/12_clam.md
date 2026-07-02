# CLAM: Clustering-Constrained Attention MIL

CLAM keeps the ABMIL forward map but adds supervision at the *instance* level.
It uses a shared attention backbone with one class-specific attention branch per
class, and it constrains the learned attention with an auxiliary instance
clustering objective. The clustering loss forces the most- and least-attended
patches to be linearly separable, which regularizes the attention measure on
small cohorts. Nature Biomedical Engineering 2021.

## Attention Backbone

Start with patch embeddings and a shared projection:

```math
h_i\in\mathbb{R}^{d},
\qquad
g_i=\tanh(V h_i)\odot\sigma(U h_i)\in\mathbb{R}^{d'}.
```

A single attention head reduces to gated ABMIL. CLAM makes the scorer
class-specific: a weight matrix produces one score per class,

```math
\ell_{i,c}=w_c^\top g_i,
\qquad
a_{i,c}=\frac{\exp(\ell_{i,c})}{\sum_{j=1}^{n}\exp(\ell_{j,c})},
\qquad
\sum_i a_{i,c}=1.
```

Each class therefore reads out its own slide vector:

```math
z_c=\sum_{i=1}^{n}a_{i,c}\,h_i,
\qquad
s_c=W_{b,c}\,z_c,
\qquad
p=\mathrm{softmax}(s).
```

Single-branch CLAM (`CLAM-SB`) shares one attention map across classes;
multi-branch CLAM (`CLAM-MB`) uses the class-specific `a_{i,c}` above.

## Instance Clustering Constraint

CLAM attaches a binary instance classifier to each class branch:

```math
\hat y_{i,c}=W_{\mathrm{inst},c}\,h_i\in\mathbb{R}^{2}.
```

Attention rank supplies pseudo-labels. For a bag with true label $Y$, the
top-$k$ patches by $a_{i,Y}$ are treated as **in-class evidence** and the
bottom-$k$ as **out-of-class evidence**:

```math
\mathcal T_c^{+}=\{\,i:a_{i,c}\ \text{among the top } k\,\},
\qquad
\mathcal T_c^{-}=\{\,i:a_{i,c}\ \text{among the bottom } k\,\}.
```

For the ground-truth branch the targets are

```math
t_{i,Y}=
\begin{cases}
1 & i\in\mathcal T_Y^{+}\\
0 & i\in\mathcal T_Y^{-},
\end{cases}
```

and, under the mutually-exclusive setting, the top-$k$ patches of every
non-ground-truth branch $c\ne Y$ are all labeled negative,

```math
t_{i,c}=0
\quad\text{for }i\in\mathcal T_c^{+},\ c\ne Y.
```

The instance loss is a smooth binary objective (SVM or cross-entropy) over the
$2k$ pseudo-labeled patches per branch:

```math
\mathcal L_{\mathrm{inst}}
=
\frac{1}{|\mathcal B|}
\sum_{(i,c)\in\mathcal B}
\mathrm{BCE}\!\left(\hat y_{i,c},\,t_{i,c}\right).
```

## Objective

```math
\mathcal L
=
c_1\,\mathcal L_{\mathrm{CE}}(p,Y)
+
c_2\,\mathcal L_{\mathrm{inst}},
\qquad
c_1+c_2=1.
```

The bag term shapes the slide prediction; the instance term shapes the geometry
of the patch-embedding space so that attention-selected patches are separable.

## Mathematical Innovation

ABMIL supervises only the expectation $z=\mathbb E_{i\sim a}[h_i]$. CLAM adds a
constraint on the *support* of that measure: the patches where $a$ is large and
where $a$ is small must land on opposite sides of a linear boundary,

```math
W_{\mathrm{inst},c}^\top h_i>0
\ \text{for}\ i\in\mathcal T_c^{+},
\qquad
W_{\mathrm{inst},c}^\top h_i<0
\ \text{for}\ i\in\mathcal T_c^{-}.
```

This is extra supervision, not a new context operator: `C` is still per-patch,
and the readout is still a weighted first moment.

## Inductive Bias

- The bag is unordered; attention is computed independently per patch.
- Each class has its own relevant-patch measure.
- The most- and least-attended patches are reliable enough to act as
  pseudo-labeled instances.
- Instance-level separability is a useful prior when bags are few.

## Failure Mode

The pseudo-labels are only as good as the current attention map. Early in
training, ranking by $a_{i,c}$ is near-random, so $\mathcal L_{\mathrm{inst}}$
can reinforce a wrong support set,

```math
\mathcal T_c^{+}\ \text{unreliable when}\ a_{i,c}\approx\tfrac1n.
```

The mutually-exclusive labeling assumes one dominant class per slide, which is
wrong for co-occurring phenotypes. And $k$ is a fixed hyperparameter: if true
disease burden is much smaller or larger than $k$, the top-$k$ set mixes
positive and negative evidence.
