# Mean-Pooling MIL

Mean pooling is the clean floor. It assumes that the first moment of the patch
embedding distribution is enough.

A slide is a bag of $n$ patch embeddings

```math
H=\{h_1,\ldots,h_n\},\qquad h_i\in\mathbb{R}^{d},
```

with one label $y$ for the whole bag and no labels on individual patches. Every
model in this repo maps $H$ to a bag prediction; mean pooling is the simplest
such map and the baseline the others are measured against.

## Forward Map

Project each patch through a shared MLP layer:

```math
u_i=W_p h_i+b_p\in\mathbb{R}^{d'},
\qquad
\phi_i=\rho(u_i),
```

where \(\rho\) is usually ReLU or GELU, and "shared" means the same $W_p,b_p$
are applied to every patch (a $1\times1$ convolution over the bag).

Aggregate by an arithmetic mean:

```math
z
=
\frac{1}{n}\sum_{i=1}^{n}\phi_i.
```

Classify:

```math
\eta=W_o z+b_o,
\qquad
p=\mathrm{softmax}(\eta).
```

The full map is:

```math
H
\xrightarrow{\text{shared }W_p}
\Phi
\xrightarrow{\frac{1}{n}\sum_i}
z
\xrightarrow{W_o}
\eta
\xrightarrow{\mathrm{softmax}}
p.
```

## Loss

```math
\mathcal L_{\mathrm{CE}}
=
-\sum_{c=1}^{C}y_c\log p_c.
```

## Gradient Shape

For one bag:

```math
\frac{\partial \mathcal L}{\partial z}
=
W_o^\top(p-y).
```

Since:

```math
z=\frac{1}{n}\sum_{i=1}^{n}\phi_i,
```

each projected patch receives:

```math
\frac{\partial \mathcal L}{\partial \phi_i}
=
\frac{1}{n}W_o^\top(p-y).
```

All patches receive the same bag-level error signal before the projector
nonlinearity. Continuing the chain rule through $\phi_i=\rho(u_i)$ gives the
update actually applied to the shared projector:

```math
\frac{\partial\mathcal L}{\partial u_i}
=
\frac1n\,\big(W_o^\top(p-y)\big)\odot\rho'(u_i),
\qquad
\frac{\partial\mathcal L}{\partial W_p}
=
\sum_{i=1}^{n}\frac{\partial\mathcal L}{\partial u_i}\,h_i^\top.
```

Two consequences follow directly. First, the $1/n$ factor means large bags
produce proportionally smaller per-patch gradients, so the projector learns more
slowly from big slides unless the learning rate compensates. Second, the signal
$W_o^\top(p-y)$ is *identical* for every patch (only the ReLU mask $\rho'(u_i)$
differs), so mean pooling cannot send patch-specific credit — the seed of its
failure mode.

## Mathematical Innovation

None. That is the point. It is the null model for learned aggregation.

Mean pooling estimates the first moment of the (projected) patch distribution:

```math
z=\mathbb E_{i\sim\mathrm{Uniform}(1:n)}[\phi_i]
=\hat\mu_\Phi.
```

### Why the mean is a stable but blunt statistic

Treat the patches in a bag as i.i.d. draws with mean $\mu$ and covariance
$\Sigma$. The pooled vector is a sample mean, so it concentrates as the bag
grows:

```math
\mathbb E[z]=\mu,
\qquad
\mathrm{Cov}(z)=\frac{1}{n}\Sigma
\ \xrightarrow[n\to\infty]{}\ 0.
```

This is exactly why mean pooling is a strong, low-variance baseline: with
thousands of patches per slide, $z$ is an extremely stable estimate of $\mu$.
It is also exactly why it is blunt — anything that is *not* a shift of the mean
(a rare cell type, a small tumor focus) is averaged toward zero as $n$ grows.

## Inductive Bias

- The bag is unordered.
- Every patch has fixed weight \(1/n\).
- Patch interactions are unnecessary.
- Disease signal shifts the global embedding mean.
- Bag size does not matter after normalization.

## Failure Mode

A rare positive phenotype can vanish:

```math
z
=
\frac{k}{n}\mu_{\mathrm{pos}}
+
\left(1-\frac{k}{n}\right)\mu_{\mathrm{bg}},
\qquad
k\ll n.
```

When \(k/n\) is tiny, the positive component contributes little to the final
logit. Concretely, with a linear head $w_c$ the class-$c$ margin between a
positive and a negative bag is

```math
w_c^\top(z_{\mathrm{pos}}-z_{\mathrm{neg}})
=
\frac{k}{n}\,w_c^\top(\mu_{\mathrm{pos}}-\mu_{\mathrm{bg}}),
```

which shrinks linearly in $k/n$. Meanwhile the noise floor from the sample-mean
covariance $\tfrac1n\Sigma$ does not shrink as fast, so the signal-to-noise
ratio of the pooled representation degrades and the positive class becomes
undetectable. This is the first reason to replace uniform weights $1/n$ with
learned weights $a_i$ — the subject of [attention MIL](02_attention_mil.md).
