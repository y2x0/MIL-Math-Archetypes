# Mean-Pooling MIL

Mean pooling is the clean floor. It assumes that the first moment of the patch
embedding distribution is enough.

## Forward Map

Project each patch:

```math
u_i=W_p h_i+b_p,
\qquad
\phi_i=\rho(u_i),
```

where \(\rho\) is usually ReLU or GELU.

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
nonlinearity.

## Mathematical Innovation

None. That is the point. It is the null model for learned aggregation.

Mean pooling estimates:

```math
z=\mathbb E_{i\sim\mathrm{Uniform}(1:n)}[\phi_i].
```

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
logit. This is the first reason to replace uniform weights with learned weights.
