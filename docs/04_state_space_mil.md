# State-Space MIL

State-space MIL treats patch embeddings as a sequence and propagates a compact
latent memory along that sequence.

## Linear State-Space Core

A continuous state-space model is:

```math
\frac{d s(t)}{dt}=As(t)+Bx(t),
\qquad
y(t)=Cs(t).
```

After discretization:

```math
\bar A=\exp(\Delta A),
```

```math
\bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\Delta B.
```

The recurrence is:

```math
s_t=\bar A s_{t-1}+\bar B x_t,
\qquad
y_t=Cs_t.
```

## Selective State-Space Core

Selective models make parameters token-dependent:

```math
\Delta_t=g_\Delta(x_t),
\qquad
B_t=g_B(x_t),
\qquad
C_t=g_C(x_t).
```

Then:

```math
\bar A_t=\exp(\Delta_t A),
\qquad
\bar B_t=\Delta_t B_t,
```

and:

```math
s_t=\bar A_t s_{t-1}+\bar B_t x_t,
\qquad
y_t=C_t s_t.
```

This gives content-dependent remembering and forgetting.

## Sequence-Reordered MIL

Let \(S=(h_1,\ldots,h_n)\) be the stored patch sequence. A scan operator maps:

```math
Y=\mathrm{SSM}(S).
```

If a second ordering \(\pi\) is used:

```math
Y^{(\pi)}
=
\mathrm{SSM}(h_{\pi(1)},\ldots,h_{\pi(n)}).
```

Restore the original order:

```math
\widehat Y_i^{(\pi)}
=
Y_{\pi^{-1}(i)}^{(\pi)}.
```

Fuse scans:

```math
\widetilde h_i
=
W_f
\left[
Y_i^{(\mathrm{id})}
+
\widehat Y_i^{(\pi)}
\right]
+
h_i.
```

Then apply a bag readout, often attention:

```math
a_i
=
\frac{\exp(w^\top\tanh(V\widetilde h_i))}
{\sum_j\exp(w^\top\tanh(V\widetilde h_j))},
\qquad
z=\sum_i a_i\widetilde h_i.
```

## Loss

Classification:

```math
\mathcal L_{\mathrm{CE}}
=
-\sum_c y_c\log p_c.
```

Discrete-time survival, if used:

```math
\lambda_k=\sigma(\eta_k),
\qquad
S_k=\prod_{j=0}^{k}(1-\lambda_j).
```

For event bin \(Y\) and censor indicator \(c=1\):

```math
\ell_{\mathrm{unc}}
=
-(1-c)(\log S_{Y-1}+\log\lambda_Y),
```

```math
\ell_{\mathrm{cens}}
=
-c\log S_Y.
```

## Mathematical Innovation

State-space MIL replaces a dense pairwise matrix with a recurrent sufficient
state:

```math
(h_1,\ldots,h_t)
\mapsto
s_t.
```

The complexity is linear in sequence length, but information must fit through
the state dimension.

## Inductive Bias

- Sequence order is meaningful.
- Useful context can be compressed into a finite recurrent memory.
- Long-range dependencies do not require all pairwise interactions.
- Different scan orders expose different predecessor sets.

## Failure Mode

State-space MIL is not inherently permutation invariant:

```math
\mathrm{SSM}(h_1,h_2,\ldots,h_n)
\ne
\mathrm{SSM}(h_{\pi(1)},h_{\pi(2)},\ldots,h_{\pi(n)}).
```

If the sequence order is arbitrary, the model learns artifacts of serialization.
If the state is too small, rare evidence can be overwritten.
