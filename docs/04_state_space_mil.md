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

Here $s(t)\in\mathbb{R}^{N}$ is a latent memory of state dimension $N$,
$x(t)\in\mathbb{R}$ is one channel of the input, and $A\in\mathbb{R}^{N\times N}$,
$B\in\mathbb{R}^{N\times1}$, $C\in\mathbb{R}^{1\times N}$. A slide is fed as a
sequence of patches, one time step per patch.

To run this on discrete tokens with step size $\Delta$, we integrate the ODE
over one interval assuming $x$ is held constant (zero-order hold), which yields
the exact discrete update:

```math
\bar A=\exp(\Delta A),
```

```math
\bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\,\Delta B.
```

The matrix exponential is what makes this stable: eigenvalues of $A$ with
negative real part become eigenvalues of $\bar A$ inside the unit disk, so the
memory decays rather than explodes. The recurrence is then:

```math
s_t=\bar A s_{t-1}+\bar B x_t,
\qquad
y_t=Cs_t.
```

### Recurrent and Convolutional Views

Unrolling the recurrence from $s_0=0$ shows the output is a convolution of the
input with a fixed kernel:

```math
y_t=\sum_{j=0}^{t}\underbrace{C\bar A^{\,j}\bar B}_{\bar K_j}\;x_{t-j}
=(\bar K * x)_t,
\qquad
\bar K=\big(C\bar B,\ C\bar A\bar B,\ C\bar A^2\bar B,\ \ldots\big).
```

The two views trade off differently: the recurrence is $O(n)$ time and $O(1)$
memory per step (good for inference), while the convolution can be evaluated in
parallel across all positions during training. This equivalence — linear-time
recurrence *and* parallel training — is the whole appeal over an $O(n^2)$
transformer.

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

This gives content-dependent remembering and forgetting. Making $\Delta_t,B_t,C_t$
functions of the token is the "selective" step: a large $\Delta_t$ (long
effective step) lets a salient patch write strongly into the state and resets
the influence of older tokens, while a small $\Delta_t$ keeps the state
unchanged. In effect $\Delta_t$ is a learned, input-dependent gate over memory —
the same role a forget gate plays in an LSTM, but derived from the discretized
ODE. The price is that the kernel $\bar K$ is no longer fixed, so the pure
convolution view is replaced by a parallel associative scan.

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
the state dimension. The state $s_t\in\mathbb{R}^{N}$ is a fixed-size summary of
the entire prefix $(h_1,\ldots,h_t)$, so it is an information bottleneck: at most
$N$ real numbers separate the past from the future. A transformer keeps all $t$
tokens addressable ($O(t)$ memory, $O(t^2)$ compute); a state-space model keeps
$O(N)$ memory and $O(t)$ compute but can only preserve what fits in $N$
dimensions. Choosing $N$ trades capacity against cost.

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
