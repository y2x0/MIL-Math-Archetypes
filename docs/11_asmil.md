# ASMIL: Attention-Stabilized MIL

ASMIL identifies a failure mode in attention-based MIL that is independent of
architecture: attention distributions oscillate across training epochs rather
than converging, degrading both stability and performance. It introduces an
EMA anchor model as a stable attention target, a normalized sigmoid attention
function to prevent over-concentration, and token random dropping for
regularization. 2026.

## Anchor Model via EMA

Let $\theta_t$ be the online model parameters at step $t$. An anchor model
with parameters $\theta'_t$ tracks the online model via exponential moving
average:

```math
\theta'_t \leftarrow m\,\theta'_{t-1} + (1-m)\,\theta_t,
```

where $m \in [0,1)$ is the momentum coefficient. The anchor receives the same
inputs but is never updated by backpropagation — gradients are blocked with a
stop-gradient operator.

## Normalized Sigmoid Function (NSF)

Standard softmax attention over scores $z = (z_1,\ldots,z_n)$:

```math
\alpha_i^{\mathrm{smx}}(z;\,T)
=
\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}.
```

ASMIL replaces this in the anchor with a normalized sigmoid:

```math
\alpha_i^{\mathrm{nsf}}(z)
=
\frac{\sigma(z_i)}{\sum_j \sigma(z_j)},
\qquad
\sigma(t) = \frac{1}{1+e^{-t}}.
```

**Theorem 1 (selective flattening):** For any target where high-scoring tokens
should be equalized and low-scoring tokens suppressed, NSF achieves both
simultaneously. Softmax with a single temperature $T$ cannot — raising $T$
equalizes the high end but raises the low end; lowering $T$ suppresses the low
end but sharpens the high end. Sigmoid outputs are bounded per token
independently, decoupling control of high and low ends.

## Attention Stabilization Loss

The online model's softmax attention $\alpha$ is pulled toward the anchor's
NSF attention $\alpha^{\mathrm{nsf}}$ via KL divergence:

```math
\mathcal{L}_{\mathrm{AS}}
=
\mathrm{KL}\!\left(\alpha^{\mathrm{nsf}} \;\|\; \alpha\right).
```

The gradient with respect to online attention score $z_i$:

```math
\frac{\partial \mathcal{L}_{\mathrm{AS}}}{\partial z_i}
=
\alpha_i - \alpha_i^{\mathrm{nsf}}.
```

This pushes the online attention toward the anchor's stable distribution.

## Token Random Dropping

During training, a random fraction $B \in [0,1)$ of the $N$ instance tokens
is dropped:

```math
|\mathcal{P}_{\mathrm{keep}}| = \tilde{N} \sim \mathrm{Binomial}(N,\, 1-B).
```

At inference, $B = 0$ and all tokens are used.

## Overall Training Objective

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\beta\,\mathcal{L}_{\mathrm{AS}},
```

where $\mathcal{L}_{\mathrm{CE}}$ is the bag-level cross-entropy and $\beta > 0$
balances the stabilization term. The anchor does not generate its own
classification loss.

## Key Innovation

ASMIL decouples stable target generation (anchor + NSF) from the model being
trained (online + softmax). The EMA ensures the target evolves slowly; NSF
ensures it is not over-concentrated; the KL penalty transfers the stable
distribution to the online model without replacing softmax at inference.

## Inductive Biases

- Attention distributions that are consistent across epochs are better
  training targets than oscillating ones.
- NSF's per-token independence produces more balanced attention than softmax
  for bag-level tasks.
- Token dropping prevents attention collapse onto a fixed small subset.

## Failure Modes

- EMA momentum $m$ is a dataset-dependent hyperparameter.
- Early in training the anchor may itself be unreliable.
- Token dropping reduces effective bag size, which can destabilize training on
  very small cohorts.
- The final inference model still uses softmax, which can still over-concentrate
  at test time.
