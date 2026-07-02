# DSMIL: Dual-Stream MIL

DSMIL runs two readouts in parallel and averages them. The first stream is a
max-pooling instance classifier that selects a single **critical instance**; the
second stream scores every patch by its similarity to that critical instance and
pools an attention-weighted bag embedding. Combining a max statistic with a
weighted first moment lets the model react to rare strong evidence while still
using the whole bag. CVPR 2021.

## Stream 1: Critical Instance by Max Pooling

Each patch gets an instance score from a shared linear classifier:

```math
h_i\in\mathbb{R}^{d},
\qquad
c_i=W_0\,h_i\in\mathbb{R}^{C}.
```

The critical instance is the arg-max over the bag, and its score is the
max-pooled logit:

```math
m=\arg\max_{i}\,\max_c\,(c_i)_c,
\qquad
s_{\max}=c_m.
```

This stream alone is a max-pooling MIL model: it preserves extreme evidence but
ignores everything except one patch.

## Stream 2: Attention to the Critical Instance

Project every patch into a query and an information (value) vector:

```math
q_i=W_q\,h_i\in\mathbb{R}^{d_q},
\qquad
v_i=W_v\,h_i\in\mathbb{R}^{d_v}.
```

Let $q_m$ be the query of the critical instance. The similarity of each patch to
the critical instance defines the attention weights:

```math
U(h_i,h_m)=\langle q_i,\,q_m\rangle,
\qquad
a_i=\frac{\exp\!\big(\langle q_i,q_m\rangle\big)}
{\sum_{j=1}^{n}\exp\!\big(\langle q_j,q_m\rangle\big)}.
```

The bag embedding is the attention-weighted sum of information vectors, followed
by a bag classifier:

```math
b=\sum_{i=1}^{n}a_i\,v_i,
\qquad
s_{\mathrm{bag}}=W_b\,b.
```

Unlike ABMIL, the attention query is not learned in isolation — it is anchored
to the data-dependent critical instance, so $a_i$ measures agreement with the
most suspicious patch rather than a free relevance score.

## Combined Prediction and Loss

The final logit averages the two streams:

```math
s=\tfrac12\big(s_{\max}+s_{\mathrm{bag}}\big),
\qquad
p=\mathrm{softmax}(s).
```

Both streams are supervised, so the objective sums a max-stream and a
bag-stream loss:

```math
\mathcal L
=
\tfrac12\,\mathcal L_{\mathrm{CE}}(s_{\max},Y)
+
\tfrac12\,\mathcal L_{\mathrm{CE}}(s_{\mathrm{bag}},Y).
```

## Mathematical Innovation

DSMIL fuses two different surviving statistics in one readout:

```math
\underbrace{s_{\max}=\max_i c_i}_{\text{extreme evidence}}
\qquad
\underbrace{s_{\mathrm{bag}}=W_b\sum_i a_i v_i}_{\text{weighted first moment}}.
```

The attention measure is centered on the arg-max patch,

```math
a_i=\mathrm{softmax}_i\,\langle q_i,q_m\rangle,
```

so the first-moment stream is computed *relative to* the extreme-evidence
stream instead of independently. This couples presence detection (does any
patch look diseased?) with amount estimation (how much of the bag agrees?).

## Inductive Bias

- The bag is unordered; both streams are permutation invariant.
- A single most-suspicious patch is a meaningful anchor for the task.
- Patches similar to the critical instance carry corroborating evidence.
- Averaging max and mean statistics is better than committing to either.

## Failure Mode

The whole second stream hangs on one arg-max choice:

```math
m=\arg\max_i\,\max_c\,(c_i)_c.
```

If the critical instance is a false positive (staining artifact, edge patch),
the attention map anchors to noise and $b$ is corrupted. The arg-max is also
non-smooth: near-ties make $m$ jump between patches across steps, which can
destabilize $a_i$. And when disease is diffuse rather than focal, a single
critical instance underrepresents the bag — the same limitation as pure max
pooling, only partly mitigated by the averaged bag stream.
