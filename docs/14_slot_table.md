# Master Slot Table

Every note in this repo can be read through the same four slots from the
[common notation](00_common_notation.md):

```math
\widetilde H=\mathcal C(H;G,S),
\qquad
z=\mathcal R(\widetilde H),
\qquad
\eta=W_o z+b_o,
\qquad
p=\mathrm{softmax}(\eta).
```

- `C`: context operator — how a patch embedding is updated using other patches.
- `R`: readout — how a set or sequence collapses to one bag vector.
- `G`: geometry used, such as grid coordinates or graph edges.
- `S`: sequence order used.

The core archetypes (docs 01–05) differ only in `C` and `R`. The extended
methods (docs 08–13) mostly keep an archetype's `C`/`R` and change the
*supervision* or add a *pre-processing* stage. This table places every method in
the same coordinates so the additions are visible as deltas, not new worlds.

## Core Archetypes

| Doc | Method | Context operator `C` | Readout `R` | Uses `G` | Uses `S` | Surviving statistic |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Mean-pooling MIL | per-patch `φ(h_i)` | arithmetic mean | no | no | first moment |
| 02 | Attention MIL (ABMIL) | per-patch `φ(h_i)` | learned measure `Σ a_i h_i` | no | no | weighted first moment |
| 03 | Transformer MIL | dense pairwise `Σ_j A_ij V h_j` | class token / attention pool | encoded | optional | pairwise correlations |
| 04 | State-space MIL | selective scan `s_t=Ā s_{t-1}+B̄ h_t` | attention / terminal state | no | yes | ordered compressed memory |
| 05 | Graph MIL | sparse message passing over `N(i)` | global pooling | yes | no | local topology |

## Extended Methods

| Doc | Method | Context operator `C` | Readout `R` | Uses `G` | Uses `S` | Added supervision / stage |
| --- | --- | --- | --- | --- | --- | --- |
| 08 | DTFD-MIL | ABMIL per-patch | two-tier attention pool | no | no | pseudo-bag split + distilled tier-2 |
| 09 | RRT-MIL | regional + cross-region attention (re-embed) | ABMIL pool on re-embedded `Z` | grid | no | drop-in `R(·)` before aggregator |
| 10 | WiKG | knowledge-aware dynamic graph message passing | graph global pool | learned graph | no | edges predicted from embeddings |
| 11 | ASMIL | ABMIL per-patch | softmax attention pool | no | no | EMA anchor + NSF + KL stabilizer |
| 12 | CLAM | per-patch, class-specific attention | per-class weighted first moment | no | no | instance clustering on top/bottom-k |
| 13 | DSMIL | per-patch + critical-instance attention | max stream + weighted-mean stream | no | no | both streams supervised |

## Reading the Deltas

Several patterns fall out of the table:

- **Same `C` and `R`, different objective.** DTFD, CLAM, DSMIL, and ASMIL are
  all built on ABMIL's per-patch context and (weighted) first-moment readout.
  What changes is supervision: pseudo-bags (DTFD), instance clustering (CLAM),
  a second max stream (DSMIL), or an attention-stabilizing penalty (ASMIL).

```math
\text{ABMIL readout fixed}
\quad\Longrightarrow\quad
\text{method}=\text{choice of auxiliary loss}.
```

- **New `C` as a pre-processor.** RRT-MIL leaves the readout to ABMIL and only
  inserts a context operator that enriches tokens first,

```math
\hat Y=C_{\mathrm{cls}}\big(A(R(F(X)))\big).
```

- **Geometry is where `G` enters.** Only Transformer MIL (via positional
  encoding), Graph MIL, RRT (grid reshape), and WiKG (learned edges) use `G`.
  Everything else is geometry-free.

- **Order is where `S` enters.** Only state-space MIL treats `S` as a real
  input, which is exactly why it is the one archetype that breaks permutation
  invariance.

## Using the Table to Design

To propose a new aggregator, fill one row:

```math
\boxed{
\big(\mathcal C,\ \mathcal R,\ G,\ S,\ \text{surviving statistic},\ \text{auxiliary loss}\big).
}
```

If two existing rows share every entry, they are the same model with different
names. If a proposed row duplicates an existing row, the novelty is elsewhere —
usually in the encoder or the data, not the aggregator. See
[design axes](07_design_axes.md) for choosing each entry deliberately.
