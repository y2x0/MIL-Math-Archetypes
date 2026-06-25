# WiKG: Knowledge-aware Graph MIL

WiKG recasts a WSI as a knowledge graph rather than an unordered bag. Each
patch has two separate learned projections — a head embedding and a tail
embedding — derived from knowledge graph embedding theory. Neighborhoods are
dynamic and content-dependent, not fixed by spatial coordinates. CVPR 2024.

## Head and Tail Projections

For patch $i$ with encoder output $f_i \in \mathbb{R}^{D}$:

```math
h_i = W_h f_i \in \mathbb{R}^{d},
\qquad
t_i = W_t f_i \in \mathbb{R}^{d},
```

where $W_h, W_t \in \mathbb{R}^{d \times D}$ are independent learnable
matrices. The head embedding represents a node's query role; the tail
represents its key/value role when referenced by others.

## Dynamic Neighbor Selection

The affinity of node $j$ as a neighbor of node $i$ is:

```math
\omega(i,j)
=
\frac{h_i^\top t_j}{\sum_{j'=1}^{n} h_i^\top t_{j'}}.
```

The top-$k$ nodes with highest $\omega(i,\cdot)$ form the dynamic neighborhood
$\mathcal{N}(i)$. Unlike spatial $k$-NN, this neighborhood is
content-dependent and different for every node.

## Edge Embedding

For each directed edge $(i \to j)$, the edge embedding interpolates between
the neighbor's tail and the source's head:

```math
r(i,j)
=
\omega(i,j)\, t_j
+
\bigl(1 - \omega(i,j)\bigr) h_i.
```

## Knowledge-Aware Attention

Relevance of neighbor $j$ to node $i$ given edge $r(i,j)$:

```math
u(h_i, r(i,j), t_j)
=
t_j^\top \tanh(h_i + r(i,j)).
```

Normalized attention weights:

```math
\pi(i,j)
=
\frac{
\exp\!\bigl(u(h_i, r(i,j), t_j)\bigr)
}{
\sum_{j' \in \mathcal{N}(i)}
\exp\!\bigl(u(h_i, r(i,j'), t_{j'})\bigr)
}.
```

Neighborhood aggregation:

```math
h_{\mathcal{N}(i)}
=
\sum_{j \in \mathcal{N}(i)}
\pi(i,j)\, t_j.
```

## Node Update: Dual-Interaction Mechanism

Updated head embedding combines additive and multiplicative branches:

```math
h_i'
=
\sigma_1\!\bigl(W_1(h_i + h_{\mathcal{N}(i)})\bigr)
+
\sigma_2\!\bigl(W_2(h_i \odot h_{\mathcal{N}(i)})\bigr).
```

The additive branch captures presence/absence; the multiplicative branch
captures co-activation of features.

## Bag-Level Readout and Loss

A global readout (e.g., mean pooling or ABMIL attention) produces the
slide-level prediction:

```math
\hat{Y} = \mathrm{Softmax}(\mathrm{Readout}(G)).
```

Cross-entropy loss over $M$ slides and $C$ classes:

```math
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{M}
\sum_{m=1}^{M}
\sum_{c=1}^{C}
Y_{m,c}\log\hat{Y}_{m,c}.
```

## Key Innovation

In Patch-GCN, edges are fixed by spatial $k$-NN on coordinates. In WiKG, the
graph is rebuilt at every forward pass from head-tail affinity. The edge
embedding $r(i,j)$ is itself a differentiable function of the model
parameters, making graph structure jointly trainable with the node features.

## Inductive Biases

- Relevant context is defined by feature similarity, not spatial proximity.
- Directed head-to-tail relationships are more expressive than symmetric
  adjacency.
- Additive and multiplicative node updates capture complementary interaction
  modes.
- Top-$k$ selection imposes hard sparsity that suppresses noisy neighbors.

## Failure Modes

- Dynamic $k$-NN requires all-pairs $h_i^\top t_j$: $O(n^2)$ per slide.
- Top-$k$ selection is non-differentiable; gradients do not flow through
  neighbor selection.
- Without spatial constraints, distant but texturally similar patches may
  dominate neighborhoods.
- Stacking layers risks oversmoothing; a single layer limits the receptive
  field.
