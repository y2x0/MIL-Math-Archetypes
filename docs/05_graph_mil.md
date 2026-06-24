# Graph MIL

Graph MIL makes patch geometry explicit. Patches are nodes, spatial
neighborhoods are edges, and aggregation happens after message passing.

## Graph Construction

Let each patch have coordinates:

```math
r_i=(x_i,y_i)\in\mathbb{R}^2.
```

Construct a spatial \(k\)-nearest-neighbor graph:

```math
\mathcal N(i)
=
\mathrm{kNN}_k(r_i;\{r_j\}_{j=1}^{n}).
```

The graph is:

```math
G=(V,E,H),
\qquad
V=\{1,\ldots,n\},
\qquad
E=\{(i,j):j\in\mathcal N(i)\}.
```

## Message Passing

A general message-passing layer is:

```math
m_i^{(\ell)}
=
\rho^{(\ell)}
\left(
\{
\phi^{(\ell)}(h_i^{(\ell)},h_j^{(\ell)},e_{ij})
:
j\in\mathcal N(i)
\}
\right),
```

```math
h_i^{(\ell+1)}
=
\gamma^{(\ell)}
\left(
h_i^{(\ell)},m_i^{(\ell)}
\right).
```

Here:

- \(\phi\) computes messages.
- \(\rho\) aggregates messages.
- \(\gamma\) updates the node.

## Graph Convolution

A simple normalized graph convolution is:

```math
H^{(\ell+1)}
=
\sigma
\left(
\widetilde D^{-1/2}
\widetilde A
\widetilde D^{-1/2}
H^{(\ell)}
W^{(\ell)}
\right),
```

where:

```math
\widetilde A=A+I,
\qquad
\widetilde D_{ii}=\sum_j\widetilde A_{ij}.
```

## Graph Attention

Graph attention computes feature-dependent edge scores:

```math
e_{ij}
=
\mathrm{LeakyReLU}
\left(
a^\top[Wh_i\Vert Wh_j]
\right),
\qquad
j\in\mathcal N(i).
```

Normalize over neighbors:

```math
\alpha_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{k\in\mathcal N(i)}\exp(e_{ik})}.
```

Update:

```math
h_i'
=
\sigma
\left(
\sum_{j\in\mathcal N(i)}
\alpha_{ij}Wh_j
\right).
```

## Bag Readout

After \(L\) graph layers:

```math
\widetilde H=H^{(L)}.
```

Use a global readout:

```math
z
=
\mathrm{READOUT}(\widetilde H).
```

Common choices:

```math
z=\frac{1}{n}\sum_i\widetilde h_i,
\qquad
z=\sum_i a_i\widetilde h_i,
\qquad
z=\max_i \widetilde h_i.
```

Then:

```math
p=\mathrm{softmax}(W_o z+b_o).
```

## Loss

```math
\mathcal L_{\mathrm{CE}}
=
-\sum_c y_c\log p_c.
```

For survival prediction, graph MIL often uses a discrete hazard head or Cox
loss, but that is a task head choice rather than a graph aggregation necessity.

## Mathematical Innovation

Graph MIL restricts interactions by geometry:

```math
j\to i
\quad\text{only if}\quad
j\in\mathcal N(i).
```

This replaces dense \(n^2\) interactions with sparse local message passing:

```math
O(|E|)\approx O(kn).
```

## Inductive Bias

- Spatial neighbors carry useful context.
- Local tissue architecture matters.
- Patch features should be contextualized before bag readout.
- Long-range information is built by stacking layers.

## Failure Mode

Stacking too many graph layers can oversmooth:

```math
h_i^{(L)}\approx h_j^{(L)}
```

for many nodes \(i,j\). Stacking too few layers causes oversquashing: many
distant signals must pass through small local bottlenecks.
