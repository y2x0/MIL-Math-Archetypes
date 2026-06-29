# 01 — Cross Entropy / BCE

MLE:

$$\max_\theta\prod_i p_\theta(y_i|x_i)\ \Longleftrightarrow\ \min_\theta\sum_i-\log p_\theta(y_i|x_i),
\qquad \mathrm{CE}=-\sum_c\delta_{yc}\log p_c=-\log p_t.$$

Softmax Jacobian:

$$p_c=\frac{e^{\zeta_c}}{\sum_j e^{\zeta_j}},\qquad
\frac{\partial p_c}{\partial\zeta_k}=p_c(\delta_{ck}-p_k).$$

Gradient:

$$\frac{\partial\,\mathrm{CE}}{\partial\zeta_k}
=-\frac1{p_y}\frac{\partial p_y}{\partial\zeta_k}
=-\frac1{p_y}p_y(\delta_{yk}-p_k)
=\boxed{p_k-\delta_{yk}}.$$

Binary, $p=\sigma(\zeta)$:

$$\mathrm{BCE}=-[y\log p+(1-y)\log(1-p)]=-\log p_t,\qquad
\frac{\partial\,\mathrm{BCE}}{\partial\zeta}=p-y.$$

Weighting:

$$\mathrm{CE}_w=-w_y\log p_t,\qquad \frac{\partial}{\partial\zeta_k}=w_y(p_k-\delta_{yk}),$$
$$w_c=\Big(\frac{\sum_j n_j}{C\,n_c}\Big)^{\!\rho}\ \ (\text{mean-norm.}),\qquad
\text{binary } \mathrm{pos\_weight}=(n_-/n_+)^{\rho}.$$

Failure mode: gradient $p_k-\delta_{yk}$ is **linear** in error; $\sum$ over many easy negatives ($p_k\!\approx\!0$, but $\#\!\to\!\infty$) dominates. Three fixes:

$$\underbrace{(1-p_t)^\gamma,\ \text{poly},\ \text{ASL}}_{\text{reshape per-ex. grad (02,04,05)}}\quad
\underbrace{w_c=\tfrac{1-\beta}{1-\beta^{n_c}}}_{\text{reshape weight (03)}}\quad
\underbrace{\zeta\to\zeta-\Delta_y,\ \zeta+\tau\log\pi}_{\text{shift boundary (06,07,08)}}$$
