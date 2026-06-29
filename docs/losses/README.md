# Loss Math

Notation. One example, true class $y$, logits $\zeta$.

$$p=\mathrm{softmax}(\zeta),\quad p_c=\frac{e^{\zeta_c}}{\sum_j e^{\zeta_j}},\quad
\sigma(x)=\frac{1}{1+e^{-x}},\quad \sigma'=\sigma(1-\sigma).$$

$$p_t=\begin{cases}p_y & \text{multiclass}\\ \sigma(\zeta)^{y}(1-\sigma(\zeta))^{1-y} & \text{binary}\end{cases}
=\Pr[\text{true class}],\qquad \mathrm{CE}=-\log p_t.$$

$n_c=$ train count of class $c$, $\ \pi_c=n_c/\sum_j n_j,\ \ C=\#$classes.

Master identities:

$$\frac{\partial p_c}{\partial\zeta_k}=p_c(\delta_{ck}-p_k),\qquad
\boxed{\frac{\partial\,\mathrm{CE}}{\partial\zeta_k}=p_k-\delta_{yk}},\qquad
\frac{\partial\,\mathrm{BCE}}{\partial\zeta}=\sigma(\zeta)-y.$$

Every loss below $=$ a distortion of $p_k-\delta_{yk}$.

| # | Loss | Paper | Core formula |
|---|---|---|---|
| 01 | CE / BCE | — | $-w_y\log p_t$ |
| 02 | Focal | Lin ICCV'17 | $-\alpha_t(1-p_t)^\gamma\log p_t$ |
| 03 | Class-Balanced | Cui CVPR'19 | $w_c=\frac{1-\beta}{1-\beta^{n_c}}$ |
| 04 | ASL | Ben-Baruch ICCV'21 | $\gamma_+\!\ne\!\gamma_-,\ p_m=\max(p-m,0)$ |
| 05 | Poly-1 | Leng ICLR'22 | $\mathrm{CE}+\varepsilon(1-p_t)$ |
| 06 | LDAM | Cao NeurIPS'19 | $\Delta_c\propto n_c^{-1/4}$ |
| 07 | Balanced Softmax | Ren NeurIPS'20 | $\mathrm{CE}(\zeta+\log\mathbf n,y)$ |
| 08 | Logit Adjust | Menon ICLR'21 | $\mathrm{CE}(\zeta+\tau\log\boldsymbol\pi,y)$ |
| 09 | Label Smoothing | Szegedy CVPR'16 | $q_c=(1-\varepsilon)\delta_{yc}+\varepsilon/C$ |
| 10 | Dice/Tversky | — (excluded) | $\mathrm{TP}/(\mathrm{TP}+\alpha\mathrm{FP}+\beta\mathrm{FN})$ |
