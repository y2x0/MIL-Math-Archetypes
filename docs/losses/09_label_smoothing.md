# 09 — Label Smoothing (Szegedy et al., CVPR 2016)

Soft target:

$$q_c=(1-\varepsilon)\delta_{yc}+\frac{\varepsilon}{C},\qquad
\mathrm{LS}=-\sum_c q_c\log p_c=(1-\varepsilon)\mathrm{CE}+\frac{\varepsilon}{C}\sum_c(-\log p_c).$$

Optimum (set grad $=0$):

$$p^\star_y=1-\varepsilon+\tfrac{\varepsilon}{C},\qquad p^\star_{k\ne y}=\tfrac{\varepsilon}{C},$$
$$\zeta_y-\zeta_k=\log\frac{p^\star_y}{p^\star_k}=\log\!\Big((C-1)\tfrac{1-\varepsilon}{\varepsilon}+1\Big)<\infty.$$

(Plain CE $\Rightarrow$ gap $\to\infty$.)

Gradient:

$$\frac{\partial\mathrm{LS}}{\partial\zeta_k}=p_k-q_k=p_k-\Big[(1-\varepsilon)\delta_{yk}+\tfrac{\varepsilon}{C}\Big].$$

$\varepsilon=0.1$. Regularizer: bounded gap $\Rightarrow$ calibration↑, overconfidence↓. Class-blind ($\varepsilon/C$ uniform) $\Rightarrow$ not an imbalance fix; composes with 02 (steadies focal) and 07/08 (orthogonal: target vs logits). Binary: soft BCE targets $\{\varepsilon,1-\varepsilon\}$.
