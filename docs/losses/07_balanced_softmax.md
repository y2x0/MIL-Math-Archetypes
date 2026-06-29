# 07 — Balanced Softmax (Ren et al., NeurIPS 2020)

Shift: train prior $\pi^{\mathrm{tr}}_c\propto n_c$, test prior uniform.

Want net to model class-conditional: $e^{\phi_c(x)}\propto p(x|y{=}c)$. Bayes:

$$p^{\mathrm{tr}}(y{=}c|x)=\frac{\pi^{\mathrm{tr}}_c\,p(x|y{=}c)}{\sum_j\pi^{\mathrm{tr}}_j p(x|y{=}j)}
=\frac{n_c e^{\phi_c}}{\sum_j n_j e^{\phi_j}}
=\mathrm{softmax}(\phi+\log\mathbf n)_c.$$

Loss (train on shifted, test on $\phi$):

$$\boxed{\ \mathrm{BalSoftmax}=-\log\frac{n_y e^{\zeta_y}}{\sum_j n_j e^{\zeta_j}}=\mathrm{CE}(\zeta+\log\mathbf n,\,y)\ }$$

($\log n_c$ vs $\log\pi_c$ differ by const, softmax-invariant.)

Gradient:

$$\frac{\partial}{\partial\zeta_k}=\tilde p_k-\delta_{yk},\qquad \tilde p=\mathrm{softmax}(\zeta+\log\mathbf n).$$

Binary: $\zeta'=\zeta+(\log n_1-\log n_0),\ \mathrm{BCE}(\zeta',y)$.

Zero hyperparams. $=\tau{=}1$ case of file 08.
