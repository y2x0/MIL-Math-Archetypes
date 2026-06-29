# 06 — LDAM (Cao et al., NeurIPS 2019)

Margin bound: test-error term $\sim\widetilde{\mathcal O}\!\big(\frac{1}{\gamma_c\sqrt{n_c}}\big)$.

Minimize $\sum_c\frac1{\gamma_c\sqrt{n_c}}$ s.t. $\sum_c\gamma_c=B$:

$$\partial_{\gamma_c}\!\Big[\tfrac1{\gamma_c\sqrt{n_c}}+\lambda\gamma_c\Big]=0
\Rightarrow-\tfrac1{\gamma_c^2\sqrt{n_c}}+\lambda=0
\Rightarrow\gamma_c^2=\tfrac1{\lambda\sqrt{n_c}}
\Rightarrow\boxed{\gamma_c\propto n_c^{-1/4}}.$$

Margin + loss:

$$\Delta_c=\frac{C_0}{n_c^{1/4}}\ \ (\max_c\Delta_c=\Delta_{\max}),\qquad
\zeta'_y=\zeta_y-\Delta_y,\ \ \zeta'_{k\ne y}=\zeta_k,$$
$$\boxed{\ \mathrm{LDAM}=-\log\frac{e^{\zeta_y-\Delta_y}}{e^{\zeta_y-\Delta_y}+\sum_{k\ne y}e^{\zeta_k}}=\mathrm{CE}(s\zeta',y)\ }$$

Scale $s$: normalized feats $\Rightarrow s\approx30$; un-normalized MIL head $\Rightarrow s=1$.

Binary: $\zeta'=\zeta-(2y-1)\Delta_y,\ \mathrm{BCE}(s\zeta',y)$.

DRW (2 phases): (1) train LDAM, weights **off**; (2) after first LR drop, weights **on** (CB/inv-freq).

Ex. $80/15/5,\Delta_{\max}{=}.5$: $n^{-1/4}{=}.334/.508/.669\to\Delta\approx.25/.38/.50$. (CE $3.24\to3.73$ on rarest.)

vs 07/08: LDAM shifts logits by fixed margin; prior methods shift by $\log\pi$.
