# 04 — Asymmetric Loss / ASL (Ben-Baruch et al., ICCV 2021)

Per-label, $p=\sigma(\zeta)$, $y\in\{0,1\}$.

Symmetric focal: $-y(1-p)^\gamma\log p-(1-y)p^\gamma\log(1-p)$, same $\gamma$.

## Change 1 — decouple focusing
$$-\,y(1-p)^{\gamma_+}\log p\;-\;(1-y)\,p^{\gamma_-}\log(1-p),\qquad \gamma_-> \gamma_+.$$
Default $\gamma_+{=}0$ (keep positives), $\gamma_-{=}4$ (kill easy negs).

## Change 2 — hard margin (prob shift)
$$p_m=\max(p-m,\,0),\qquad m\approx0.05.$$

## Full ASL
$$\boxed{\ \mathrm{ASL}=-y(1-p)^{\gamma_+}\log p-(1-y)(p_m)^{\gamma_-}\log(1-p_m)\ }$$

Equiv. official: $x_{\mathrm{neg}}=\min(1-p+m,1),\ p_t=p\,y+x_{\mathrm{neg}}(1-y)$, weight $(1-p_t)^{\gamma}$, $\gamma=\gamma_+y+\gamma_-(1-y)$.

## Negative-term gradient ($y=0$)
$$L=-(p_m)^{\gamma_-}\log(1-p_m):\quad
p\le m\Rightarrow p_m{=}0\Rightarrow L{=}0,\ \nabla{=}0\ (\text{hard cut});$$
$$p\to0,\ p>m:\ \nabla\sim p_m^{\gamma_-}\!\to0;\qquad p\ \text{large (hard neg)}:\ \nabla\ \text{kept}.$$

Num. $\gamma_-{=}4,m{=}.05$: hard neg $\zeta{=}1\!\to\!0.25$; easy neg $\zeta{=}{-}3\!\to\!\approx0$.

| | $\gamma_+$ | $\gamma_-$ | $m$ |
|--|--|--|--|
| default | 0 | 4 | 0.05 |

Multiclass: one-vs-rest over one-hot, sum over $C$. Use: heavy-neg binary / multi-label. $m$ also clips label-noise positives; keep $m\le0.2$.
