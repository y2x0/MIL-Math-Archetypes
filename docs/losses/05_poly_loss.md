# 05 — PolyLoss (Leng et al., ICLR 2022)

Mercator $-\log(1-z)=\sum_{j\ge1}\frac{z^j}{j}$, set $z=1-p_t$:

$$\mathrm{CE}=-\log p_t=\sum_{j\ge1}\frac1j(1-p_t)^j=(1-p_t)+\tfrac12(1-p_t)^2+\cdots$$

Focal $=$ same series, shifted:

$$\mathrm{FL}=(1-p_t)^\gamma\mathrm{CE}=\sum_{j\ge1}\frac1j(1-p_t)^{\,j+\gamma}.$$

## Poly-1: perturb leading coeff $1\to1+\varepsilon$
$$\boxed{\ \mathrm{Poly\text{-}1}=\mathrm{CE}+\varepsilon(1-p_t)=-\log p_t+\varepsilon(1-p_t)\ }$$
$$\mathrm{Poly\text{-}1\,Focal}=\mathrm{FL}+\varepsilon(1-p_t)^{\gamma+1}.$$

## Gradient
$\frac{\partial(1-p_t)}{\partial\zeta_k}=p_t(p_k-\delta_{yk})$, so

$$\frac{\partial\,\mathrm{Poly\text{-}1}}{\partial\zeta_k}=(p_k-\delta_{yk})\big(1+\varepsilon\,p_t\big).$$

$\varepsilon>0\Rightarrow$ amplify confident-example grad ($\propto p_t$) — opposite of focal. $\varepsilon=0\Rightarrow$ base loss exactly.

Range $\varepsilon\in[-1,2]$; defaults $\varepsilon\approx1$ (CE base), $\approx2$ (det/seg). One knob, no schedule, drop-in. Smaller gains than 06–08 on extreme tails.
