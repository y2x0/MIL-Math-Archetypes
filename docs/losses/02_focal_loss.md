# 02 — Focal Loss (Lin et al., ICCV 2017)

Definition:

$$\boxed{\ \mathrm{FL}=-\alpha_t(1-p_t)^{\gamma}\log p_t\ },\qquad \gamma\ge0,$$
$$\alpha_t=\begin{cases}\alpha,&y=1\\1-\alpha,&y=0\end{cases}\ (\text{binary}),\qquad
\alpha_t=w[y]\ (\text{multiclass}).$$

Relative weight vs CE $=(1-p_t)^\gamma$. $\gamma=0\Rightarrow$ CE. Paper: $\gamma=2,\alpha=0.25$.

$$(1-p_t)^2:\quad p_t{=}.5\!\to\!.25,\quad .9\!\to\!10^{-2},\quad .99\!\to\!10^{-4},\quad .1\!\to\!.81.$$

## Gradient (binary, $y=1$, $p=\sigma(x)$, $p'=p(1-p)$)

$$\mathrm{FL}=-(1-p)^\gamma\log p,\qquad u=(1-p)^\gamma,\ v=\log p.$$
$$u'=\gamma(1-p)^{\gamma-1}(-p')=-\gamma\,p(1-p)^\gamma,\qquad v'=\tfrac1p p'=(1-p).$$
$$\frac{d\mathrm{FL}}{dx}=-(u'v+uv')=-\big[-\gamma p(1-p)^\gamma\log p+(1-p)^\gamma(1-p)\big]$$
$$\boxed{\ \frac{d\mathrm{FL}}{dx}=(1-p)^{\gamma}\big(\gamma\,p\log p+p-1\big)\ }$$

Checks:

$$\gamma=0:\ \tfrac{d\mathrm{FL}}{dx}=p-1=p-y\ (\text{CE});\qquad
p\to1:\ (1-p)^\gamma\!\to\!0\ (\text{suppress});\qquad
p\to0:\ \to-1\ (\text{keep}).$$

General sign: $\dfrac{d\mathrm{FL}}{dx}=\pm(1-p_t)^\gamma(\gamma p_t\log p_t+p_t-1).$

## Multiclass

$$\mathrm{FL}=-w_y(1-p_y)^\gamma\log p_y,$$
$$\frac{\partial\mathrm{FL}}{\partial\zeta_k}=w_y(p_k-\delta_{yk})(1-p_y)^{\gamma-1}\big[(1-p_y)-\gamma p_y\log p_y\big].$$

$=$ CE grad $(p_k-\delta_{yk})\times$ envelope $\to0$ as $p_y\to1$.

## Knobs
$\gamma$: within-class (easy↓hard↑), class-blind. $\alpha$: between-class const. Orthogonal; $\gamma\!\uparrow\Rightarrow\alpha\!\downarrow$. Init bias to prior $\pi\!\approx\!0.01$ stabilizes.

## Example
$\gamma{=}2,\alpha{=}.75,x{=}.3\Rightarrow p{=}.5744$:
$$\mathrm{FL}=-.75(.4256)^2\log(.5744)=0.0753\ \ (\text{wCE}=.4158,\ 5.5\times\downarrow).$$
