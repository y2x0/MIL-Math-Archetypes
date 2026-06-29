# 03 — Class-Balanced / Effective Number (Cui et al., CVPR 2019)

Volume model, $N=$ max prototypes, $\beta=\frac{N-1}{N}$. Recurrence:

$$E_n=1+\beta E_{n-1},\quad E_1=1\ \Rightarrow\ E_n=\sum_{j=0}^{n-1}\beta^j=\boxed{\frac{1-\beta^{\,n}}{1-\beta}}.$$

Limits:

$$\beta\to0:\ E_n=1\ (\text{no reweight});\quad
\beta\to1:\ E_n\to n\ (\text{inv-freq});\quad
n\to\infty:\ E_n\to\tfrac1{1-\beta}\ (\text{saturate}).$$

Weight + loss:

$$w_c\propto\frac1{E_{n_c}}=\frac{1-\beta}{1-\beta^{\,n_c}},\qquad
\mathrm{CB}(\mathcal L)=w_y\,\mathcal L.$$

$$\mathrm{CB\text{-}focal}=-w_y(1-p_t)^\gamma\log p_t.$$

Saturation scale $=\frac1{1-\beta}$:

$$\beta:\ .9\!\to\!10,\ .99\!\to\!10^2,\ .999\!\to\!10^3,\ .9999\!\to\!10^4.$$

Ex. counts $80/15/5,\ \beta{=}.9999$: $E\approx79.7/15/5\Rightarrow w\approx0.14/0.72/2.15$ (mean-norm).

Rule: set $\frac1{1-\beta}\approx n_{\min}$. Pure reweight (idea #2); composes with any base; halve if also using weighted sampler.
