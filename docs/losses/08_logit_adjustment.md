# 08 — Logit Adjustment (Menon et al., ICLR 2021)

Balanced error $\Rightarrow$ uniform test prior. Bayes-optimal rule:

$$\arg\max_c\frac{p(y{=}c|x)}{\pi_c}=\arg\max_c\big[\log p(y{=}c|x)-\log\pi_c\big].$$

**(a) Post-hoc:** predict $\arg\max_c[\,f_c(x)-\tau\log\pi_c\,]$.

**(b) Loss (shift inside softmax):**

$$\boxed{\ \mathrm{LA}=-\log\frac{e^{\zeta_y+\tau\log\pi_y}}{\sum_j e^{\zeta_j+\tau\log\pi_j}}=\mathrm{CE}(\zeta+\tau\log\boldsymbol\pi,\,y)\ }$$

Gradient:

$$\frac{\partial\mathrm{LA}}{\partial\zeta_k}=\tilde p_k-\delta_{yk},\qquad \tilde p=\mathrm{softmax}(\zeta+\tau\log\boldsymbol\pi).$$

Binary: $\zeta'=\zeta+\tau(\log\pi_1-\log\pi_0),\ \mathrm{BCE}(\zeta',y)$.

$$\tau=0:\ \mathrm{CE};\qquad \tau=1:\ \text{Balanced Softmax (file 07)};\qquad \tau>1:\ \text{over-correct}.$$

Tune $\tau\in[0.5,2]$. Fisher-consistent for balanced error — changes the **rule**, not just grad magnitude (unlike 02–05). Start $\tau=1$.
