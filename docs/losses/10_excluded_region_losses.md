# 10 — Dice / Tversky / Focal-Tversky (excluded)

Soft set over $N$ locations, pred $p\in[0,1]^N$, GT $g\in\{0,1\}^N$:

$$\mathrm{TP}=\sum_i p_i g_i,\quad \mathrm{FP}=\sum_i p_i(1-g_i),\quad \mathrm{FN}=\sum_i(1-p_i)g_i.$$

Dice (Milletari'16, Sudre'17):

$$\mathrm{Dice}=\frac{2\mathrm{TP}+\epsilon}{\sum_i p_i+\sum_i g_i+\epsilon},\qquad \mathcal L_{\mathrm{Dice}}=1-\mathrm{Dice}.$$

Tversky (Salehi'17):

$$\mathrm T=\frac{\mathrm{TP}}{\mathrm{TP}+\alpha\mathrm{FP}+\beta\mathrm{FN}},\quad \alpha{+}\beta\ \text{tune FP/FN};\ \ \alpha{=}\beta{=}.5\Rightarrow\text{Dice}.$$

Focal-Tversky (Abraham'19): $\ \mathcal L_{\mathrm{FT}}=(1-\mathrm T)^{1/\gamma},\ \gamma\approx4/3$.

## Why excluded
Bag-level MIL: **1 scalar pred/slide, `batch_size=1`**, no per-instance GT $g$.

$$\sum_i(\cdot)\ \text{has one term}\Rightarrow \mathrm{TP/FP/FN}\ \text{degenerate},\ \nabla\approx0;$$
$$\text{no mask } g\ \text{at bag level};\qquad \text{non-decomposable}\Rightarrow\text{needs real batch}.$$

Valid iff instance/patch-seg head exists ($\beta>\alpha$ for rare positive patches). Else use 02–08.
