# DMQ 算法原理：从稠密向量到 SINDI 稀疏向量压缩

本文从数学角度说明两类 DMQ：

1. `DMQ-main` 中面向稠密向量、结合 IVF 残差的 DMQ；
2. VSAG SINDI 中面向稀疏向量内积检索的 DMQ8 与 DMQ1bit。

文中还单独推导 8-bit DMQ 查询表的低秩矩阵分解，说明如何用两个 16 项小表
近似完整的 256 项 LUT，并讨论 rank-$R$ 推广与误差校正。

全文只讨论决定算法性质的中心化、码本、缩放、打分和误差，不讨论类、接口、序列化及
SIMD 实现。若无特别说明，SINDI 部分以当前代码中的“每个 term 一个码本、非零值均值
中心化”方案为准。

## 1. 问题与核心思想

DMQ 要解决的问题是：用很少的 bit 表示向量分量，同时仍能准确估计查询与原向量的
内积或距离。

它与普通均匀标量量化的主要区别有三点：

- 先去掉每条向量沿全一向量方向的均值分量；
- 根据训练样本的加权经验分布选择非均匀代表值；
- 为每条向量增加一个缩放系数，使量化向量与原残差的内积尺度一致。

因此，DMQ 可以概括为

$$
\boxed{
\text{DMQ}
=\text{自均值中心化}
+\text{分布感知码本}
+\text{逐向量尺度校正}
}
$$

## 2. `DMQ-main` 的稠密 DMQ

### 2.1 IVF 残差与自均值中心化

设数据向量为 $x\in\mathbb{R}^D$，查询向量为 $q\in\mathbb{R}^D$，两者所属 IVF
聚类的中心为 $c$。定义

$$
r=x-c,\qquad \rho=q-c.
$$

DMQ 不直接量化 $r$，而是分别去掉 $r$ 和 $\rho$ 的坐标均值：

$$
a_x=\frac{1}{D}\mathbf{1}^{\mathsf T}r,
\qquad
a_q=\frac{1}{D}\mathbf{1}^{\mathsf T}\rho,
$$

$$
\bar r=r-a_x\mathbf{1},
\qquad
\bar\rho=\rho-a_q\mathbf{1}.
$$

等价地，令

$$
P=I-\frac{1}{D}\mathbf{1}\mathbf{1}^{\mathsf T},
$$

则 $\bar r=Pr$、$\bar\rho=P\rho$。矩阵 $P$ 把向量投影到与 $\mathbf 1$ 正交的
$D-1$ 维子空间，所以

$$
\mathbf{1}^{\mathsf T}\bar r
=\mathbf{1}^{\mathsf T}\bar\rho=0.
$$

这一步的意义是：没有随机旋转时，各坐标可能带有共同偏移；去掉全一方向后，码本只需
描述真正变化的部分。

### 2.2 分布感知的加权分位点

设量化位宽为 $b$，码本大小为

$$
M=2^b.
$$

对某个聚类的第 $j$ 维，收集中心化残差样本

$$
V_j=\{v_1,v_2,\ldots,v_N\},
\qquad v_i=\bar r_{i,j}.
$$

DMQ 给每个样本 $v_i$ 分配权重

$$
w_i=Nv_i^2+\sum_{m=1}^{N}v_m^2-2v_i\sum_{m=1}^{N}v_m.
$$

展开可得

$$
\boxed{
w_i=\sum_{m=1}^{N}(v_i-v_m)^2
}.
$$

因此，远离同组其他样本的值拥有更大的权重。与普通等频分位数相比，DMQ 会在对
成对平方差贡献更大的区域分配更多量化分辨率。

定义加权经验分布及其逆函数

$$
F_w(z)=\frac{1}{W}\sum_{i=1}^{N}w_i\,\mathbf 1[v_i\le z],
\qquad
W=\sum_{i=1}^{N}w_i,
$$

$$
Q_w(p)=\inf\{z:F_w(z)\ge p\}.
$$

第 $k$ 个代表值取在 $2M$ 份加权质量中的奇数分割点：

$$
c_{j,k}=Q_w\!\left(\frac{2k+1}{2M}\right),
\qquad k=0,1,\ldots,M-1.
$$

相邻代表值之间的阈值取在偶数分割点：

$$
\tau_{j,k}=Q_w\!\left(\frac{k+1}{M}\right),
\qquad k=0,1,\ldots,M-2.
$$

所以一个 $b$ bit 码本包含 $M$ 个代表值和 $M-1$ 个阈值。例如：

| 位宽 | 代表值数 | 阈值数 | 总参数数/维 |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 1 | 3 |
| 2 | 4 | 3 | 7 |
| 4 | 16 | 15 | 31 |
| 8 | 256 | 255 | 511 |

若所有样本相同，则 $W=0$，码本退化为这个相同值，不产生额外量化误差。

### 2.3 编码、代表向量与逐向量缩放

对 $\bar r_j$，编码为它跨过的阈值数：

$$
k_j=\sum_{m=0}^{M-2}\mathbf 1[\bar r_j>\tau_{j,m}],
\qquad k_j\in\{0,1,\ldots,M-1\}.
$$

解码得到代表向量 $u\in\mathbb{R}^D$：

$$
u_j=c_{j,k_j}.
$$

仅靠 $u$ 不能恢复每条向量不同的模长。DMQ 为每条向量保存尺度

$$
\boxed{
\gamma_x=\frac{\|\bar r\|_2^2}{\langle u,\bar r\rangle}
}.
$$

于是中心化残差近似为

$$
\bar r\approx\gamma_xu,
$$

原向量近似为

$$
\hat x=c+a_x\mathbf 1+\gamma_xu.
$$

这个 $\gamma_x$ 不是最小化 $\|\bar r-\gamma u\|_2^2$ 的普通最小二乘系数；它满足

$$
\left\langle
\bar r-\gamma_xu,\bar r
\right\rangle=0,
$$

即量化误差与原中心化残差正交。其目标是校准后续内积估计，而不只是最小化逐坐标
重建误差。

### 2.4 查询内积与 L2 距离估计

由中心化定义可得

$$
\langle r,\rho\rangle
=\langle\bar r,\bar\rho\rangle+Da_xa_q.
$$

用 $\gamma_xu$ 近似 $\bar r$：

$$
\langle\bar r,\bar\rho\rangle
\approx\gamma_x\langle u,\bar\rho\rangle.
$$

因此平方 L2 距离估计为

$$
\boxed{
\widehat{\|x-q\|_2^2}
=\|r\|_2^2+\|\rho\|_2^2
-2Da_xa_q
-2\gamma_x\langle u,\bar\rho\rangle
}.
$$

查询到来后，为每个维度构造查找表

$$
L_{j,k}=\bar\rho_jc_{j,k}.
$$

于是

$$
\langle u,\bar\rho\rangle
=\sum_{j=1}^{D}L_{j,k_j}.
$$

这说明搜索阶段只需用量化码选择查找表项并求和，不必先完整重建 $u$。

### 2.5 分段与渐进量化

`DMQ-main` 中的 1、2、4、8 bit 以及分段版本共享上述数学核心。

若只计算维度子集 $S$，令 $\delta=a_x-a_q$，则精确分段距离为

$$
\begin{aligned}
\|x_S-q_S\|_2^2
={}&\|\bar r_S\|_2^2+\|\bar\rho_S\|_2^2
-2\langle\bar r_S,\bar\rho_S\rangle\\
&+2\delta\left(
\mathbf 1^{\mathsf T}\bar r_S-
\mathbf 1^{\mathsf T}\bar\rho_S
\right)
+|S|\delta^2.
\end{aligned}
$$

将交叉项替换为

$$
\langle\bar r_S,\bar\rho_S\rangle
\approx\gamma_{x,S}\langle u_S,\bar\rho_S\rangle
$$

即可得到分段 DMQ。分段允许不同维度区间拥有独立尺度，也允许搜索逐段累加得分。

另一类变体把总预算拆成“第一层 1 bit + 后续残差 bit”。令

$$
e^{(0)}=\bar r,
$$

第 $\ell$ 层量化 $e^{(\ell)}$ 得到 $u^{(\ell)}$，并计算

$$
\gamma_\ell=
\frac{\|e^{(\ell)}\|_2^2}
{\langle u^{(\ell)},e^{(\ell)}\rangle},
\qquad
e^{(\ell+1)}=e^{(\ell)}-\gamma_\ell u^{(\ell)}.
$$

经过 $B$ 层后

$$
\bar r\approx
\sum_{\ell=0}^{B-1}\gamma_\ell u^{(\ell)},
$$

$$
\langle\bar r,\bar\rho\rangle
\approx
\sum_{\ell=0}^{B-1}
\gamma_\ell\langle u^{(\ell)},\bar\rho\rangle.
$$

这使得算法可以先用 1 bit 粗估，只有未被排除的候选才继续计算剩余 bit。

### 2.6 误差解释

定义中心化量化误差

$$
e=\bar r-\gamma_xu.
$$

内积估计误差恰好为

$$
\langle\bar r,\bar\rho\rangle
-\gamma_x\langle u,\bar\rho\rangle
=\langle e,\bar\rho\rangle.
$$

确定性的 Cauchy--Schwarz 界为

$$
|\langle e,\bar\rho\rangle|
\le\|e\|_2\|\bar\rho\|_2.
$$

在随机方向或近似各向同性假设下，`DMQ-main` 使用更紧的高维集中形式：

$$
\Pr\!\left(
|\langle e,\bar\rho\rangle|>A
\right)
\lesssim
2\exp\!\left(
-\frac{D A^2}
{8\|e\|_2^2\|\bar\rho\|_2^2}
\right).
$$

维度越高，随机方向上的误差通常越集中；但这是一条依赖分布假设的概率界，不是对任意
固定数据都成立的确定性界。

## 3. 低秩矩阵加速计算

### 3.1 8-bit LUT 为什么适合低秩分解

8-bit DMQ 的每个维度有 256 个代表值。对中心化查询分量 $\bar\rho_d$，
完整查询表为

$$
L_d(k)=\bar\rho_dc_{d,k},
\qquad k=0,1,\ldots,255.
$$

将 code 拆成高、低两个 4-bit 整数：

$$
k=16h+\ell,
\qquad h,\ell\in\{0,1,\ldots,15\}.
$$

再把第 $d$ 维的 256 项 LUT 重排成矩阵

$$
M_d(h,\ell)=L_d(16h+\ell)
=\bar\rho_dc_{d,16h+\ell},
\qquad M_d\in\mathbb R^{16\times16}.
$$

若直接计算，查询需要生成 $256D$ 个表项。低秩加速的目标是用少量只依赖 $h$ 或
$\ell$ 的小表近似 $M_d(h,\ell)$，从而把一次 256 项查表改写为若干次 16 项查表。

更一般地，对 $b$ bit code 选择

$$
b=b_H+b_L,
$$

并写成

$$
k=2^{b_L}h+\ell.
$$

两个小表的总大小为

$$
2^{b_H}+2^{b_L}.
$$

当 $b_H$ 与 $b_L$ 尽量接近时该值最小。因此 8 bit 使用 $4+4$ 拆分，
把每维 256 个表项降为 $16+16=32$ 个表项。

### 3.2 加性低秩近似

`DMQ-main` 采用的基本近似是

$$
\boxed{
M_d(h,\ell)
\approx A_d(h)+B_d(\ell)
}.
$$

定义行均值、列均值与全局均值

$$
r_d(h)=\frac{1}{16}\sum_{\ell=0}^{15}M_d(h,\ell),
$$

$$
c_d(\ell)=\frac{1}{16}\sum_{h=0}^{15}M_d(h,\ell),
$$

$$
g_d=\frac{1}{256}
\sum_{h=0}^{15}\sum_{\ell=0}^{15}M_d(h,\ell).
$$

取

$$
A_d(h)=r_d(h),
\qquad
B_d(\ell)=c_d(\ell)-g_d,
$$

便得到

$$
\boxed{
\widehat M_d(h,\ell)
=r_d(h)+c_d(\ell)-g_d
}.
$$

该解等价于求解带约束的最小二乘问题

$$
\min_{A,B}
\sum_{h=0}^{15}\sum_{\ell=0}^{15}
\left(M_d(h,\ell)-A(h)-B(\ell)\right)^2,
$$

$$
\text{s.t.}\qquad \sum_{\ell=0}^{15}B(\ell)=0.
$$

因此它是 $M_d$ 在“行主效应 + 列主效应”子空间上的正交投影。
残差矩阵

$$
E_d(h,\ell)
=M_d(h,\ell)-r_d(h)-c_d(\ell)+g_d
$$

满足

$$
\sum_{\ell=0}^{15}E_d(h,\ell)=0,
\qquad
\sum_{h=0}^{15}E_d(h,\ell)=0.
$$

`DMQ-main` 将这种形式称为 rank-1 加性分解。严格按线性代数定义，
$A\mathbf 1^{\mathsf T}+\mathbf 1B^{\mathsf T}$ 的矩阵秩最多为 2；
这里的“rank-1”表示只保留一层高、低 nibble 主效应，而不是标准 SVD 的秩 1。

### 3.3 将分解移到码本上预计算

因为

$$
M_d(h,\ell)=\bar\rho_dC_d(h,\ell),
\qquad
C_d(h,\ell)=c_{d,16h+\ell},
$$

行、列和全局均值都能提出查询标量 $\bar\rho_d$。定义仅依赖码本的量

$$
H_d(h)=\frac{1}{16}\sum_{\ell=0}^{15}C_d(h,\ell),
$$

$$
G_d=\frac{1}{256}
\sum_{h=0}^{15}\sum_{\ell=0}^{15}C_d(h,\ell),
$$

$$
T_d(\ell)=
\frac{1}{16}\sum_{h=0}^{15}C_d(h,\ell)-G_d.
$$

则查询表可以直接写成

$$
L_d^{H}(h)=\bar\rho_dH_d(h),
\qquad
L_d^{L}(\ell)=\bar\rho_dT_d(\ell).
$$

对数据向量第 $d$ 维的 code $(h_d,\ell_d)$，有

$$
L_d(k_d)
\approx L_d^{H}(h_d)+L_d^{L}(\ell_d).
$$

因此 DMQ 内积项变为

$$
\boxed{
\langle u,\bar\rho\rangle
\approx
\sum_{d=1}^{D}
\left(
L_d^{H}(h_d)+L_d^{L}(\ell_d)
\right)
}.
$$

$H_d$ 和 $T_d$ 与查询无关，可以在码本训练完成后预计算。
查询侧的乘法量由 $256D$ 降为 $32D$，理论上减少到原来的 $1/8$。

这种近似对 DMQ 码本通常有效，是因为 $c_{d,k}$ 随有序 code $k$ 单调变化。
若代表值恰好满足仿射关系

$$
c_{d,k}=a_dk+b_d,
$$

则

$$
c_{d,16h+\ell}
=16a_dh+a_d\ell+b_d,
$$

高、低位完全可加，分解误差为零。实际非均匀分位码本不是严格线性的，
但若量化曲线足够平滑，其高低位交互项通常较小。

### 3.4 低秩得分误差

在码本空间定义加性残差

$$
\varepsilon_d(h,\ell)
=C_d(h,\ell)-H_d(h)-T_d(\ell).
$$

对给定数据 code，低秩内积误差为

$$
\Delta_{\mathrm{LR}}
=\sum_{d=1}^{D}
\bar\rho_d\varepsilon_d(h_d,\ell_d).
$$

令

$$
\varepsilon_x=
\left(
\varepsilon_1(h_1,\ell_1),
\ldots,
\varepsilon_D(h_D,\ell_D)
\right),
$$

则有确定性界

$$
\boxed{
|\Delta_{\mathrm{LR}}|
\le\|\bar\rho\|_2\|\varepsilon_x\|_2
}.
$$

矩阵层面可用相对 Frobenius 误差衡量平均近似质量：

$$
\eta_d=
\frac{\|M_d-\widehat M_d\|_F}
{\|M_d\|_F}.
$$

不过 $\eta_d$ 对 256 个 code 一视同仁，而真实索引中的 code 分布通常不均匀。
更贴近检索误差的训练目标应按 code 出现概率 $p_d(h,\ell)$ 加权：

$$
\mathcal L_{\mathrm{LR}}
=\sum_d\sum_{h,\ell}
p_d(h,\ell)
\varepsilon_d(h,\ell)^2.
$$

### 3.5 与 DMQ 量化误差的合成

低秩近似不会改变 8-bit code，只是把代表值 $u$ 进一步近似成
$\widetilde u$。原始 DMQ 误差为

$$
e_{\mathrm{DMQ}}=\bar r-\gamma_xu.
$$

加入低秩分解后的总误差为

$$
\begin{aligned}
e_{\mathrm{total}}
&=\bar r-\gamma_x\widetilde u\\
&=e_{\mathrm{DMQ}}
+\gamma_x(u-\widetilde u).
\end{aligned}
$$

相应的查询内积误差分解为

$$
\boxed{
\langle e_{\mathrm{total}},\bar\rho\rangle
=\langle e_{\mathrm{DMQ}},\bar\rho\rangle
+\gamma_x
\langle u-\widetilde u,\bar\rho\rangle
}.
$$

第一项来自 DMQ value 量化，第二项来自低秩 LUT 近似。
两项可能同向叠加，也可能相互抵消，因此不能只凭码本重建误差判断最终排序质量。

### 3.6 从加性分解推广到真正的 rank-$R$

加性模型无法表达高位 $h$ 与低位 $\ell$ 的交互。可先去掉行列主效应，
再对残差矩阵 $E_d$ 做奇异值分解：

$$
E_d=P_d\Sigma_dQ_d^{\mathsf T}.
$$

保留前 $R$ 个奇异分量：

$$
E_d(h,\ell)
\approx
\sum_{s=1}^{R}
\sigma_{d,s}p_{d,s}(h)q_{d,s}(\ell).
$$

最终得到

$$
\boxed{
C_d(h,\ell)
\approx
H_d(h)+T_d(\ell)
+\sum_{s=1}^{R}
\sigma_{d,s}p_{d,s}(h)q_{d,s}(\ell)
}.
$$

根据 Eckart--Young 定理，这是给定 $R$ 时对交互残差的最优 Frobenius 范数近似，且

$$
\left\|
E_d-
\sum_{s=1}^{R}
\sigma_{d,s}p_{d,s}q_{d,s}^{\mathsf T}
\right\|_F^2
=\sum_{s>R}\sigma_{d,s}^2.
$$

rank-$R$ 越大，近似误差越小，但每个维度需要更多小表查询和乘加。
加性分解只需两次查表与一次加法；每增加一个 SVD 分量，通常还需读取
$p_{d,s}(h)$、$q_{d,s}(\ell)$ 并计算一次乘积。因此 $R$ 是吞吐与精度的显式旋钮。

### 3.7 两阶段精确校正

若低秩点估计仅用于候选生成，可以采用

$$
\text{低秩快速打分所有候选}
\longrightarrow
\text{完整 256 项 LUT 重算 top-}M
$$

的两阶段策略。这样多数候选只支付低秩代价，最终候选使用精确 DMQ8 LUT。

但 top-$M$ 精确校正本身不能保证零召回损失：若真实近邻已在低秩阶段被排除，
后续无法恢复。若要求可证明的安全剪枝，可为每条向量保存
$\|u-\widetilde u\|_2$，并使用

$$
B_x^{\mathrm{LR}}(q)
=|\gamma_x|\,\|\bar\rho\|_2
\|u-\widetilde u\|_2
$$

构造低秩得分区间；否则 $M$ 应通过召回率与吞吐实验选择。

### 3.8 在 SINDI 稀疏 DMQ 中的对应形式

SINDI 的每个 term $t$ 也有 256 个代表值 $c_{t,k}$，因此可写成

$$
C_t(h,\ell)=c_{t,16h+\ell}
\approx H_t(h)+T_t(\ell).
$$

对文档非零项的 code $k_{x,t}=16h_{x,t}+\ell_{x,t}$，第 4.4 节的
$U_x(q)$ 可近似为

$$
\widetilde U_x(q)=
\sum_{t\in H(q,x)}q_t
\left(
H_t(h_{x,t})+T_t(\ell_{x,t})
\right).
$$

于是低秩 SINDI DMQ8 得分为

$$
\boxed{
\widehat{\operatorname{IP}}_{8,\mathrm{LR}}(q,x)
=\mu_xS_x(q)+\gamma_x\widetilde U_x(q)
}.
$$

它把每个 term 的 256 项查询 LUT 降为两个 16 项 LUT。
这类分解只会在批量、小表查找能够替代 256 项随机查找时形成计算优势；
若搜索本来就是逐 posting 直接读取一个浮点代表值，低秩形式未必更快。
DMQ1bit 每个 term 只有两个候选值，不需要低秩分解。

### 3.9 复杂度总结

设参与计算的维度数为 $D_s$。忽略常数与 LUT 的整数化误差：

| 方法 | 每维模型 | 查询表规模 | 每维得分操作 | 额外近似 |
| --- | ---: | ---: | --- | --- |
| 完整 DMQ8 | 256 个值 | $256D_s$ | 1 次 256 项查表 | 无 |
| 高低位加性 | $16+16$ 个值 | $32D_s$ | 2 次 16 项查表 + 加法 | 交互残差 |
| 加性 + rank-$R$ | $32+32R$ 个值 | 依实现而定 | 小表查找 + $R$ 次乘加 | 尾部奇异值 |
| 低秩 + top-$M$ 校正 | 同上 + 完整码本 | 两阶段 | 全量近似、少量精确 | 取决于候选覆盖 |

低秩分解压缩的是“查询计算所需的 LUT”，不是数据向量的 8-bit code。
它与第 2 节的 DMQ 量化正交：前者减少打分计算，后者减少向量 value 存储。

## 4. SINDI 的稀疏 DMQ

### 4.1 稀疏向量不能直接照搬稠密中心化

设词表大小为 $V$，稀疏文档向量为 $x\in\mathbb{R}^V$，非零支持集为

$$
S_x=\operatorname{supp}(x),
\qquad L_x=|S_x|.
$$

若直接在全部 $V$ 维减均值，会把原本为零的坐标变成非零值，破坏稀疏性。SINDI 只在
$S_x$ 上计算均值：

$$
\mu_x=\frac{1}{L_x}\sum_{t\in S_x}x_t.
$$

令 $s_x\in\{0,1\}^V$ 为支持集指示向量，并定义稀疏残差

$$
r_{x,t}=
\begin{cases}
x_t-\mu_x,&t\in S_x,\\
0,&t\notin S_x.
\end{cases}
$$

则

$$
\boxed{x=\mu_xs_x+r_x},
\qquad
\sum_{t\in S_x}r_{x,t}=0.
$$

这里 $\mu_xs_x$ 仍然只在原支持集上非零，所以不会稠密化。

### 4.2 从“每个坐标”变成“每个 term”

稠密 DMQ 在某个聚类的每个坐标上训练分布；SINDI 没有固定长度的连续坐标扫描，而是把
term $t$ 视为一个稀疏坐标。它为 term $t$ 收集

$$
R_t=\{r_{x,t}:t\in S_x\}.
$$

随后完全复用第 2.2 节的加权分位点规则，为每个 term 训练

$$
C_t=\{c_{t,0},c_{t,1},\ldots,c_{t,255}\}
$$

以及 255 个阈值。对非零项 $(t,x_t)$：

$$
k_{x,t}
=\sum_{m=0}^{254}
\mathbf 1[r_{x,t}>\tau_{t,m}],
$$

$$
u_{x,t}=c_{t,k_{x,t}}.
$$

当前 SINDI 主路径使用 8 bit，因此每个非零 value 只需保存 $k_{x,t}$ 的一个字节。

### 4.3 SINDI DMQ8 的尺度校正

对文档 $x$ 的全部已保留非零项，定义

$$
\gamma_x=
\frac{\sum_{t\in S_x}r_{x,t}^2}
{\sum_{t\in S_x}u_{x,t}r_{x,t}}.
$$

于是每个非零 value 的重建值为

$$
\boxed{
\hat x_t=\mu_x+\gamma_xu_{x,t},
\qquad t\in S_x
}.
$$

分布码本负责描述 term 条件分布，$\mu_x$ 恢复文档自身的偏移，$\gamma_x$ 恢复文档
残差的尺度。

### 4.4 稀疏内积估计

设查询支持集为 $S_q$，查询与文档的交集为

$$
H(q,x)=S_q\cap S_x.
$$

精确内积只在交集上求和：

$$
\begin{aligned}
\langle q,x\rangle
&=\sum_{t\in H(q,x)}q_tx_t\\
&=\mu_x\sum_{t\in H(q,x)}q_t
+\sum_{t\in H(q,x)}q_tr_{x,t}.
\end{aligned}
$$

定义两个查询相关量

$$
S_x(q)=\sum_{t\in H(q,x)}q_t,
$$

$$
U_x(q)=\sum_{t\in H(q,x)}q_tu_{x,t}.
$$

SINDI DMQ8 的近似内积为

$$
\boxed{
\widehat{\operatorname{IP}}_8(q,x)
=\mu_xS_x(q)+\gamma_xU_x(q)
}.
$$

等价地，可为每个查询 term 建立

$$
L_{t,k}=q_tc_{t,k},
$$

再通过 code $k_{x,t}$ 取表并累加。

SINDI 对外使用的距离为

$$
d(q,x)=1-\operatorname{IP}(q,x).
$$

候选阶段内部也可使用 $-\operatorname{IP}$；两者只差常数 1，不改变排序。

### 4.5 为什么稀疏 DMQ 需要额外的交集和

稠密向量中，均值项为

$$
a_x\sum_{j=1}^{D}q_j,
$$

其中 $\sum_jq_j$ 对所有数据向量相同，可以每个查询只计算一次。

稀疏向量中，均值项变成

$$
\mu_xS_x(q)
=\mu_x\sum_{t\in S_q\cap S_x}q_t.
$$

交集依赖具体文档 $x$，因此 $S_x(q)$ 必须随候选分别累加。这个额外量不是实现偶然产生
的开销，而是“只在非零支持集内中心化”后的必然数学结果。

### 4.6 DMQ8 的两种用途

SINDI 中 8-bit 稀疏 DMQ 可用于两个阶段，但二者的估计式相同：

- 倒排候选生成：沿 query term 的倒排表累加 $S_x(q)$ 和 $U_x(q)$；
- 正排候选重排：对少量候选的 term 序列与 query 做稀疏交集，再计算同一公式。

区别只在于遍历方向和参与计算的候选数量，不在量化模型本身。DMQ8 给出的是点估计；若
候选阶段不再重排，其量化误差会直接影响最终次序。

### 4.7 SINDI DMQ1bit 粗筛

1-bit 路径只保存残差的符号。当前 SINDI 从全部文档的中心化非零残差中训练一对全局
代表值：

$$
a=\mathbb E[r_{x,t}\mid r_{x,t}\le0],
\qquad
b=\mathbb E[r_{x,t}\mid r_{x,t}>0].
$$

对每个非零项定义

$$
u^{(1)}_{x,t}=
\begin{cases}
a,&r_{x,t}\le0,\\
b,&r_{x,t}>0.
\end{cases}
$$

令

$$
\gamma_x^{(1)}=
\frac{\|r_x\|_2^2}
{\langle u_x^{(1)},r_x\rangle}.
$$

当前数学记账方式保存

$$
\alpha_x^{(1)}=-2\gamma_x^{(1)},
$$

所以

$$
r_x\approx
-\frac{1}{2}\alpha_x^{(1)}u_x^{(1)}.
$$

1-bit 内积估计为

$$
\boxed{
\widehat{\operatorname{IP}}_1(q,x)
=\mu_xS_x(q)
-\frac{1}{2}\alpha_x^{(1)}
\sum_{t\in H(q,x)}q_tu^{(1)}_{x,t}
}.
$$

因为通常 $\langle u_x^{(1)},r_x\rangle>0$，所以
$\alpha_x^{(1)}<0$，公式中的第二项实际恢复正尺度
$\gamma_x^{(1)}\langle q,u_x^{(1)}\rangle$。

### 4.8 1-bit 误差带与候选剪枝

定义 1-bit 残差误差

$$
w_x=r_x-\gamma_x^{(1)}u_x^{(1)}
=r_x+\frac{1}{2}\alpha_x^{(1)}u_x^{(1)}.
$$

精确 IP 与 1-bit 估计的差为

$$
\operatorname{IP}(q,x)-
\widehat{\operatorname{IP}}_1(q,x)
=\langle q_{H(q,x)},w_{x,H(q,x)}\rangle.
$$

无条件成立的确定性界是

$$
\left|
\operatorname{IP}-\widehat{\operatorname{IP}}_1
\right|
\le
\|q_{H(q,x)}\|_2\|w_{x,H(q,x)}\|_2
\le\|q\|_2\|w_x\|_2.
$$

SINDI 采用更紧的高维集中误差带

$$
\boxed{
E_x(q)=
c_\varepsilon
\frac{\|w_x\|_2\|q\|_2}
{\sqrt{\max(L_x-1,1)}}
}.
$$

若候选阶段使用内部距离

$$
\hat d_x=-\widehat{\operatorname{IP}}_1(q,x),
$$

则距离下界取为

$$
d_x^{\mathrm{lower}}=\hat d_x-E_x(q).
$$

当该下界已经不可能优于当前 top-$k$ 阈值时，可以跳过后续 DMQ8 重排。

需要特别指出：式中的 $1/\sqrt{L_x-1}$ 来自高维集中假设。使用完整 $\|q\|_2$ 代替
交集范数使界更保守，但不能把概率界自动变成确定性界。因此，$c_\varepsilon$ 控制的是
召回风险与剪枝强度之间的权衡；若需要对任意数据成立的严格界，应使用不含该分母的
Cauchy--Schwarz 界。

## 5. 压缩率与模型开销

以下只计算 value code 和 DMQ 标量，不包含文档 ID、term ID、倒排表指针、对齐和容器
开销。

### 5.1 DMQ8

一条有 $L_x$ 个非零项的 FP32 稀疏向量，其 value 部分需要

$$
B_{\mathrm{FP32}}=32L_x\quad\text{bits}.
$$

DMQ8 为每个非零项保存 8-bit code，并为每条向量保存两个 FP32 标量
$(\mu_x,\gamma_x)$：

$$
B_{\mathrm{DMQ8}}=8L_x+64\quad\text{bits}.
$$

忽略码本后，相对比例为

$$
\frac{B_{\mathrm{DMQ8}}}{B_{\mathrm{FP32}}}
=\frac14+\frac{2}{L_x}.
$$

每个 8-bit term 码本包含 256 个代表值和 255 个阈值，共 511 个 FP32：

$$
B_{\mathrm{codebook/term}}
=511\times32\quad\text{bits}
=2044\quad\text{bytes}.
$$

因此，长尾 term 很多时，码本开销可能超过 value code 的节省；这是“每 term 自适应”
精度与模型内存之间的主要矛盾。

### 5.2 DMQ1bit

1-bit 粗筛为每个非零项保存 1 bit，并为每条向量保存

$$
\left(
\mu_x,\alpha_x^{(1)},\|w_x\|_2,
1/\sqrt{\max(L_x-1,1)}
\right)
$$

四个 FP32 标量：

$$
B_{\mathrm{DMQ1}}=L_x+128\quad\text{bits}.
$$

忽略全局 $(a,b)$ 后，相对比例为

$$
\frac{B_{\mathrm{DMQ1}}}{B_{\mathrm{FP32}}}
=\frac1{32}+\frac{4}{L_x}.
$$

DMQ1bit 在 SINDI 中是粗筛表示，通常还要与最终的 DMQ8 重排表示共同存在，不能把上述
比例直接当成整个索引的总压缩率。

若正排表示还需保存 term ID，设有效 term ID 范围为 $[0,U)$，则每个 ID 至少需要

$$
B_{\mathrm{id}}=\lceil\log_2U\rceil
$$

bit；这一部分与 value 的 DMQ 量化相互独立。

## 6. 稠密 DMQ 与 SINDI Sparse DMQ 的对应关系

| 数学角色 | `DMQ-main` 稠密向量 | SINDI 稀疏向量 |
| --- | --- | --- |
| 基准 | IVF 中心 $c$ | 支持集外隐式为 0 |
| 中心化范围 | 全部 $D$ 维 | 文档非零支持集 $S_x$ |
| 分布作用域 | 聚类内的每个坐标 | 每个 term |
| 量化对象 | $x-c-a_x\mathbf 1$ | $x_t-\mu_x$ |
| 逐向量尺度 | $\gamma_x$ | $\gamma_x$ 或 $\gamma_x^{(1)}$ |
| 查询核心量 | $\langle u,\bar\rho\rangle$ | $S_x(q)$ 与 $U_x(q)$ |
| 目标 | 主要估计 L2/稠密相似度 | 稀疏最大内积搜索 |
| 低 bit 用法 | 分段或逐层过滤 | 1-bit 粗筛后 8-bit 重排 |

两者真正共享的不是存储布局，而是同一个估计框架：

$$
\boxed{
\text{原向量}
=\text{均值部分}+\text{中心化残差}
\approx\text{均值部分}+\text{逐向量尺度}\times\text{分布码字}
}.
$$

稠密 DMQ 的难点是高维距离的快速查表；SINDI Sparse DMQ 的难点则是支持集随文档变化，
必须同时恢复“交集上的 query 和”与“交集上的量化残差内积”。

## 7. 适用边界

- DMQ 的收益来自训练分布与检索分布相对稳定；分布漂移会增加量化误差。
- 每 term 码本适合高频 term，长尾词表需要额外权衡码本共享、回退码本或训练样本不足。
- 文档剪枝会改变实际支持集。上述公式在固定支持集上成立，结构剪枝造成的误差应与
  DMQ value 量化误差分开分析。
- DMQ8 是近似点估计；DMQ1bit 的紧误差带依赖高维集中假设。
- 低秩 LUT 会在 DMQ 量化误差之外引入一层交互残差；若不构造误差界，
  它应被视为近似候选分数，而不是安全剪枝条件。
- 当 $L_x$ 很小时，逐向量标量的固定成本和集中效应都会变差，DMQ 的优势会减弱。

## 8. 总结

`DMQ-main` 先对 IVF 残差做自均值中心化，再按坐标的加权经验分布训练多 bit 非均匀
码本，并用逐向量尺度 $\gamma$ 校准内积，最终通过查询 LUT 估计 L2 距离。分段和
“1 bit + 残差 bit”只是这一基本估计器的空间或阶段化展开。

对 8-bit DMQ，还可以把 256 项查询 LUT 重排为 $16\times16$ 矩阵，
用高、低 nibble 的行列主效应近似，将查询表规模从每维 256 项降至 32 项。
若交互残差仍然过大，可继续用 rank-$R$ SVD 补偿，或只对 top-$M$ 候选恢复完整 LUT。

SINDI 将同一思想改写到稀疏支持集上：均值只在文档非零项内计算，码本按 term 训练，
查询只在 $S_q\cap S_x$ 上累加。DMQ8 用

$$
\widehat{\operatorname{IP}}_8
=\mu_xS_x(q)+\gamma_xU_x(q)
$$

作为候选分数或重排分数；DMQ1bit 则用一个符号 bit、逐向量误差范数和概率误差带先做
低成本筛选。前者提供较高精度的 value 压缩，后者提供更激进的候选过滤，二者共同构成
SINDI 稀疏 DMQ 的多阶段压缩检索思路。
