
# 第四章习题解答

## 习题 4.1 (★☆☆) 多项式模型的最小二乘解
考虑多项式模型
$$
y(x, \boldsymbol{w}) = w_{0} + w_{1}x + w_{2}x^{2} + \cdots + w_{M}x^{M}
$$
并使用平方损失函数
$$
E(\boldsymbol{w}) = \frac{1}{2} \sum\limits_{n=1}^{N} \Big[ y(x_{n}, \boldsymbol{w}) - t_{n} \Big]^{2} 
$$
我们需要求使得平方损失函数最小的 $\boldsymbol{w}$。

为了简便，我们首先将 $y(x, \boldsymbol{w})$ 写成 $y(\boldsymbol{x}, \boldsymbol{w}) = \boldsymbol{w}^{\top}\boldsymbol{x}$，其中 $\boldsymbol{w} = [w_{0}, w_{1}, \dots, w_{M}]^{\top}$，$\boldsymbol{x} = [1, x, x^{2}, \dots, x^{M}]$。于是平方损失函数就可以写为 
$$E(\boldsymbol{w}) = \frac{1}{2}\sum\limits_{n=1}^{N} \Big[ \boldsymbol{w}^{\top}\boldsymbol{x} - t_{n} \Big]^{2} = \frac{1}{2}\| \boldsymbol{X}\boldsymbol{w} - \boldsymbol{t} \|_{2}^{2} \tag{1}$$
这里需要稍稍解释一下这个变来变去的notation。首先 $\|\cdot\|_{2}$ 是 Euclid 范数，对于向量 $\boldsymbol{x} = [x_{1}, x_{2}, \dots, x_{n}]$，我们有 $\displaystyle \|\boldsymbol{x}\|_{2}^{2} = \sum\limits_{i=1}^{n} x_{i}^{2}$。其次我们将所有的 $\boldsymbol{x}_{i}$ 转置，然后纵向堆叠，就得到
$$
\boldsymbol{X} = \begin{bmatrix}
\boldsymbol{x}_{1}^{\top}\\
\boldsymbol{x}_{2}^{\top}\\
\boldsymbol{x}_{3}^{\top}\\
\vdots \\
\boldsymbol{x}_{N}^{\top}
\end{bmatrix} = \begin{bmatrix}
1 & x_{1} & x_{1}^{2} & \cdots  & x_{1}^{M}\\
1 & x_{2} & x_{2}^{2} & \cdots  & x_{2}^{M}\\
1 & x_{3} & x_{3}^{2} & \cdots  & x_{3}^{M}\\
\vdots & \vdots  & \vdots & \ddots & \vdots \\
1 & x_{N} & x_{N}^{2} & \cdots  & x_{N}^{M}\\
\end{bmatrix} \in \mathbb{R}^{N \times M},
$$
同样地，我们把所有标签堆叠成一个向量 $\boldsymbol{t} = [t_{1}, t_{2}, \dots, t_{N}]^{\top}$，可以验证我们上面的改写是正确的。我们这样改写的目的是方便矩阵求导，这样形式更加简洁：
$$
\displaystyle \frac{ \partial E(\boldsymbol{w}) }{ \partial \boldsymbol{w} } = \boldsymbol{X}^{\top}(\boldsymbol{X}\boldsymbol{w} - \boldsymbol{t}).
$$
这里有个技巧。**面对简单形式的矩阵微分的时候我们可以将其看作是标量先求导，然后我们调整矩阵的相乘顺序和转置，使得其形状与所对求导的参数的形状相同。当式子中遇到方阵时，不妨将其看做非方阵。** 这样的话我们就能很轻易的得到最优的 $\boldsymbol{w}$ 满足的条件：

$$
\boldsymbol{X}^{\top}\boldsymbol{X}\boldsymbol{w} = \boldsymbol{X}^{\top}\boldsymbol{t} \iff \sum\limits_{n=1}^{N} \boldsymbol{x}_{n}\boldsymbol{x}_{n}^{\top}\boldsymbol{w} = \sum\limits_{j=1}^{n} \boldsymbol{x}_{n}t_{j}
$$
注意等式两边都是向量，形状是 $M \times 1$。这相当于两个向量对应分量相同，也就是
$$
\sum\limits_{n=1}^{N} x_{n}^{i}\left[  \sum\limits_{j=1}^{M} x^{j}_{n} w_{j}  \right] = \sum\limits_{j=1}^{M} \left\{\left[ {\color{red} \sum\limits_{n=1}^{N} x_{n}^{i+j} } \right] w_{j}\right\} = {\color{blue} \sum\limits_{n=1}^{N} x_{n}^{i}t_{n} } 
$$
其中红色的项就是 $A_{i,j}$，蓝色的项就是 $T_{i}$.

## 习题 4.2 (★☆☆) 带正则项的多项式模型的最小二乘解

我们复用 (1)。加上正则项后我们就有
$$
\tilde{E}(\boldsymbol{w}) = \frac{1}{2}\| \boldsymbol{X}\boldsymbol{w} - \boldsymbol{t} \|_{2}^{2} + \frac{\lambda}{2}\|\boldsymbol{w}\|_{2}^{2}.
$$
然后对 $\boldsymbol{w}$ 求偏导，有
$$
\displaystyle \frac{ \partial E(\boldsymbol{w}) }{ \partial \boldsymbol{w} } = \boldsymbol{X}^{\top}(\boldsymbol{X}\boldsymbol{w} - \boldsymbol{t}) + \lambda \boldsymbol{w}
$$
令其为零，整理一下有
$$
(\boldsymbol{X}^{\top}\boldsymbol{X} + \lambda I)\boldsymbol{w} = \boldsymbol{X}^{\top}\boldsymbol{t}.
$$
剩下的步骤和上面一样，我就不写了。

## 习题 4.3 (★☆☆) 双曲正切基函数和 Sigmoid 基函数等价

首先写出双曲正切函数和 Sigmoid 函数
$$
\tanh(a) = \frac{e^{a} - e^{-a}}{e^{a} + e^{-a}}, \quad \sigma(a) = \frac{1}{1 + e^{-a}}
$$
我们不难发现
$$
\begin{align*}
2\sigma(2a) - 1 &= \frac{2}{1 + e^{-2a}} - 1\\
&= \frac{2e^{a}}{e^{a} + e^{-a}} - 1 & 两边同时乘~e^{a}\\
&= \tanh(a).
\end{align*}
$$
所以我们有 $\displaystyle \sigma(a) = \frac{1}{2}\left( \tanh\left( \frac{a}{2} \right) + 1\right)$。现在考虑使用 Sigmoid 基函数的模型
$$
y(x, \boldsymbol{w}) = w_{0} + \sum\limits_{j=1}^{M} w_{j} \sigma\left( \frac{x - \mu_{j}}{s} \right)
$$
我们做这个代换，就有
$$
\begin{align*}
y(x, \boldsymbol{w}) &= w_{0} + \sum\limits_{j=1}^{M} w_{j}\cdot  \frac{1}{2} \left[ \tanh\left( \frac{x - \mu_{j}}{2s} \right) + 1\right]\\
&= {\color{red} w_{0} + \frac{1}{2}\sum\limits_{i=1}^{M} w_{i} }  + \sum\limits_{j=1}^{M} {\color{blue} \frac{1}{2}w_{j}  } \cdot \tanh\left( \frac{x - \mu_{j}}{2s} \right)\\
&= {\color{red} u_{0} }  + \sum\limits_{j=1}^{M} {\color{blue} u_{j} }  \tanh\left( \frac{x - \mu_{j}}{2s} \right).
\end{align*}
$$

## 习题 4.4  (★★★) $\Phi(\Phi ^{\top}\Phi)^{-1}\Phi ^{\top}$ 是投影算子

这个事实在前文中提到过。我们可以验证它是一个正交投影矩阵（算子），假如记该矩阵为 $P$，我们可以证明 $P^{2} =P$，以及 $P^{\top} = P$。我们也可以考虑 $(1 - P)\boldsymbol{x}$，对任意的 $\boldsymbol{x}$，可以证明它总是在 $\text{Col}(\Phi)^{\perp}$ 中，根据正交分解的唯一性，可以得到 $P$ 是投影算子，将任意向量投影到 $\Phi$ 的列空间中。

## 习题 4.5  (★☆☆) 加权平方损失

对于加权平方损失我们可以采用下面的写法，首先定义方阵 $R = \text{diag}\{ r_{1}, r_{2}, \dots, r_{N} \}$，显然 $R$ 是对称且正定的。因此我们可以将原来的加权损失写成加权范数的形式，也就是
$$
E_{D}(\boldsymbol{w}) = \frac{1}{2}\|\boldsymbol{t} - \Phi \boldsymbol{w}\|_{R}^{2} = \frac{1}{2}\|R^{-\frac{1}{2}}(\boldsymbol{t} - \Phi \boldsymbol{w})\|_{2}^{2}
$$
还是对 $\boldsymbol{w}$ 求偏导，我们得到
$$
\begin{align*}
\displaystyle \frac{ \partial E_{D}(\boldsymbol{w}) }{ \partial \boldsymbol{w} } &= \displaystyle \frac{ \partial E_{D}(\boldsymbol{w}) }{ \partial R^{-\frac{1}{2}}(\boldsymbol{t} - \Phi \boldsymbol{w}) } \displaystyle \frac{ \partial R^{-\frac{1}{2}}(\boldsymbol{t} - \Phi \boldsymbol{w}) }{ \partial (\boldsymbol{t} - \Phi \boldsymbol{w}) } \displaystyle \frac{ \partial (\boldsymbol{t} - \Phi \boldsymbol{w}) }{ \partial \boldsymbol{w} } \\
&= -\Phi ^{\top}R^{-\frac{\top}{2}}R^{-\frac{1}{2}}(\boldsymbol{t} - \Phi \boldsymbol{w})\\
&= -\Phi ^{\top}R^{-1}(\boldsymbol{t} - \Phi \boldsymbol{w})
\end{align*}
$$
令其为零，我们有
$$
\Phi ^{\top}R^{-1}\Phi \boldsymbol{w} = \Phi ^{\top}R^{-1}\boldsymbol{t} \iff \boldsymbol{w} = (\Phi ^{\top}R^{-1}\Phi)^{-1}\Phi ^{\top}R^{-1}\boldsymbol{t}.
$$
这样的方法可以对离群值、重复数据赋予低权重，以提高模型的鲁棒性，降低过拟合。

## 习题 4.6  (★☆☆) 岭回归的解析解

其形式和 4.2 中得出的结果一样，这里不再赘述。

## 习题 4.7  (★★☆) 向量值目标线性回归之参数的极大似然估计

现在我们考虑标签值不是标量的情况，并假设服从Gauss分布
$$
p(\boldsymbol{t}|\boldsymbol{W}, \boldsymbol{\Sigma}) = \mathcal{N}(\boldsymbol{t}|\boldsymbol{y}(\boldsymbol{x}, \boldsymbol{W}), \Sigma) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{ \Sigma } |^{1/2}} \exp\left\{  -\frac{1}{2}(\boldsymbol{t} - \boldsymbol{W}^{\top}\boldsymbol{\phi}(\boldsymbol{x}))^{\top}\boldsymbol{ \Sigma }  ^{-1}(\boldsymbol{t} - \boldsymbol{W}^{\top}\boldsymbol{\phi}(\boldsymbol{x}))  \right\}
$$
首先我们来算对数似然函数
$$
\begin{align*}
\ln p(\boldsymbol{T}|\boldsymbol{X}, \boldsymbol{W}, \boldsymbol{ \Sigma } ) &= \sum\limits_{n=1}^{N} \ln \mathcal{N}(\boldsymbol{t}_{n}|\boldsymbol{y}(\boldsymbol{x}_{n}, \boldsymbol{W}), \Sigma)\\
&= \sum\limits_{n=1}^{N} \left[ -\frac{1}{2}\ln|\boldsymbol{\Sigma}| - \frac{n}{2}\ln(2\pi) - \frac{1}{2}(\boldsymbol{t}_{n} - \boldsymbol{W}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_{n}))^{\top}\boldsymbol{ \Sigma }  ^{-1}(\boldsymbol{t}_{n} - \boldsymbol{W}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_{n})) \right]\\
&= -\frac{N}{2} \ln|\boldsymbol{\Sigma}| - \frac{1}{2} \sum\limits_{n=1}^{N}(\boldsymbol{t}_{n} - \boldsymbol{W}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_{n}))^{\top}\boldsymbol{ \Sigma }  ^{-1}(\boldsymbol{t}_{n} - \boldsymbol{W}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_{n})) + C
\end{align*}
$$
我们先对 $\boldsymbol{W}$ 求偏导
$$
\begin{align*}
\displaystyle \frac{ \partial \ln p(\boldsymbol{T}|\boldsymbol{X}, \boldsymbol{W}, \boldsymbol{ \Sigma } ) }{ \partial \boldsymbol{W} } &= \left[ \sum\limits_{n=1}^{N} \underbrace{ \boldsymbol{\phi}(\boldsymbol{x}_{n}) }_{ M \times 1 } \big[\underbrace{ \boldsymbol{t}_{n} - \boldsymbol{W}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_{n})\big]^{\top} }_{ 1 \times T } \right] \underbrace{ \boldsymbol{\Sigma}^{-1} }_{ T \times T } \in \mathbb{R}^{M \times T}
\end{align*} 
$$
令其为零，因为 $\boldsymbol{\Sigma}$ 是对称正定矩阵，可以把它丢掉。有
$$
\sum\limits_{n=1}^{N} \boldsymbol{\phi}(\boldsymbol{x}_{n})\boldsymbol{t}_{n}^{\top} = \left[ \sum\limits_{n=1}^{N} \boldsymbol{\phi}(\boldsymbol{x}_{n})\boldsymbol{\phi}(\boldsymbol{x}_{n})^{\top} \right]\boldsymbol{W}_{\text{ML}} \iff \boldsymbol{\Phi}^{\top}\boldsymbol{T} = \boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}\boldsymbol{W}_{\text{ML}}
$$
其中 $\boldsymbol{T} = [\boldsymbol{t}_{1}, \boldsymbol{t}_{2}, \dots, \boldsymbol{t}_{N}]$。可见对于 $\boldsymbol{W}_{\text{ML}}$ 的每一列 $\boldsymbol{w}_{\text{ML},i}$，都满足 $\boldsymbol{\Phi}^{\top}\boldsymbol{t}_{i} = \boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}\boldsymbol{w}_{\text{ML},i}$。

接下来我们对协方差矩阵求偏导
$$
\begin{align*}
\displaystyle \frac{ \partial \ln p(\boldsymbol{T}|\boldsymbol{X}, \boldsymbol{W}_{\text{ML}}, \boldsymbol{ \Sigma } ) }{ \partial \boldsymbol{\Sigma} } &= - \frac{N}{2} \boldsymbol{\Sigma}^{-{\top}} - \frac{1}{2}\sum\limits_{n=1}^{N} (\boldsymbol{t}_{n} - \boldsymbol{W}_{\text{ML}}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_{n}))(\boldsymbol{t}_{n} - \boldsymbol{W}_{\text{ML}}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_{n}))^{\top},
\end{align*} 
$$
令其为零，我们立刻得到
$$
\boldsymbol{\Sigma}_{\text{ML}} = \frac{1}{N} \sum\limits_{n=1}^{N} (\boldsymbol{t}_{n} - \boldsymbol{W}_{\text{ML}}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_{n}))(\boldsymbol{t}_{n} - \boldsymbol{W}_{\text{ML}}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_{n}))^{\top}.
$$

本题中涉及较多的矩阵求导操作，读者可使用 https://www.matrixcalculus.org/ 快速获得求导的结果。

## 习题 4.8  (★☆☆) 用变分法求向量值决策平方损失的最小值

我们回忆一下之前介绍过的泛函导数。我们发现它符合下面的形式
$$
F[y] = \displaystyle \int G(y, x) \, \mathrm d{x}
$$
我们令 $\displaystyle G(\boldsymbol{ f }, \boldsymbol{ x }) = \int \|\boldsymbol{f} - \boldsymbol{t}\|_{2}^{2} \cdot p(\boldsymbol{x}, \boldsymbol{ t }) \, \mathrm d{\boldsymbol{t}}$。按照我们之前的结果，泛函 $F$ 关于 $\boldsymbol{f}$ 的导数是
$$
\frac{\delta \mathbb{E}[L(\boldsymbol{t}, \boldsymbol{f}(\boldsymbol{x}))]}{\delta \boldsymbol{f}(\boldsymbol{x})} = \displaystyle \frac{ \partial G }{ \partial \boldsymbol{f} } = \int (\boldsymbol{f} - \boldsymbol{t})p(\boldsymbol{x}, \boldsymbol{t}) \, \mathrm d{\boldsymbol{t}} 
$$
令其为零，得到
$$
\begin{align*}
& \int \boldsymbol{f}(\boldsymbol{x})p(\boldsymbol{x}, \boldsymbol{t}) \, \mathrm d\boldsymbol{t} = \int \boldsymbol{t}p(\boldsymbol{x}, \boldsymbol{t}) \, \mathrm d{\boldsymbol{t}}\\
\iff & \boldsymbol{f}(\boldsymbol{x}) p(\boldsymbol{x}) = p(\boldsymbol{x})\int \boldsymbol{t}p(\boldsymbol{t}|\boldsymbol{x}) \, \mathrm d{\boldsymbol{t}}\\
\iff & \boldsymbol{f}(\boldsymbol{x}) p(\boldsymbol{x}) = p(\boldsymbol{x}) \,\mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}]\\
\iff & \boldsymbol{f}(\boldsymbol{x}) = \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}]\\
\end{align*} 
$$

## 习题 4.9 & 4.10  (★☆☆) 向量值决策平方损失的偏差-方差分解

和我们在之前看到的技巧一样，我们在 Euclid 范数里面加一项减一项
$$
\begin{align*}
\mathbb{E}[L(\boldsymbol{t}, \boldsymbol{f}(\boldsymbol{x}))] &= \iint \|\boldsymbol{f}(\boldsymbol{x}) - \boldsymbol{t}\|_{2}^{2} p(\boldsymbol{x}, \boldsymbol{t}) \,\mathrm{d}\boldsymbol{x}\,\mathrm{d}\boldsymbol{t}\\
&= \iint \|\boldsymbol{f}(\boldsymbol{x}) - \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}] + \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}] - \boldsymbol{t}\|_{2}^{2} p(\boldsymbol{x}, \boldsymbol{t}) \,\mathrm{d}\boldsymbol{x}\,\mathrm{d}\boldsymbol{t}\\
&= \iint \Big[ \|\boldsymbol{f}(\boldsymbol{x}) - \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}]\|_{2}^{2} + \| \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}] - \boldsymbol{t} \|_{2}^{2} \\&\quad \quad \quad \quad + 2\big\langle \boldsymbol{f}(\boldsymbol{x}) - \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}], \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}] - \boldsymbol{t} \big\rangle  \Big]  p(\boldsymbol{x}, \boldsymbol{t}) \,\mathrm{d}\boldsymbol{x}\,\mathrm{d}\boldsymbol{t}\\
&= \iint \|\boldsymbol{f}(\boldsymbol{x}) - \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}]\|_{2}^{2}p(\boldsymbol{x}, \boldsymbol{t}) \,\mathrm{d}\boldsymbol{x}\,\mathrm{d}\boldsymbol{t} 
\\
&\quad \quad + \iint \| \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}] - \boldsymbol{t} \|_{2}^{2} p(\boldsymbol{x}, \boldsymbol{t}) \,\mathrm{d}\boldsymbol{x}\,\mathrm{d}\boldsymbol{t}
\\
&\quad \quad + 2\left\langle \boldsymbol{f}(\boldsymbol{x}) - \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}], \underbrace{ \iint (\mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}] - \boldsymbol{t})  p(\boldsymbol{x}, \boldsymbol{t}) \,\mathrm{d}\boldsymbol{x}\,\mathrm{d}\boldsymbol{t} }_{ 0 }\right\rangle & 内积的二重线性\\
&= \int \|\boldsymbol{f}(\boldsymbol{x}) - \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}]\|_{2}^{2}p(\boldsymbol{x}) \,\mathrm{d}\boldsymbol{x} 
\\
&\quad \quad + \int p(\boldsymbol{x}) \left[ \int \| \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}] - \boldsymbol{t} \|_{2}^{2} p(\boldsymbol{t}|\boldsymbol{x}) \,\mathrm{d}\boldsymbol{t}  \right]\,\mathrm{d}\boldsymbol{x}\\
&= \int \|\boldsymbol{f}(\boldsymbol{x}) - \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}]\|_{2}^{2}p(\boldsymbol{x}) \,\mathrm{d}\boldsymbol{x} + \int \text{var}[\boldsymbol{t}|\boldsymbol{x}] p(\boldsymbol{x}) \,\mathrm{d}\boldsymbol{x}
\end{align*}
$$
可以看到这个过程和我们之前的推导几乎一模一样。第二项和 $\boldsymbol{f}$ 的选择无关，因此很容易得到它的极小值点，也就是 $\boldsymbol{f}(\boldsymbol{x}) = \mathbb{E}_{\boldsymbol{t}}[\boldsymbol{t}|\boldsymbol{x}]$。

## 习题 4.11  (★★☆) 广义 Gauss 分布下回归问题的对数似然函数

我们首先证明它具有归一化性质。
$$
\begin{align*}
& \int_{-\infty}^{+\infty} \frac{q}{2(2\sigma^{2})^{1/q}\Gamma(1/q)} \exp\left\{  -\frac{|x|^{q}}{2\sigma^{2}}  \right\} \, \mathrm d{x} \\
=& \frac{q}{2(2\sigma^{2})^{1/q}\Gamma(1/q)}  \int_{-\infty}^{+\infty} \exp\left\{  -\frac{|x|^{q}}{2\sigma^{2}}  \right\} \, \mathrm d{x} \\
=& \frac{q}{(2\sigma^{2})^{1/q}\Gamma(1/q)}  \int_{\color{red} 0}^{+\infty} \exp\left\{  -\frac{x^{q}}{2\sigma^{2}}  \right\} \, \mathrm d{x} & 被积函数是偶函数
\end{align*}
$$
读者也许可以从前面的系数里面看出一些端倪。毕竟我们需要证明上面的式子等于 $1$。一个自然的想法是换元，然后能凑成 Gamma 函数的形式，从而把前面那堆系数消掉。我们令 $\displaystyle u = \frac{x^{q}}{2\sigma^{2}}$，就有
$$
x = [(2\sigma)^{2}]^{1/q}u^{1/q}
$$
于是上面的式子就变成
$$
\begin{align*}
& \frac{q}{(2\sigma^{2})^{1/q}\Gamma(1/q)}  \int_{0}^{+\infty} \exp\left\{  -u \right\} \, \mathrm d\{[(2\sigma)^{2}]^{1/q}u^{1/q}\}\\
=& \frac{\cancel{q}}{\cancel{(2\sigma^{2})^{1/q}}\Gamma(1/q)} \int_{0}^{+\infty} \cancel{[(2\sigma)^{2}]^{1/q}} \cdot\exp\left\{  -u \right\} \cdot \frac{u^{1/q - 1}}{\cancel{q}} \cdot \, \mathrm d u\\
=& \frac{1}{\Gamma(1/q)} \underbrace{ \int_{0}^{+\infty} \cdot\exp\left\{  -u \right\} \cdot u^{1/q - 1} \cdot \, \mathrm d u }_{ \Gamma(1/q) } = 1.\\
\end{align*}
$$
显然当 $q = 2$ 时该分布退化为 Gauss 分布。

接下来我们处理回归模型中标签分布假设是这个分布的情况。我们依然计算其对数似然函数：
$$
\begin{align*}
\ln p(\boldsymbol{t}|\boldsymbol{X}, \boldsymbol{w}, \sigma^{2}) &= \sum\limits_{n=1}^{N} \ln \left[\frac{{\color{blue} q}}{ {\color{blue} 2}(2\sigma^{2})^{1/q}{\color{blue} \Gamma(1/q)}} \exp\left\{  -\frac{|y(\boldsymbol{x}_{n}, \boldsymbol{w}) - t_{n}|^{q}}{2\sigma^{2}}  \right\}\right]\\
&= \sum\limits_{n=1}^{N} \left[ -\frac{1}{q}\ln (2\sigma^{2}) - \frac{|y(\boldsymbol{x}_{n}, \boldsymbol{w}) - t_{n}|^{q}}{2\sigma^{2}} + {\color{blue} C' }  \right]\\
&= - \frac{N}{q} \ln(2\sigma^{2}) - \frac{1}{2\sigma^{2}}\underbrace{ \sum\limits_{n=1}^{N} |y(\boldsymbol{x}_{n}, \boldsymbol{w}) - t_{n}|^{q} }_{ L_{q} } + \text{const.}
\end{align*}
$$
第二项正好就是之前提到的 Minkovski 损失。

## 习题 4.12  (★★☆) 不同情形下期望 Minkovski 损失的极小值点

首先写出期望 Minkovski 损失的表达式
$$
\begin{align*}
\mathbb{E}[L_{q}] &= \iint |y(\boldsymbol{x}) - t|^{q}p(\boldsymbol{x}, t)\,\mathrm{d}\boldsymbol{x}\,\mathrm{d}t
\end{align*}\tag{Min}
$$
我们想再次运用之前的泛函导数的技巧。但注意 $|\cdot|^{q}$ 在原点处不连续，我们考虑它的 **弱导数**，也即符号函数 $\text{sgn}(\cdot)$。于是我们只需要求
$$
\begin{align*}
&\int q|y(\boldsymbol{x}) - t|^{q-1} \cdot \text{sgn}(y(\boldsymbol{x}) - t) \cdot p(\boldsymbol{x}, t)\,\mathrm{d}t = 0 \\
\iff & \int |y(\boldsymbol{x}) - t|^{q-1} \cdot \text{sgn}(y(\boldsymbol{x}) - t) \cdot p(\boldsymbol{x}, t)\,\mathrm{d}t = 0 
\end{align*}\tag{Min-Diff}
$$
当 $q = 1$ 时，我们有
$$
\int_{y(\boldsymbol{x}) < t} p(\boldsymbol{x}, t) \, \mathrm dt = \int_{y(\boldsymbol{x}) \geqslant t} p(\boldsymbol{x}, t) \, \mathrm dt 
$$
这给出了 $t$ 的条件中位数。

当 $q \rightarrow 0$ 时，(Min) 中 $|y(\boldsymbol{x}) - t|^{q}$ 对任意的 $y(\boldsymbol{x}) - t \neq 0$ 将会趋于 $1$，然后在零处有一个 ”凹陷“。而我们分布中概率密度最高的那部分落在这个凹陷之中。所以对于每个 $\boldsymbol{x}$，$y(\boldsymbol{x})$ 给出条件分布 $p(t|\boldsymbol{x})$ 的众数（mode）。

---

PS. 私以为这题才是习题里面最难的一个，卡在求导这一步，因为觉得它不可导，于是在想能不能用次梯度（应该不行，因为 $q < 1$ 的时候它好像不是凸的），结果一查答案发现直接不讲道理的直接求导了......