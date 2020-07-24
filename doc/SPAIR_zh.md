
__cell个数__
$$\begin{array}{l}
H = H_{img} / c_h \\
W = W_{img} / c_w
\end{array}$$

$value~[shape]$表示value的形状是shape. 

## 4.3 Encoder

编码器: $q_\phi(z|x)$
* 输入: 图像$x, [H_{img}, W_{img}, 3]$
* 输出: 
$\begin{aligned}
    &z_{where}, &[H, W, 4] \\
    &z_{depth}, &[H, W] \\
    &z_{pres}, &[H, W] \\
    &z_{what}, &[H, W, A]
\end{aligned}$

1. 特征提取

    一个CNN: $e^{conv}_\phi(x)$, 
    * 输出: 
    $
        features_{H\times W\times F}
    $

2. 产生分布参数

    一个MLP: $e^{lat}_\phi(features_F, sampled Objects_{neighboor})$

    * 分别计算: 
    $
        e^{lat}(features_F, sampled Objects_{neighboor}) =
        \left\{\begin{aligned}
        \mu_{where}^{i, j}&, \log\sigma_{where}^{i, j} &[4] \\
        \mu_{depth}^{i, j}&, \log\sigma_{depth}^{i, j} &[1] \\
        \beta_{pres}^{i, j}& &\in [0, 1]
        \end{aligned}\right.
    $
    * 取样: 
    $\begin{aligned}
        &\epsilon_1 \sim N(0, 1) &[4] \\
        &\epsilon_2 \sim N(0, 1) &[1] \\
        &g=-\log(-\log(u)), u \sim U(0, 1) &[2]
    \end{aligned}$
    * 输出: 
    $\begin{aligned}
        z_{where}^{i, j} &= \mu_{where}^{i, j}+\epsilon_1\times \sigma_{where}^{i, j} \\
        z_{depth}^{i, j} &= \mu_{depth}^{i, j}+\epsilon_2\times \sigma_{depth}^{i, j} \\
        z_{pres} &= (0, 1) \cdot softmax(\frac{(1 - \beta_{pres}^{ij}, \beta_{pres}^{ij}) + g}{\lambda}) \\
        &=softmax(\frac{(1 - \beta_{pres}^{ij}, \beta_{pres}^{ij}) + g}{\lambda})_1
    \end{aligned}$

3. 编码图像到$z_{what}$

    又一个MLP: $e^{obj}_\phi(x, z_{where}^{i, j})$

    * STN: $G = T(x, z_{where}^{i, j})$
    * 产生参数: 
    $\begin{array}{l}
        e^{obj}(G) =
        \mu_{what}^{i, j}, \log\sigma_{what}^{i, j} ~~~~~~~ [A]
    \end{array}$
    > 也就是$
        e^{obj}(T(x, z_{where}^{i, j})) =
        \mu_{what}^{i, j}, \log\sigma_{what}^{i, j} ~~~~~~~ [A]
    $
    * 取样: 
    $
        \epsilon_3 \sim N(0, 1)
    $
    * 输出: 
    $
        z_{what}^{i, j} = \mu_{what}^{i, j}+\epsilon_3\times \sigma_{what}^{i, j}
    $

## 4.4 Decoder

解码器: $d_\theta(z_{what}^{i, j})$
* 输入: 图像编码$z_{what}^{i, j}$
* 输出: 
$\begin{array}{l}
    o^{i, j}, [H, W, 3] \\
    \alpha^{i, j}, [H, W, 1]
\end{array}$

> $o$是表示物体的图像, 而$\alpha$是它的透明度

使用$z_{where}$限制物体的位置, $z_{depth}$表示物体在z轴上的层叠顺序. $z_{pres}$与透明度$\alpha$相乘, 保证没出现的物体透明度为0(领会精神)

* 绘图背景
    理论上应该用一个单独的网络来学习每张图的背景; 然而这篇论文(偷懒)只是用了单色背景. 可以用统计方法计算颜色, 也可以用一个MLP来学一个颜色出来. 

## 附录

### A. $p(z_{pres})$的先验概率

1. $z_{pres}$是一个长为$HW$的二值向量.
2. 令$z_{pres}$中含有C个1, C服从(截断)几何分布. 
3. 现在已知$nz(z_{pres})$就是$z_{pres}$中1的个数, 那么根据几何分布的分布, $$p(C=nz(z_{pres}))=s(1-s)^{nz(z_{pres})}$$
4. 由于其实是截断的几何分布(即$nz(z_{pres})\in\{0, ..., HW\}$而不能一直往大了取)于是除以一个规范项: $$\begin{aligned}p(C=nz(z_{pres}))&=\frac{s(1-s)^{nz(z_{pres})}}{\sum^{HW}_{c=0}s(1-s)^c} \\ &=\frac{s(1-s)^{nz(z_{pres})}}{1-(1-s)^{HW+1}}\end{aligned}\tag{A.1}$$
5. 如果C确定的话, 那么$z_{pres}$的取值就是一个排列组合问题. 总共有$C_{HW}^{nz(pres)}$种取值. 每种取值的概率相同, 那么这个概率就是$$p(z_{pres}|C=nz(z_{pres})=\frac{1}{C_{HW}^{nz(pres)}}\tag{A.2}$$
6. 于是先验概率$$\begin{aligned}p(z_{pres})&=p(z_{pres},C=nz(z_{pres}))\\&=p(z_{pres}|C=nz(z_{pres}))p(C=nz(z_{pres}))\\&=\frac{s(1-s)^{nz(z_{pres})}}{1-(1-s)^{HW+1}C_{HW}^{nz(pres)}}\end{aligned}\tag{A.3}$$