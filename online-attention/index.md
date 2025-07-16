# online attention

### 3-pass 
$\mathsf{NO}$ TATIONS

$\{m_i\}{:}\max_{j=1}^i\left\{x_j\right\}$, with initial value $m_0=-\infty.$
$\{d_i\}{:}\sum_{j=1}^ie^{x_j-m_N}$, with initial value $d_0=0,d_N$ is the denominator of safe softmax.
$\{a_i\}{:\text{ the final softmax value}}.$

BODY
$\textbf{for }i\leftarrow 1, N\textbf{ do}$
$$m_i\leftarrow\max\left(m_{i-1},x_i\right)$$
$\mathbf{end}$

$\textbf{for }i\leftarrow 1, N\textbf{ do}$
$$d_i\leftarrow d_{i-1}+e^{x_i-m_N}$$
$\mathbf{end}$

$\textbf{for }i\leftarrow 1, N\textbf{ do}$
$$a_i\leftarrow\frac{e^{x_i-m_N}}{d_N}$$
$\mathbf{end}$

这是 3 step 计算 attention 的方法，每一步都需要上一步的结果才可以继续计算。这样的话由于 sram 中没有足够的存储空间，因此需要多次访存。
### Online attention
$$\begin{aligned}
d_i^{\prime}& =\sum_{j=1}^ie^{x_j-m_i} \\
&= \left(\sum_{j=1}^{i-1} e^{x_j-m_i}\right)+e^{x_i-m_i} \\
&= \left(\sum_{j=1}^{i-1} e^{x_j-m_{i-1}}\right)e^{m_{i-1}-m_i}+e^{x_i-m_i} \\
&= d_{i-1}' e^{m_{i-1}-m_i}+e^{x_i-m_i}
\end{aligned}$$
找到迭代式之后就可以从 3 step 降到 2 step
$$\begin{aligned}&\mathbf{for~}i\leftarrow1,N\textbf{ do}\\&&&m_i&&\leftarrow&\max\left(m_{i-1},x_i\right)\\&&&d_i^{\prime}&&\leftarrow&d_{i-1}^{\prime}e^{m_{i-1}-m_i}+e^{x_i-m_i}\\&\mathbf{end}\\&\mathbf{for~}i\leftarrow1,N\textbf{ do}\\&&&a_i\leftarrow&&\frac{e^{x_i-m_N}}{d_N^{\prime}}\\&\mathbf{end}\end{aligned}$$
好像 FLOPs 计算量并没有减少，甚至还略有增加，因为现在每次都需要计算额外的 scale

> X 值，也就是 pre-softmax logits，由于需要 O (N^2) 的显存无法放在 SRAM 中。因此：  
> 1. 要么提前计算好 x，保存在全局显存中，需要 O (N^2) 的显存，容易爆显存。  
> 2. 要么在算法中 online 计算，每次循环中去 load 一部分 Q，K 到片上内存，计算得到 x。

Attention 优化的目标就是避开第一种情况，尽可能节省显存，否则，LLM 根本无法处理类似 100 K 以上这种 long context 的情况。而对于第二种情况，我们不需要保存中间矩阵 x，节省了显存，但是计算没有节省，并且增加了 HBM IO Accesses（需要不断地 load Q, K）。此时，2-pass 算法相对于 3-pass 算法，可以减少一次整体的 load Q, K 以及减少一次对 xi 的 online recompute，因为在 2-pass 的第一个 pass 中， xi 是被两次计算共享的。类似 online-softmax 这种算法，对应到 Attention 中的应用，就是 Memory Efficient Attention（注意不是 FlashAttention）。
