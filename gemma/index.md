# Gemma

## Gemma 3

### QK-Norm

简单来说就是在Q和K矩阵上进行RMSNorm，即：

$$
\begin{aligned}
O &= softmax(\bar{Q}\bar{K}^T)V \\
\bar{Q} &=RMSNorm(Q) \\ 
\bar{K} &=RMSNorm(V)
\end{aligned}\
$$

但这种方法的问题是不适用MLA的推理阶段，因为推理阶段的MLA将Wk吸取到了Q中，具体见[MLA](../Attention/MLA.md)


