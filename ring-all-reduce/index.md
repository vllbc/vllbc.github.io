# ring-all-reduce

All-reduced=all-gather+reduce-scatter

- **All-Gather** ：将分布式数据汇总到所有节点，适用于需要**全局数据**同步的场景。
- **Reduce-Scatter**：将分布式数据进行**规约**并**分散**到所有节点，适用于需要局部结果分发的场景。
- **All-Reduce** ： Reduce-Scatter 和 All-Gather 的组合。

## All-gather


**核心功能**：将每个节点的部分数据汇总到所有节点，最终所有节点拥有**完整数据**副本。  
**适用场景**：模型并行中的参数同步、全局统计信息聚合。

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250717212746.png)

## Reduce-Scatter

**核心功能**：先对多节点数据进行规约（如求和），再将结果分散到各节点，使每个节点仅保留部分规约结果。  
**适用场景**：ZeRO显存优化、梯度分片更新。

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250717212840.png)


## 区别

- **All-Gather**：只进行数据收集和分发，不进行任何计算或规约操作。每个节点拥有所有节点的数据副本。
- **Reduce-Scatter**：先进行数据规约（reduce），然后再进行数据分散（scatter）。每个节点只拥有部分规约后的数据，而不是所有的数据

## 例子
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250717213413.png)
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250717213426.png)

## 通信量计算

假设模型参数大小为 $\theta$，GPU 个数为 N，则每一个梯度块大小为 $\frac{\theta}{N}$

对于单卡而言：
- Reduce-Scatter 阶段通讯量：$(N-1) \frac{\theta}{N}$
- All-Reduce 阶段通讯量：$(N-1) \frac{\theta}{N}$

单卡通讯量为 $2(N-1) \frac{\theta}{N}$，所有卡的通讯量为 $2(N-1) \theta$

## 参考

[# 分布式训练中All-Reduce、All-Gather、Reduce-Scatter原理介绍](https://zhuanlan.zhihu.com/p/17201336684)
