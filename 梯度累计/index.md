# 梯度累计

**梯度累积**使我们能够通过按顺序处理较小的批次来扩展到更大的有效批次。我们不是一次计算整个批次的梯度（这需要将所有激活存储在内存中），而是在更新模型参数之前将每个小批次的梯度相加。这减少了内存使用量，但需要更多的向前/向后传递。

## 代码

```python
loss = loss / gradient_accumulation_stepsloss.backward()
if step% gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
    step+=1
```

gradient_accumulation_steps 是梯度累积次数，累积几次，原本的 loss 就要除以几，这是为了对多个批次的数据的梯度做累积。

有人说，应该有这一行代码才算累加

```Plain
losses += loss
```

这样理解是错误的。

要明白最重要的一点是，梯度累加，累加的并不是损失，而是根据损失得到的梯度。

  

## 梯度累计如何节省显存

- 减少瞬时激活值显存：每次仅处理小批量的数据，激活值显存占用降低为原来的 `1/k`（例如 `k=4` 时，显存占用降至 25%）。
    
- 复用显存：每次小批量计算完成后，释放当前激活值显存，供下一次计算使用（显存占用峰值始终为小批量对应的量）。
    
- 梯度显存不变：模型参数和梯度的显存占用与批量大小无关，因此不受影响（但需额外存储累积梯度的变量，这部分开销极小）。

## [梯度累积两次，跟 batch size 增大 2 倍，在多数情况下，效果一样吗？](https://www.zhihu.com/question/583011902/answer/7205474551)（loss 的 3 次平均）

理论上，[梯度累计](https://zhida.zhihu.com/search?content_id=694556710&content_type=Answer&match_order=1&q=%E6%A2%AF%E5%BA%A6%E7%B4%AF%E8%AE%A1&zhida_source=entity)在数学上应该等同于[全批量训练](https://zhida.zhihu.com/search?content_id=694556710&content_type=Answer&match_order=1&q=%E5%85%A8%E6%89%B9%E9%87%8F%E8%AE%AD%E7%BB%83&zhida_source=entity)，但实际发现 loss 并不匹配。( [Gradient accumulation yields worse results than the equivalent batch size · Issue #2175 · huggingface/trl](https://link.zhihu.com/?target=https%3A//github.com/huggingface/trl/issues/2175))

一般情况下，loss 计算会经历三次平均

1. micro batch 维度，分母是这个 micro batch 中的所有 label 不是 -100 的 token 数**（不同 token 之间 loss 的平均）**
    
2. DP 维度，分母是 DP size **（和** **GPU** **数量相关，不同机器之间 loss 的平均）**
    
3. 梯度累加维度，分母是梯度累加数。**（不同 batch 之间的 loss 的平均）**

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250710002102.png)

