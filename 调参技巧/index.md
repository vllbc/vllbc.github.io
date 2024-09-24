# 调参技巧


- 基本原则：快速试错。
- 小步试错，快速迭代
- 可以试试无脑的配置
- 实时打印一些结果
- 自动调参：网格搜索、random search、贝叶斯优化、
- 参数初始化
- 学习率warmup，慢慢增加，然后学习率衰减。

# batch_size和lr
**大的batchsize收敛到[sharp minimum](https://zhida.zhihu.com/search?q=sharp+minimum&zhida_source=entity&is_preview=1)，而小的batchsize收敛到[flat minimum](https://zhida.zhihu.com/search?q=flat+minimum&zhida_source=entity&is_preview=1)，后者具有更好的泛化能力。**两者的区别就在于变化的趋势，一个快一个慢，如下图，造成这个现象的主要原因是小的batchsize带来的噪声有助于逃离sharp minimum。

**大的batchsize性能下降是因为训练时间不够长，本质上并不少batchsize的问题**，在同样的[epochs](https://zhida.zhihu.com/search?q=epochs&zhida_source=entity&is_preview=1)下的参数更新变少了，因此需要更长的迭代次数。

- **如果增加了学习率，那么batch size最好也跟着增加，这样收敛更稳定。**
- **尽量使用大的学习率，因为很多研究都表明更大的学习率有利于提高泛化能力。**如果真的要衰减，可以尝试其他办法，比如增加batch size，学习率对模型的收敛影响真的很大，慎重调整。

## 总结
**学习率直接影响模型的收敛状态，batchsize则影响模型的泛化性能**
