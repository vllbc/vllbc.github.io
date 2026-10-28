# Skywork-R1V3 Technical Report

- 思考时间和性能之间有一个平衡，超过这个平衡，就算增加思考时间性能也不会有大的提升了。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image20251013214458445.png)
- RL相对于SFT来说可以提高泛化能力，SFT之后out-domain数据集性能反而下降。
- 课程学习如果是针对同一个数据集，即同分布的情况，先训练简单题再逐步增加难度会提高性能。但如果是两个难度不同的数据集，并不一定能提高性能，这可以归因于分布偏移。因为两种数据的推理路径可能大相径庭。此外学习率也是个重要的超参数。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image20251013215212367.png)

- 熵的总体是下降趋势，但极值的数量增加，这说明RL训练会让模型某些地方更不确定。![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image20251013215100450.png)

