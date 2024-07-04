# warmup



在训练开始的时候，如果学习率太高的话，可能会导致loss来回跳动，会导致无法收敛，因此在训练开始的时候就可以设置一个很小的learning rate，然后随着训练的批次增加，逐渐增大学习率，直到达到原本想要设置的学习率。

关于warmup的好处，有：
- 有助于减缓模型在初始阶段对[mini-batch](https://www.zhihu.com/search?q=mini-batch&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A771252708%7D)的提前[过拟合](https://www.zhihu.com/search?q=%E8%BF%87%E6%8B%9F%E5%90%88&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A771252708%7D)现象，保持分布的平稳
- 有助于保持模型深层的稳定性。

warmup有助于网络的收敛，当达到预期的学习率时，之后的步骤就可以每固定批次进行学习率衰减，防止过拟合，以慢慢达到收敛。
