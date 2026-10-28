# data_parallel

如果想将模型训练扩展到大的批次，则很快就会达到在单个 GPU 上可以做的极限。具体来说，会发生 `RuntimeError: CUDA out of memory`。
[梯度累计](梯度累计.md)、[Activation checkpointing](Activation%20checkpointing.md) 和 [CPU offloading](CPU%20offloading.md) 都可以一定程度上减少显存的占用，为了_有效地_扩展到更大的模型大小和不断增长的数据集，同时仍然在合理的时间内训练模型，我们需要将计算**分布在**一组机器上。

3 D 并行即：数据并行、张量并行、流水线并行
后两者可以统一划分到模型并行，区别是一个是层内并行，一个是层间并行。

这里介绍数据并行。

## Naive data parallel

一个很直觉的做法就是在 batch 维度上进行划分，各个卡上初始化完整的模型，然后将将划分的不同的 batch 发送到各个卡上进行前向传播和反向传播，再由一个卡整合梯度再下发给各 GPU，然后各 GPU 更新自己维护的模型参数。

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250723214626.png)


但这种做法显然有很多问题，需要有一个 gpu 担任梯度聚合和下发的角色，如果这个 gpu 出问题了怎么办？每一个 gpu 都需要维护完整的模型参数、梯度和优化器，这部分的显存没有得到减少；此外这种方式通讯量很大，详见[显存占用计算](../Infra/显存占用计算.md)

## DDP

DDP 解决的问题就是将 Server 上的通讯压力均衡转移到各个 worker 上（Server 即担任梯度聚合和下发的角色，而 worker 就是各个 gpu），因此引入了 [ring-all-reduce](ring-all-reduce.md) 算法来解决这个问题。需要把反向传播后的梯度切分成 N（world_size）份来进行 ring-all-reduce 算法。

## zero

[zero](zero.md)
## fsdp

[fsdp](fsdp.md)
## 参考

- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://cdn-lfs-us-1.hf.co/repos/e7/07/e7077a163ab0f314cedbb8ddd44667d765205ee536e8b4785fdd0872534107db/274a19a2577ed220cd3a102b4469c44310e4a7c8e8f8ebc36842d907cb51e127?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27The_Ultra-Scale_Playbook_Training_LLMs_on_GPU_Clusters.pdf%3B+filename%3D%22The_Ultra-Scale_Playbook_Training_LLMs_on_GPU_Clusters.pdf%22%3B&response-content-type=application%2Fpdf&Expires=1751735939&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1MTczNTkzOX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2U3LzA3L2U3MDc3YTE2M2FiMGYzMTRjZWRiYjhkZGQ0NDY2N2Q3NjUyMDVlZTUzNmU4YjQ3ODVmZGQwODcyNTM0MTA3ZGIvMjc0YTE5YTI1NzdlZDIyMGNkM2ExMDJiNDQ2OWM0NDMxMGU0YTdjOGU4ZjhlYmMzNjg0MmQ5MDdjYjUxZTEyNz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=jer8tObN1q6%7Eij8fX2vLIiox2VNNX0yAD9hjDxq9JXGDmzou6ONo7lnwIlrn%7ECbbaP-BXm80YdFMAgI2SbINgrxMfxLHTkp5IVwqppQ1INlC8K6JrZS3T8QlL4aY5jY7wX7SCUvweSuxEWA2QXMYwHWWV2Iy-OQAMkcdvvxDvjIZZwlYZqJ0tccDbpSYrOhNfkMcGYyxhp3HPgcEd6gVPydQE6g2wM8ErR04u-9dzwkJrIBowWrr8OSD9HJraRyr5XObTaBx3NEADn9De8Zyo%7EknwQs4MDxWSueQCYTlCfFElMF0%7EVMXYh%7EVfDSV5lZZiuxCFfke43Z12VSK5cMV%7EA__&Key-Pair-Id=K24J24Z295AEI9)
- [💥 Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups \| by Thomas Wolf \| HuggingFace \| Medium](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)
- [Training extremely large neural networks across thousands of GPUs.](https://www.jeremyjordan.me/distributed-training/)
- [# 图解大模型训练之：数据并行上篇(DP, DDP与ZeRO)](https://zhuanlan.zhihu.com/p/617133971)
