# zero

分为zero1、zero2、zero3，虽然zero3对模型进行了分割，但是本质上还是属于数据并行，因为在前向传播和反向传播需要all-gather模型参数，需要完整的模型权重。

先来一张经典的图片：

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250805202834.png)

这里的K具体是多少可以看[显存占用计算](../infra/显存占用计算.md)

接下来逐个介绍各个stage zero。随着zero优化的深入，越来越节省显存，但通讯时间也大大增加，导致训练时间增加。
## zero1
针对优化器状态进行分割。

## zero2
针对优化器状态和梯度进行分割。

## zero3
针对优化器状态、梯度和模型参数进行分割。
