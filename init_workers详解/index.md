# init_workers详解

前置知识，[ray前置知识](ray前置知识.md)
我将用此文来详细介绍veRL中有关single_controller和SPMD的相关内容。本文不涉及ppo训练相关，只是记录一下理解veRL架构实现的核心。

实现single_controller方法的核心有以下方法：

- Worker
- RayResourcePool
- RayWorkerGroup
- RayClassWithInitArgs

实现SPMD的有以下方法：

- register
- dispatch_fn和collect_fn
- execute_fn
- func_generator

推荐阅读：[verl 解读 - 源码阅读(part3)](https://zhuanlan.zhihu.com/p/1926337913513292229)


