# veRL中colocate实现源码解析（二）


> 来源：[【AI Infra】【RLHF框架】二、VeRL中colocate实现源码解析](https://zhuanlan.zhihu.com/p/31595392436)（作者：白强伟）
>
> 前置知识：
> - [[Engineering/coding/verl/ray前置知识]]
> - [[Engineering/coding/verl/veRL_Ray执行流程源码解析]]

`colocate` 的作用是使多个 Worker 共享相同的资源池。目前版本 verl 中所有模型的 Worker 都共享相同的资源池：`global_pool`。这篇博客主要通过例子和源代码理解 verl 中 colocate 的实现。

## 一、一个例子

下面简单修改了 verl 的单元测试作为示例，先直观感受下 colocate 的作用：

```python
import ray

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup, create_colocated_worker_cls

from verl import DataProto

@ray.remote
class Actor(Worker):
    def __init__(self) -> None:
        super().__init__()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        data.batch['a'] = data.batch['a'].to("cuda")
        data.batch['a'] += self.rank
        return data

@ray.remote
class Critic(Worker):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def sub(self, data: DataProto):
        data.batch['a'] = data.batch['a'].to("cuda")
        data.batch['a'] -= self.config['b']
        return data

def test_colocated_workers():
    ray.init()

    import torch
    # 构建一个 DataProto，其中属性 a 是维度为 10 的零向量
    data = DataProto.from_dict({'a': torch.zeros(10)})
    print(data.batch["a"])
    # 利用 RayClassWithInitArgs 将自定义的 worker 和参数封装起来
    actor_cls = RayClassWithInitArgs(cls=Actor)
    critic_cls = RayClassWithInitArgs(cls=Critic, config={'b': 10})
    # 定义资源池，仅包含一个 2 GPU 的节点
    resource_pool = RayResourcePool(process_on_nodes=[2])
    # 利用 create_colocated_worker_cls 将自定义的两个 worker 绑定到 WorkerDict 上
    cls_dict = {'actor': actor_cls, 'critic': critic_cls}
    ray_cls_with_init = create_colocated_worker_cls(cls_dict)
    # 启动 WorkerDict
    wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    # 分别获取 actor 和 critic 的 workergroup
    spawn_wg = wg_dict.spawn(prefix_set=cls_dict.keys())
    colocated_actor_wg = spawn_wg['actor']
    colocated_critic_wg = spawn_wg['critic']
    # actor 执行 add、critic 执行 sub
    actor_output = colocated_actor_wg.add(data)
    critic_output = colocated_critic_wg.sub(data)
    # actor_output.batch["a"] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # critic_output.batch["a"] == [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10]
    print(actor_output.batch["a"])
    print(critic_output.batch["a"])
    ray.shutdown()

if __name__ == '__main__':
    test_colocated_workers()
```

### 1. Actor 和 Critic 的解释

`Actor` 和 `Critic` 的定义比较简单，在 `add` 和 `sub` 方法上使用了装饰器 `@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)`。

这个装饰器的作用就是 **数据并行**，即将数据 chunk 后交由定义 worker 的两个实例分别计算，再将结果合并。

下面是 Actor 执行过程的示意图，Critic 类似：

![Actor DP 流程：dispatch -> rank(0) + rank(1) -> collect](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260604155100_blog2_actor_dp.jpg)

### 2. Actor 和 Critic 绑定至 WorkerDict

通过打印一些元信息，可知 `ray_cls_with_init` 是 `RayClassWithInitArgs`，其中持有的 `cls` 是一个定义为 `WorkerDict` 的类，其基类同样是 `Worker`：

```python
cls_dict = {'actor': actor_cls, 'critic': critic_cls}
ray_cls_with_init = create_colocated_worker_cls(cls_dict)
print(type(ray_cls_with_init))                              # RayClassWithInitArgs
print(ray_cls_with_init.cls.__ray_actor_class__)            # WorkerDict
print(ray_cls_with_init.cls.__ray_actor_class__.__base__)   # Worker
print(ray_cls_with_init.cls.actor_add)
print(ray_cls_with_init.cls.critic_sub)
```

`create_colocated_worker_cls` 的工作过程示意图：

![RayClassWithInitArgs -> create_colocated_worker_cls -> WorkerDict](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260604155100_blog2_workerdict.jpg)

### 3. 启动 WorkerDict 并执行操作

```python
# 启动 WorkerDict
wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
spawn_wg = wg_dict.spawn(prefix_set=cls_dict.keys())
# 获得 actor 和 critic 的 RayWorkerGroup
colocated_actor_wg = spawn_wg['actor']
colocated_critic_wg = spawn_wg['critic']
# 执行
actor_output = colocated_actor_wg.add(data)
critic_output = colocated_critic_wg.sub(data)
```

通过 `RayWorkerGroup` 启动 WorkerDict。`spawn` 方法返回一个 `RayWorkerGroup` 的字典：

```python
{'actor': <verl.single_controller.ray.base.RayWorkerGroup object at 0x7f2efc719790>,
 'critic': <verl.single_controller.ray.base.RayWorkerGroup object at 0x7f2f0e125dd0>}
```

获得 actor 和 critic 的 RayWorkerGroup 后直接执行操作即可。

## 二、`create_colocated_worker_cls` 源码解析

简化后的源码：

```python
# 原始代码位于 verl/single_controller/ray/base.py
def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    # {"actor": Actor, "critic": Critic}
    cls_dict = {}
    # {'actor': {'args': (...), 'kwargs': {}}, 'critic': {'args': (...), 'kwargs': {...}}}
    init_args_dict = {}
    # worker_cls 是指 Actor 和 Critic，其实就是 Worker
    worker_cls = None
    for key, cls in class_dict.items():
        worker_cls = cls.cls.__ray_actor_class__.__base__
        cls_dict[key] = cls.cls
        init_args_dict[key] = {'args': cls.args, 'kwargs': cls.kwargs}

    class WorkerDict(worker_cls):
        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                # 去除掉 ray.remote 的包装，这里 user_defined_cls 就是 Actor 和 Critic 两个类
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                with patch.dict(os.environ, {'DISABLE_WORKER_INIT': '1'}):
                    self.worker_dict[key] = user_defined_cls(
                        *init_args_dict[key].get('args', ()),
                        **init_args_dict[key].get('kwargs', {}))

    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    return remote_cls
```

### 1. 类 `WorkerDict`

`WorkerDict` 是定义在 `create_colocated_worker_cls` 内部的类，其初始化方法 `__init__` 中核心就是构建 `self.worker_dict`。

在本文例子中，key 是 `actor` 和 `critic`，对应的 value 就是 `Actor` 和 `Critic` 的实例。

**关键点**：使用 `patch.dict(os.environ, {'DISABLE_WORKER_INIT': '1'})` 来 **抑制子 worker 的初始化**，避免重复执行 `super().__init__()` 中的环境变量配置 / register_center 创建等副作用，因为这些工作由 `WorkerDict` 自己（作为 master）已经完成。

### 2. `_bind_workers_method_to_parent`

一句话说明：将 `user_defined_cls` 中使用装饰器 `register` 的方法绑定到 `WorkerDict`，key 是方法绑定至 `WorkerDict` 的前缀。

```python
# 原始代码位于 verl/single_controller/ray/base.py
def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    for method_name in dir(user_defined_cls):
        if hasattr(method, MAGIC_ATTR):
            # 遍历 user_defined_cls 的所有方法，找到使用装饰器 `register` 装饰的方法
            def generate_function(name):
                def func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)
                return func

            func = generate_function(method_name)
            # 将原始函数 `add` 和 `sub` 的 MAGIC_ATTR 绑定到 func 上
            setattr(func, MAGIC_ATTR, getattr(method, MAGIC_ATTR))
            try:
                method_name_with_prefix = key + '_' + method_name
                setattr(cls, method_name_with_prefix, func)
            except Exception as e:
                raise ValueError(f'Fail to set method_name {method_name}')
```

`_bind_workers_method_to_parent` 实现可能比较难理解一些，先跳过 `generate_function`。

```python
setattr(func, MAGIC_ATTR, getattr(method, MAGIC_ATTR))
```

这行代码主要是将 `add` 或 `sub` 的装饰器提供的信息复制到 `func` 上。

```python
method_name_with_prefix = key + '_' + method_name
setattr(cls, method_name_with_prefix, func)
```

这段代码就是将 `Actor` 的方法 `add` 或 `Critic` 的方法 `sub` 绑定至 `WorkerDict` 上。

**理解 `generate_function` 的等价代码**：

将 `add` 绑定到 `WorkerDict` 后，等价于：

```python
class WorkerDict(worker_cls):
    def __init__(self):
        ...

    def actor_add(self, *args, **kwargs):
        return getattr(self.worker_dict["actor"], "add")(*args, **kwargs)

    def critic_sub(self, *args, **kwargs):
        return getattr(self.worker_dict["critic"], "sub")(*args, **kwargs)
```

## 三、`spawn` 的作用

回顾一下开始例子中的这段代码：

```python
wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
spawn_wg = wg_dict.spawn(prefix_set=cls_dict.keys())
print(spawn_wg["actor"].workers)
print(spawn_wg["critic"].workers)
```

`wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)` 这行代码会启动两个 `WorkerDict` 的远程实例。

> 注意：这里不使用 Ray 中 actor 的称呼，因为和示例中的 actor 会混淆。

在调用 `spawn` 后返回字典 `spawn_wg`，打印 `spawn_wg["actor"].workers` 和 `spawn_wg["critic"].workers` 可以发现 **这两个 workergroup 中持有的 workers 都是相同的**（都是同一组 `WorkerDict` 实例）。

那么 `spawn` 的作用是什么？看下源码：

```python
def spawn(self, prefix_set):
    def _rebind_actor_methods(worker_group, actor_name):
        prefix: str = actor_name + '_'
        for method_name in dir(worker_group):
            if method_name.startswith(prefix):
                # only valid when Python >= 3.9
                original_method_name = method_name.removeprefix(prefix)
                method = getattr(worker_group, method_name)
                setattr(worker_group, original_method_name, method)

    new_worker_group_dict = {}
    for prefix in prefix_set:
        # 从现有的 workers 中选出名字为 self._worker_names 的 worker 并构成新的 RayWorkerGroup
        new_worker_group = self.from_detached(worker_names=self._worker_names,
                                              ray_cls_with_init=self.ray_cls_with_init)
        # 将带有前缀的方法名移除前缀。例如：`actor_add` -> `add`
        _rebind_actor_methods(new_worker_group, prefix)
        new_worker_group_dict[prefix] = new_worker_group
    return new_worker_group_dict
```

- `self.from_detached` 作用是利用现有的 worker 构造一个新的 `RayWorkerGroup`，**其并不会启动新的 Worker**。
- `_rebind_actor_methods` 则是将 `actor_add` 这种带前缀的方法名改为 `add`，然后将 `add` 绑定到新的 `RayWorkerGroup` 上。

**所以，`spawn` 的作用就是保证可以像非 colocate 那样来执行具体的功能**——返回的 `spawn_wg['actor']` 表面上是一个独立的 `RayWorkerGroup`（只持有 `actor` 相关的 `add` 方法），但底层 worker 仍共享同一份 `WorkerDict`。

## 参考资料

- [verl 官方文档 - Hybrid Flow](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
- [veRL GitHub 仓库](https://github.com/volcengine/verl)
- [VeRL(v0.2.0.post2)系列文章](https://zhuanlan.zhihu.com/p/1918280771946382774)

## 相关笔记

- [[Engineering/coding/verl/ray前置知识]]
- [[Engineering/coding/verl/veRL_Ray执行流程源码解析]]
- [[Engineering/coding/verl/verl总体概览]]
- [[Engineering/coding/verl/init_workers详解]]
- [[Engineering/coding/verl/DataProto]]

