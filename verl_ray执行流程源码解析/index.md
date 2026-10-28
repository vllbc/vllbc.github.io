# veRL基于Ray的执行流程源码解析（一）


> 来源：[【AI Infra】【RLHF框架】一、VeRL中基于Ray的执行流程源码解析](https://zhuanlan.zhihu.com/p/29997527557)（作者：白强伟）
>
> 前置知识：[[Engineering/coding/verl/ray前置知识]] 提到此篇博客

VeRL 是字节开源的 LLM RL 后训练框架，完整设计思路见论文《HybridFlow: A Flexible and Efficient RLHF Framework》。这篇博客从源码角度深入理解其基于 Ray 的执行流程，不涉及具体 RL 算法。

## 一、Overview

### 1. 一个例子

假设有 4 个 GPU，每个 GPU 实例化一个 `GPUAccumulator`，其成员变量 `value` 初始化为 GPU rank，然后对所有 value 执行加 1 操作：

```python
import ray
import torch

from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayWorkerGroup, RayResourcePool, RayClassWithInitArgs
from verl.single_controller.base.decorator import register, Dispatch

@ray.remote
class GPUAccumulator(Worker):
    def __init__(self):
        super().__init__()
        self.value = torch.zeros(size=(1,), device="cuda") + self.rank

    @register(Dispatch.ONE_TO_ALL)
    def add(self, x):
        self.value += x
        print(f'rank {self.rank}, value: {self.value}')
        return self.value.cpu()

class_with_args = RayClassWithInitArgs(GPUAccumulator)
resource_pool = RayResourcePool([4], use_gpu=True)
workergroup = RayWorkerGroup(resource_pool, class_with_args)
print(workergroup.add(x=1))  # 输出：[tensor([1.]), tensor([2.]), tensor([3.]), tensor([4.])]
```

执行过程示意图（DriverProcess → WorkerGroup → ResourcePool → Worker）：

![GPUAccumulator 执行流程](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260604155100_blog1_example.jpg)

### 2. 整体设计

为了实现上面的功能，VeRL 中抽象出了 **Worker**、**WorkerGroup**、**ResourcePool**、**DriverProcess** 的概念：

- **Worker**：基础执行单元
- **WorkerGroup**：Worker 的集合，统一调度
- **ResourcePool**：资源池
- **DriverProcess**：驱动进程

下面是官方文档中这些概念的示例图：

![Worker/WorkerGroup/ResourcePool 概念图](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260604155100_blog1_overview.jpg)

后面的章节通过源码的方式深入理解这些概念。

## 二、数据交换协议：DataProto

`DataProto` 是 veRL 中各种功能之间用于数据交换的标准协议。

### 1. 基础结构

```python
# 原始代码位于 verl/protocol.py
@dataclass
class DataProto:
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)
    ...
```

`DataProto` 持有三种类型的数据：`batch`、`non_tensor_batch` 和 `meta_info`。通过命名可以看出，`DataProto` 是以 batch 的方式组织数据的。

### 2. 使用示例

更多使用示例见 `tests/utility/test_tensor_dict_utilities.py`：

```python
# 构造
obs = torch.randn(100, 10)  # batch_size 为 100 的 tensor
labels = [random.choice(['abc', 'cde']) for _ in range(100)]  # 随机挑选 'abc' 和 'cde' 构造大小为 100 的 list
dataset = DataProto.from_dict(tensors={'obs': obs}, non_tensors={'labels': labels})  # 实例化
# pop
poped_dataset = dataset.pop(batch_keys=['obs'])  # 将 batch 中的 "obs" 弹出，poped_dataset 仍为 DataProto
```

### 3. DataProtoFuture

在 `verl/protocol.py` 中除 `DataProto` 外，还定义了 Future 版本的 `DataProtoFuture`，保存了 `ray.ObjectRef` 对象列表 `futures` 以及两个函数 `collect_fn` 和 `dispatch_fn`：

```python
@dataclass
class DataProtoFuture:
    collect_fn: Callable  # 用于将 future 对象列表简化为一个 DataProto
    futures: List[ray.ObjectRef]
    dispatch_fn: Callable = None  # 用于将 DataProto 划分为大小为 world_size 的多个 DataProto 列表

    def get(self):
        output = ray.get(self.futures)
        for o in output:
            assert isinstance(o, DataProto)
        output = self.collect_fn(output)
        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)
        return output
```

## 三、基础执行单元：Worker

`Worker` 负责执行具体任务的基础执行单元，保存分布式执行时必要的信息。具体的执行任务需要用户继承 `Worker` 类来实现。

### 1. 代码解析

在 verl 中，通过 `WorkerGroup` 统一实例化并调度 Worker，概念上存在 **master worker**。先看单个 Worker。

简单来说，`Worker` 类负责存储 `world_size`、`rank` 等分布式计算的相关信息并配置环境变量，**Master Worker** 相对于其他 Worker 需要额外保存 `MASTER_ADDR` 和 `MASTER_PORT` 的信息。简化后的代码：

```python
# 简化后代码，原始代码位于 verl/single_controller/base/worker.py
class Worker(WorkerHelper):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        rank = os.environ.get("RANK", None)
        worker_group_prefix = os.environ.get("WG_PREFIX", None)
        if None not in [rank, worker_group_prefix] and 'ActorClass(' not in cls.__name__:
            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))
        return instance

    def _configure_before_init(self, register_center_name: str, rank: int):
        # rank=0 时，配置 MASTER_ADDR 和 MASTER_PORT 环境变量
        if rank == 0:
            master_addr, master_port = self.get_availale_master_addr_port()
            rank_zero_info = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            }
            if os.getenv("WG_BACKEND", None) == "ray":
                from verl.single_controller.base.register_center.ray import create_worker_group_register_center
                self.register_center = create_worker_group_register_center(name=register_center_name, info=rank_zero_info)
            os.environ.update(rank_zero_info)

    def __init__(self, cuda_visible_devices=None) -> None:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        self._rank = rank
        self._world_size = world_size

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        store = {
            '_world_size': world_size,
            '_rank': rank,
            '_local_world_size': local_world_size,
            '_local_rank': local_rank,
            '_master_addr': master_addr,
            '_master_port': master_port
        }
        # WorkerMeta 仅是将 store 信息存储在实例对象中
        meta = WorkerMeta(store=store)
        # 将 meta(store) 信息更新到当前实例的 __dict__ 中并配置环境变量
        self._configure_with_meta(meta=meta)
```

**要点**：
- `Worker` 继承自 `WorkerHelper`，`WorkerHelper` 提供唯一的功能就是获得当前节点的 IP 地址和可用端口号。
- 核心方法包括 `__new__` 和 `__init__`：
  - `__new__` 针对 rank 0 配置 `MASTER_ADDR` 和 `MASTER_PORT`
  - `__init__` 将 `world_size`、`rank` 保存在当前实例中

### 2. 自定义 Worker 的例子

Worker 本身不执行具体计算任务，需要用户继承该类实现自定义计算任务。下面是一个模拟"数据并行 (DP) 前向"的例子：

```python
import os
import ray
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from verl.single_controller.base import Worker

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.linear = nn.Linear(100, 1)

    def compute_loss(self, x, y):
        pred = self.linear(x)
        loss = F.mse_loss(pred, y)
        return loss

@ray.remote(num_gpus=1)
class MyNetWorker(Worker):
    def __init__(self):
        super().__init__()
        self.network = MyNet()
        self.device = torch.device(f"cuda")
        self.network.to(self.device)

    def compute_network_loss(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        return self.network.compute_loss(inputs, targets).item()

if __name__ == "__main__":
    ray.init()
    options0 = {'runtime_env': {'env_vars': {'LOCAL_WORLD_SIZE': '2', 'LOCAL_RANK': '0',
                                              'WORLD_SIZE': '2', 'RANK': '0',
                                              'WG_PREFIX': 'test', 'WG_BACKEND': 'ray'}}}
    options1 = {'runtime_env': {'env_vars': {'LOCAL_WORLD_SIZE': '2', 'LOCAL_RANK': '1',
                                              'WORLD_SIZE': '2', 'RANK': '1',
                                              'WG_PREFIX': 'test', 'WG_BACKEND': 'ray'}}}
    # 初始化 Master Worker
    worker0 = MyNetWorker.options(**options0).remote()
    time.sleep(10)  # 等待 register_center_actor 创建
    register_center_actor = ray.get_actor("test_register_center")
    rank_zero_info = ray.get(register_center_actor.get_rank_zero_info.remote())
    print(f"rank_zero_info: {rank_zero_info}")
    # 更新 MASTER_ADDR 和 MASTER_PORT 至 options1，并创建 worker1
    options1['runtime_env']['env_vars'].update(rank_zero_info)
    worker1 = MyNetWorker.options(**options1).remote()

    # 生成测试数据
    batch_size = 16
    inputs = torch.randn(batch_size, 100)
    targets = torch.randn(batch_size, 1)

    # 分割数据为两份
    split_size = batch_size // 2
    inputs_split = torch.chunk(inputs, 2)
    targets_split = torch.chunk(targets, 2)

    # 并行计算
    future0 = worker0.compute_network_loss.remote(inputs_split[0], targets_split[0])
    future1 = worker1.compute_network_loss.remote(inputs_split[1], targets_split[1])

    # 获取结果并平均
    loss0 = ray.get(future0)
    loss1 = ray.get(future1)
    avg_loss = (loss0 + loss1) / 2

    print(f"Worker 0 Loss: {loss0:.4f}")
    print(f"Worker 1 Loss: {loss1:.4f}")
    print(f"Averaged Loss: {avg_loss:.4f}")

    ray.shutdown()
```

### 3. 装饰器

在自定义 Worker 时，可以使用装饰器 `register` 扩展函数功能。为什么要用这个装饰器，在 [[#六、RayWorkerGroup]] 章节详细介绍。

#### 3.1 一个示例

```python
import os
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch, Execute

os.environ['LOCAL_WORLD_SIZE'] = '2'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '2'
os.environ['RANK'] = '0'
os.environ['WG_PREFIX'] = 'test'
os.environ['WG_BACKEND'] = 'ray'

class GPUAccumulatorDecorator(Worker):
    def __init__(self):
        super().__init__()
        self.value = torch.zeros(size=(1,), device="cuda") + self.rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def add(self, x):
        self.value += x
        return self.value.cpu()

worker = GPUAccumulatorDecorator()
print(worker.add.attrs_3141562937)
```

输出：`{'dispatch_mode': <Dispatch.ONE_TO_ALL: 1>, 'execute_mode': <Execute.RANK_ZERO: 1>, 'blocking': True}`

所以装饰器 `register` 的核心功能就是给函数添加一个新属性 `attrs_3141562937`，该属性保存 `dispatch_mode` 等信息。

#### 3.2 源码解析

```python
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL,
             blocking=True, materialize_futures=True):
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                # _materialize_futures 将 future 转换为实际的数据
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        # 为被装饰函数新增一个 MAGIC_ATTR 属性
        attrs = {'dispatch_mode': dispatch_mode, 'execute_mode': execute_mode, 'blocking': blocking}
        setattr(inner, MAGIC_ATTR, attrs)
        return inner

    return decorator
```

`register` 仅包含两个功能：
1. 若参数是 future，则获得实际的值
2. 为被装饰函数添加 `MAGIC_ATTR` 属性

## 四、RayClassWithInitArgs

`RayClassWithInitArgs` 保存了通过 `@ray.remote` 定义的 Actor 类，以及一些用于异步调用该 Actor 时所需要的参数。更像是一个工厂类，在调用 `__call__` 时再启动 Actor。

```python
# 简化后代码，原始代码位于 verl/single_controller/ray/base.py
class RayClassWithInitArgs(ClassWithInitArgs):
    def __init__(self, cls, *args, **kwargs) -> None:
        super().__init__(cls, *args, **kwargs)  # 保存 Actor
        self._options = {}
        self._additional_resource = {}

    def __call__(self,
                 placement_group,
                 placement_group_bundle_idx,
                 use_gpu: bool = True,
                 num_gpus=1,
                 sharing_with=None) -> Any:
        # 调度策略
        options = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=placement_group_bundle_idx)}
        # 更新 actor 启动的一些参数
        options.update(self._options)

        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v
        # Ray 中异步启动 Actor
        return self.cls.options(**options).remote(*self.args, **self.kwargs)
```

## 五、资源池：RayResourcePool

### 1. Ray 中的 Placement Group

`Placement Group` 是 Ray 中的资源组概念，允许用户跨多个节点原子性地保留一组资源。创建时需指定 `bundles`，单个 bundle 是一个"资源"的集合，也是 `Placement Group` 的资源预留单位。

例如，包含 8 GPU 16 CPU 的节点，想用一半资源：

```python
import ray
from ray.util.placement_group

# 创建包含 8GPU、16CPU 的 Ray 集群
ray.init(num_cpus=16, num_gpus=8)
# 创建包含 4 个 bundle 的 placement group，共使用一半资源
pg = placement_group(bundles=[{"CPU": 2, "GPU": 1},
                             {"CPU": 2, "GPU": 1},
                             {"CPU": 2, "GPU": 1},
                             {"CPU": 2, "GPU": 1}])
# 等待资源组创建完成
ray.get(pg.ready(), timeout=10)
```

### 2. RayResourcePool

`RayResourcePool` 继承自 `ResourcePool`，`ResourcePool` 负责存储资源相关信息，`RayResourcePool` 则通过 Ray 的 Placement Group 实现资源池分配。

![RayResourcePool 与 PlacementGroup 的关系](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260604155100_blog1_rayresourcepool.jpg)

```python
resource_pool = RayResourcePool(process_on_nodes=[4, 4], max_colocate_count=2, use_gpu=True)  # 创建资源池
pgs = resource_pool.get_placement_groups()  # 创建 placement group 列表
```

该段代码会创建 **两个 Placement Group**，每个包含 4 个 GPU 和 8 个 CPU。单个 Placement Group 为：

```python
pg = placement_group(bundles=[{"CPU": 2, "GPU": 1},
                             {"CPU": 2, "GPU": 1},
                             {"CPU": 2, "GPU": 1},
                             {"CPU": 2, "GPU": 1}])
```

即 `process_on_nodes` 指定要创建几个 Placement Group 以及每个包含多少 GPU。`max_colocate_count` 是 bundle 中单个 GPU 对应多少个 CPU，因为 colocate 的 actor 至少要有 1 个 CPU。

## 六、RayWorkerGroup

`RayWorkerGroup` 扮演着资源分配和调度 Worker 的协调器角色，核心方法包括：`_init_with_resource_pool`、`_bind_worker_method` 和 `execute_all_async`。

`__init__` 中核心两件事：
1. 在指定的资源池上启动 worker
2. 将 worker 的某些方法绑定到 workergroup 上，方便直接通过 workergroup 进行调用

```python
# 简化后代码，原始代码位于 verl/single_controller/ray/base.py
class RayWorkerGroup(WorkerGroup):
    def __init__(self,
                 resource_pool: RayResourcePool = None,
                 ray_cls_with_init: RayClassWithInitArgs = None,
                 bin_pack: bool = True,
                 name_prefix: str = None,
                 detached=False,
                 worker_names=None,
                 **kwargs) -> None:
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_cls_with_init = ray_cls_with_init
        ...
        # 基于 resource_pool 信息，启动 worker
        self._init_with_resource_pool(resource_pool=resource_pool,
                                      ray_cls_with_init=ray_cls_with_init,
                                      bin_pack=bin_pack,
                                      detached=detached)
        # ray_cls_with_init.cls 中的某些方法绑定到 RayWorkerGroup 上
        self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)
```

### 1. 启动 Workers：`_init_with_resource_pool`

```python
# 简化后代码，原始代码位于 verl/single_controller/ray/base.py
def _init_with_resource_pool(self, resource_pool, ray_cls_with_init):
    # max_collocate_count 意味着单个 GPU 上至多有几个 worker
    num_gpus = 1 / resource_pool.max_collocate_count
    # resource_pool.store: [4, 4]
    for pg_idx, local_world_size in enumerate(resource_pool.store):
        pg = pgs[pg_idx]  # 当前的 PlacementGroup
        for local_rank in range(local_world_size):
            # 1. 构造 worker 需要的配置信息，包括环境变量等
            # 2. 通过 ray_cls_with_init.update_options 更新这些配置信息
            # 启动 worker
            worker = ray_cls_with_init(placement_group=pg,
                                       placement_group_bundle_idx=local_rank,
                                       num_gpus=num_gpus)
            self._workers.append(worker)
            self._worker_names.append(name)
```

### 2. 异步执行：`execute_all_async`

`_init_with_resource_pool` 后，`self._workers` 中保存着所有 worker。当需要调用所有 worker 某个方法时，可以直接使用 `execute_all_async`。

假设有 4 个 worker，需要每个 worker 都执行方法 `add`，参数为 10：

```python
workergroup.execute_all_async("add", x=10)
# 或者
workergroup.execute_all_async("add", x=[10, 10, 10, 10])
```

但用 `execute_all_async` 调用 worker 方法并不自然，所以利用装饰器 `register` 和 `_bind_worker_method` 让调用更自然。

### 3. worker 方法绑定至 workergroup：`_bind_worker_method`

`_bind_worker_method` 来自基类 `WorkerGroup`，参数包含 `user_defined_cls` 和 `func_generator`，其中 `user_defined_cls` 就是用户自定义的 worker 类。

#### 3.1 func_generator

```python
# 原始代码位于 verl/single_controller/ray/base.py
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):

    def func(*args, **kwargs):
        args, kwargs = dispatch_fn(self, *args, **kwargs)
        output = execute_fn(method_name, *args, **kwargs)
        if blocking:
            output = ray.get(output)
        output = collect_fn(self, output)
        return output

    return func
```

`func_generator` 的功能是构造一个函数。`method_name` 实际就是自定义 worker 中使用装饰器 `register` 的函数。构造的新函数会：
1. 先对参数调用 `dispatch_fn`
2. 然后执行 `method_name` 方法
3. 最后用 `collect_fn` 产生最终输出

#### 3.2 `_bind_worker_method` 源码

```python
# 简化代码，原始代码位于 verl/single_controller/base/worker_group.py
def _bind_worker_method(self, user_defined_cls, func_generator):
    for method_name in dir(user_defined_cls):  # 遍历类的所有方法
        if hasattr(method, MAGIC_ATTR):  # 寻找使用装饰器 `register` 的方法
            # 获取 dispatch_fn、collect_fn 和 execut_fn
            dispatch_mode = attribute['dispatch_mode']
            execute_mode = attribute['execute_mode']
            blocking = attribute['blocking']

            fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
            dispatch_fn = fn['dispatch_fn']
            collect_fn = fn['collect_fn']
            execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
            wg_execute_fn_name = execute_mode['execute_fn_name']
            execute_fn = getattr(self, wg_execute_fn_name)

            # 利用 func_generator 将 dispatch_fn、collect_fn 组装到 method_name 上
            func = func_generator(self,
                                  method_name,
                                  dispatch_fn=dispatch_fn,
                                  collect_fn=collect_fn,
                                  execute_fn=execute_fn,
                                  blocking=blocking)
            # func 绑定到 workgroup 上，func 名字仍保持为 method_name
            setattr(self, method_name, func)
```

## 参考资料

- [verl 官方文档 - Hybrid Flow](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
- [veRL GitHub 仓库](https://github.com/volcengine/verl)
- [VeRL(v0.2.0.post2)系列文章](https://zhuanlan.zhihu.com/p/1918280771946382774)

## 相关笔记

- [[Engineering/coding/verl/ray前置知识]]
- [[Engineering/coding/verl/verl总体概览]]
- [[Engineering/coding/verl/init_workers详解]]
- [[Engineering/coding/verl/veRL_colocate实现源码解析]]

