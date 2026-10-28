# 自创容器

# 构建

通过镜像源拉去镜像

```SQL
docker pull docker.m.daocloud.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.
04
```

创建Dockerfile，基于拉去的镜像创建自定义的镜像

```SQL
mkdir ~/docker-projects/gpu-experiment/
cd ~/docker-projects/gpu-experiment/
vim Dockerfile
```

```Markdown
# 1. 基础镜像：CUDA 11.8 + CuDNN 8 + 开发工具（devel）+ Ubuntu 22.04
# 保留devel版本以支持后续可能的编译需求，但仅包含最核心组件
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. 关闭交互模式，避免安装时弹窗
ENV DEBIAN_FRONTEND=noninteractive

# 3. 安装基础系统工具并添加sudo支持
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    openssh-client \
    # 基础shell工具
    bash curl wget ca-certificates \
    # 文本编辑工具
    vim nano \
    # 版本控制工具（后续拉代码用）
    git \
    # 基础编译工具（devel已包含大部分，这里补充必要的）
    build-essential \
    # 添加sudo支持
    sudo \
    # 清理缓存，保持镜像精简
    && rm -rf /var/lib/apt/lists/*

# 4. 配置用户和权限（与主机UID/GID匹配）
# 创建与主机相同GID的用户组
# 红色的部分需要根据自己主机的UID/GID进行修改。查看方式：在终端输入 id 命令即可。
RUN groupadd -g gid username && \
    # 创建与主机相同UID的用户并加入组
    useradd -u uid -g gid -m -s /bin/bash username && \
    # 允许该用户使用sudo且无需密码
    echo "username ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 5. 配置CUDA环境变量（确保后续在容器内编译时能找到CUDA）
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 6. 创建工作目录并设置权限
RUN mkdir -p /workspace && \
    chown -R username:username /workspace
WORKDIR /workspace

# 7. 切换到普通用户（与主机用户匹配）
USER username

# 8. 启动容器时直接进入bash，等待手动配置环境
CMD ["/bin/bash"]
```

给拉取的镜像打标签（与 Dockerfile 中的 FROM 一致）： 因为 Dockerfile 中写的是 `nvidia/cuda:xxx`，需要将手动拉取的镜像重命名为该标签，否则构建时仍会去官方源拉取

```SQL
docker tag docker.m.daocloud.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

构建本地镜像

```SQL
docker build -t gpu-experiment:latest .
```

# 使用

如果计划**长期在容器内进行实验**（需要保留环境配置、代码、实验数据，避免每次启动都重新搭建），则需要调整容器启动参数，核心是**去掉** **`--rm`** **并增加“持久化配置”**，确保容器停止后不被删除，且数据和环境能复用。以下是详细的参数配置和操作指南：

### 一、核心参数调整（针对长期使用）

首先明确长期使用容器的关键需求：**容器不自动删除、数据不丢失、环境可复用**。推荐的启动命令如下，我们逐参数解释：

```Bash
docker run -it \
  --name gpu_exp_container \  # 给容器起固定名称（核心：方便后续找到/启动容器）
  --gpus all \                # 启用所有GPU（必选，实验需要GPU）
  -v /home/lbwang/sxliu:/workspace \  # 挂载宿主机目录（核心：持久化数据）
  -p 8888:8888 \              # 端口映射（可选：若用Jupyter/远程调试）
  --shm-size=32g \            # 增大共享内存（可选：避免数据加载时内存不足）
  --restart unless-stopped \  # 自动重启（可选：服务器重启后容器自动启动）
  gpu-experiment:latest       # 镜像名（你的GPU实验镜像）
```

### 二、关键参数详解（针对长期使用）

#### 1. 去掉 `--rm`：保留容器不自动删除

- **原** **`--rm`** 问题**：容器停止后会被自动删除，之前配置的环境（如安装的Python库、编译的工具）全部丢失。
    
- **去掉后的效果**：容器停止后仍保留在系统中（状态为 `Exited`），下次可直接启动，环境完全复用。
    

#### 2. `--name gpu_exp_container`：给容器固定名称

- **作用**：用自定义名称（如 `gpu_exp_container`）标识容器，避免Docker自动生成随机名称（如 `nostalgic_turing`），方便后续操作（启动、进入、停止）。
    
- **后续操作示例**：
    
    - 启动已停止的容器：`docker start gpu_exp_container`
        
    - 进入运行中的容器：`docker exec -it gpu_exp_container /bin/bash`
        
    - 停止容器：`docker stop gpu_exp_container`
        

#### 3. `-v /home/lbwang/sxliu:/workspace`：挂载宿主机目录（数据持久化核心）

- **作用**：将宿主机的 `~/sxliu` 目录（你的实验目录）与容器内的 `/workspace` 目录**双向绑定**，实现：
    
    - **数据不丢失**：容器内 `/workspace` 下的代码、实验结果、日志会同步到宿主机 `~/sxliu`，即使容器被删除，数据仍在宿主机。
        
    - **跨容器复用**：若后续重建容器，重新挂载该目录即可恢复所有数据。
        
    - **宿主机编辑方便**：可在宿主机用熟悉的工具（如VS Code）编辑代码，容器内直接运行（无需在容器内用 `vim` 编辑）。
        
- **注意**：替换 `/home/lbwang/sxliu` 为你宿主机实际的实验目录路径（用 `pwd` 命令可查看当前目录路径）。
    

#### 4. 可选但推荐的参数

- **`-p 8888:8888`****：端口映射**若需要在容器内启动Jupyter Notebook/Lab（可视化调试代码），通过该参数将容器的8888端口映射到宿主机的8888端口，之后在宿主机浏览器访问 `http://localhost:8888` 即可进入Jupyter。
    
- **`--shm-size=32g`****：增大共享内存**深度学习实验中，数据加载（如用 `DataLoader`）常依赖共享内存，默认共享内存较小（约64MB），可能导致“内存不足”错误。根据你的GPU内存配置（如16G/32G/64G）设置，建议设为GPU内存的2倍（如32G）。
    
- **`--restart unless-stopped`****：自动重启容器**若服务器意外重启（如断电、维护），该参数会让容器在服务器启动后自动恢复运行，避免手动重新启动。
    

### 三、长期使用容器的核心操作流程

#### 1. 首次启动容器（用上述命令）

```Bash
docker run -it \
  --name gpu_container_cuda118 \
  --gpus all \
  -v /data/lbwang:/data/lbwang \
  -v /home/lbwang:/home/lbwang \
  -p 8888:8888 \
  --shm-size=32g \
  --restart unless-stopped \
  gpu-experiment:latest
```

进入容器后，即可开始配置环境（如安装Python、深度学习库、下载数据集等），所有操作会保存在容器内，且 `/workspace` 目录的数据会同步到宿主机。

#### 2. 日常使用：启动/进入容器

- **若容器已停止，先启动**：
    
    ```Bash
    docker start gpu_container_cuda118  # 启动容器（后台运行）
    ```
    
- **进入已启动的容器**（核心操作，每次用容器都需执行）：
    
    ```Bash
    docker exec -it gpu_container_cuda118 /bin/bash
    ```
    
    - `exec`：在运行的容器内执行命令；
        
    - `-it`：交互式终端（允许输入命令、查看输出）；
        
    - 执行后会进入容器的 `/workspace` 目录，可直接运行实验代码。
        

#### 3. 停止容器（不删除，下次可启动）

```Bash
docker stop gpu_container_cuda118  # 停止容器，环境和数据全部保留
```

  

  

#### 4. 备份容器（可选，防止容器损坏）

若配置了复杂的实验环境，可将容器保存为新镜像（相当于“快照”），即使原容器损坏，也能从新镜像重建：

```Bash
# 将运行中的容器保存为新镜像（镜像名：gpu_exp_backup，版本：v1）
docker commit gpu_container_cuda118 gpu_exp_backup:v1
```

后续重建容器时，用新镜像启动：

```Bash
docker run -it --name gpu_exp_new --gpus all gpu_exp_backup:v1
```

  

  

### 四、注意事项

1. **容器命名唯一性**：若提示 `Error response from daemon: Conflict. The container name "/gpu_exp_container" is already in use`，说明该名称已被占用，可换个名称（如 `gpu_exp_container_v2`），或删除旧容器（`docker rm 旧容器名`，需先停止旧容器）。
    
2. **数据备份**：虽然 `-v` 挂载实现了数据同步，但仍建议定期在宿主机备份 `/home/lbwang/sxliu` 目录（如用 `tar` 压缩），防止宿主机磁盘故障导致数据丢失。
    
3. **环境更新**：若后续需要安装新依赖（如新增Python库），直接在容器内用 `pip install` 即可，安装结果会保留在容器内，下次启动仍有效。
    
4. 如果需要root权限，使用`sudo su -`命令。
    

通过以上配置，你可以在容器内长期稳定地进行实验，既隔离了宿主机环境，又能确保环境和数据的持久化，无需每次重新搭建。
