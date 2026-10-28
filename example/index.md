# example

```yaml
[project]
name = "rl-train"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "codetiming>=1.4.0",
    "datasets>=4.2.0",
    "flash-attn==2.7.4.post1",
    "gpustat>=1.1.1",
    "hydra-core[all]>=1.3.2",
    "ninja>=1.13.0",
    "peft>=0.17.1",
    "python-dotenv[cli]>=1.1.1",
    "ray[default]>=2.50.0",
    "requests>=2.32.5",
    "setuptools>=80.9.0",
    "tensordict>=0.10.0",
    "torch>=2.6.0",
    "torchaudio>=2.6.0",
    "torchdata>=0.11.0",
    "torchvision>=0.21.0",
    "transformers>=4.57.1",
    "wandb>=0.22.2",
]

[[tool.uv.index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
default = true

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cu124", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu124", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

torchaudio = [
  { index = "pytorch-cu124", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[tool.uv]
no-build-isolation-package = ["flash-attn"]
```
