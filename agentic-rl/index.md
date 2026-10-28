# agentic RL

>在一个需要多轮对话完成的长rollout，目前有两种主流做法，第一种是把每一轮的上下文内容给拼接起来。最后形成一条长长的轨迹通过小心的设置mask来让模型可以训练。第二种是把所有的单轮交互切开，直接看做独立的rollout送进batch，每一个步骤都送一个相同的final reward来训练。这两种做法其实都有各自的问题。第一种做法最后一次完整对话会变超级长，非常耗显存。第二种做法的话，模型会明显遇到上下文缺失的问题。

第一种的代表论文为[RAGEN：Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](../../Research/Papers/Agent/RAGEN：Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning.md)，第二种的代表论文为[Group-in-Group Policy Optimization for LLM Agent Training](../../Research/Papers/Planning/Group-in-Group Policy Optimization for LLM Agent Training.md)。
现在的框架基本上都是后者，比如agent-lighnting、verl-agent。目前进入决赛圈的框架有：
- Agent-Lightning
- RLLM
- Verl (toy实现)
- Verl-Agent
前两个较为通用，但不支持MLLM，后两者支持MLLM但一个是toy，另一个只支持特定环境，需要自己实现环境。

## 参考
[# DeepWerewolf 如何搭建一个LLM狼人杀 Agentic RL实时训练系统](https://zhuanlan.zhihu.com/p/1961088836735042597)
