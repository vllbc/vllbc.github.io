# agent概览

根据Anthropic的定义，agent定义如下：

>At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:
**Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
**Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

简单来说就是Workflows是预先定义好的一个路径，而Agents是让其自主完成各种流程，而无需预先定义。

## workflows

### Prompt chaining
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250728181523.png)

### Routing
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250728181536.png)

### Parallelization
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250728181544.png)

### Orchestrator-workers

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250728181559.png)

### Evaluator-optimizer
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250728181607.png)

最近热门的Gemini 2.5 Pro Capable of Winning Gold at IMO 2025论文就是使用的这个工作流。

## agents

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250728161628.png)

## 参考

[Agents](https://huyenchip.com//2025/01/07/agents.html)

[Zero to One: Learning Agentic Patterns](https://www.philschmid.de/agentic-pattern)

[Building Effective AI Agents \\ Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

[How we built our multi-agent research system \\ Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
