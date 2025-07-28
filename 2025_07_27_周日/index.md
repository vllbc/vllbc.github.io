# 

## 论文阅读情况
```dataview
Table file.mtime as "最后修改时间"
WHERE tags[0] = "Reading" and this.file.ctime - file.mtime <= dur(7 days)
Sort file.mtime desc
```
## PlAN-AND-ACT
[PLAN-AND-ACT：Improving Planning of Agents for Long-Horizon Tasks](../reading/Planning/PLAN-AND-ACT：Improving%20Planning%20of%20Agents%20for%20Long-Horizon%20Tasks.md)
这篇论文解答了我心中一直存在的一个想法，就是计划和行动是分离的。在agent的执行过程中，要先制定计划，然后按照计划一步步执行，计划并不是可以执行的动作，因此需要一个转换。即分离规划器和执行器。

此外这篇论文还提出了一种数据合成的方法：
1.  **行动轨迹生成（Action Trajectory Generation）**：首先，利用一个强大的“教师 LLM”（如 GPT-4 o），在真实或模拟环境（如 WebArena）中执行任务，收集成功的“行动轨迹”（即一系列成功的底层操作序列）。
2.  **接地规划生成（Grounded Plan Generation）**：接着，让另一个“教师 LLM”扮演“事后诸葛亮”的角色，对上一步收集到的成功行动轨迹进行“逆向工程”。它会分析这些具体的行动序列，反推出一个合乎逻辑的高层计划，并将每个计划步骤与具体的行动序列进行关联。这一步至关重要，因为它确保了生成的计划是**“接地的”（Grounded）**，即与真实世界的可行操作紧密相连，而非凭空想象。
3.  **合成计划扩展（Synthetic Plan Expansion）**：最后，将上一步生成的“计划-行动”对作为“种子”，利用 LLM 强大的泛化能力，生成大量结构相似但内容多样的新查询和计划。例如，从一个关于“查找商品”的计划，可以衍生出关于“查找不同商品”、“比较价格”等多种新计划。

此文还提出了动态重规划，也就是发生意外时（和我的第一个科研点很像）会重新调整原来的计划，而不是计划一成不变，这样确保了一定的泛化性能。


这篇论文验证了我一直以来的想法，就是计划和行动分离，计划和行动之间是有gap的。在这基础上可以做的有：
1. 
## Routine

[Routine：A Structural Planning Framework for LLM Agent System in Enterprise](../reading/Planning/Routine：A%20Structural%20Planning%20Framework%20for%20LLM%20Agent%20System%20in%20Enterprise.md)
这篇论文沿用了PLAN-AND-ACT的想法，将行动和规划分离，与上面论文的区别在于：
>与之前依赖模型在运行时进行“黑盒”规划的**Plan-and-Act**模式相比，Routine将“Plan”过程前置和显式化，使得整个工作流变得**透明、可调试、可维护**。开发者可以轻易地修改、增加或删除Routine中的步骤。

此外最大的创新就是双重内存模块（Memory Module）。
*   **程序内存（Procedure Memory）**：用于存储和检索整个Routine库。当用户提出请求时，系统能快速匹配到最合适的Routine。
*   **变量内存（Variable Memory）**：当工具调用返回过长的结果（如一整篇文章）时，系统会将其存入变量内存，并只在上下文中保留一个简短的键（如`VAR_AI_ETHICS_TEXT_001`）。这极大地压缩了上下文长度，降低了token消耗，并减少了长文本可能导致的模型“分心”或语法错误。

从论文的图中也可以看到，routine生成的计划不是纯粹的自然语言，而是一种结构化文本，还告知了使用哪个工具，这减小了从计划到行动的gap。


## 笔记整理情况
未完成笔记：
```dataview
Table rows.file.link as filename
WHERE todo and title != "2025_07_27_周日"
Sort file.ctime desc
GROUP BY tags[1] as "category"
```

已完成笔记：
```dataview
Table file.mtime as "最后修改时间"
WHERE !todo and this.file.ctime - file.mtime <= dur(7 days) and tags[0] != "Reading" and title
Sort file.mtime desc
```

## 科研情况

- 准备复现Text-Based World Simulators作为下一篇文章的一个测试集

## 其它学习情况

- 为verl中的agent_loop修复了一个bug并提交了pr。
- 复现了zerosearch，但没有跑完
- 更深入阅读了verl的源码，接下来整理关于device mesh的内容，下周需要学习完fsdp和zero，这是继续深入阅读的基础。
- 准备培养写周报的习惯，并且利用dataview高效整理一周做了什么事情。
