# MCTS和PRM


## 核心总结
- **PRM和MCTS实际上是两种可以独立使用的技术，只不过，往往它们组合使用时往往能产生1+1>2的效果**。例如，
	- 单独使用PRM：我们可以让模型对同一个prompt采样多个不同solution，无需MCTS，只需利用模型的temperature等随机参数让每次生成结果不同，然后用PRM对每个solution的每一步打分，最终选择分数最高的路径返回。
	- 单独使用MCTS：使用MCTS生成多个解题路径时，不一定要用PRM来决定哪个节点值得扩展，可以用外部大模型（如GPT-4）来选择，也可以用模型自身的perplexity来判断。本质上，我们需要的是找到最值得扩展的节点，PRM只是挑选的众多方法之一。

- **PRM 和 MCTS 既可以应用于优化训练数据，也可以用来预测用**
	- 用于得到高质量训练数据：如rStar论文中，可以用PRM和MCTS的方式来迭代地筛选得到质量更好的思维链SFT数据或者RLHF数据，还可以生成更精确的reward model训练数据。
	- 用于推理：很简单，推理用MCTS的方式把 test-scaling 做上来，再结合PRM的方式从众多路径中挑选最佳答案。

- **PRM和MCTS的缺点**  
    这方面 DeepSeek-R1和 kimi1.5的论文已经说得很情况了。
- Process Reward Model(PRM) 在实际应用中有三大局限：
	- 第一，难以清晰界定一般推理中的细粒度步骤，说白了，怎么定义什么为一个步骤。
	- 第二，判断当前步骤的正误难度大，模型自动化标注不如人意，人工标注又难以拓展。
	- 第三，引入基于模型的PRM易致reward hacking，有时为了训练 policy model，但反而更多时间去优化 reward model 去了。

- 对MCTS的看法：
	- 文本的生成搜索空间指数级增长，为应对，给节点设扩展上限，却容易让模型陷入局部最优解困境。
	- MCTS往往要结合一个精确的PRM来用才能发挥最大效果，但PRM又有上述的问题，陷入一个死循环。
## 参考
https://zhuanlan.zhihu.com/p/27278317894
rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
