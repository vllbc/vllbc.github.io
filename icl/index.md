# ICL

ICL即In-contexting Learning。
ICL 包含三种分类：
- Few-shot learning，允许输入数条示例和一则任务说明；
- One-shot learning，只允许输入一条示例和一则任务说明；
- Zero-shot learning，不允许输入任何示例，只允许输入一则任务说明。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20240411115552.png)

# 训练时

# 推理时
与微调不同，使用ICL推理时并不更新参数。可以简单理解为带有Tuning的都会更新参数，因为叫“微调”，而ICL并不更新参数，但训练时的ICL会更新参数，这显而易见。

# 参考
[In-context Learning学习笔记 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/625116295)
