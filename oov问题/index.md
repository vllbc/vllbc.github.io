# OOV问题


面试中经常被问到的一个问题就是out of vocabulary，可能是因为当前数据集中出现了提前准备好的单词表中没有的word，也可能是因为test中出现了train中没有的word。
## 解决办法：
1. 直接Ignore
2. 将token分配为[unk]
3. 增大词表
4. 检查拼写
5. BPE算法或word piece
面试时可以展开说一下具体的算法过程，不再赘述。
