# SimCSE

# 无监督
## info Noise Contrastive Estimation loss
# 有监督

# 复现代码
只贴最核心的损失函数代码

```python
def simcse_unsup_loss(y_pred, device, temp=0.05):

    """无监督的损失函数

    y_pred (tensor): bert的输出, [batch_size * 2, 768] ,2为句子个数，即一个句子对

  

    """

    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]

    y_true = torch.arange(y_pred.shape[0], device=device)

    y_true = (y_true - y_true % 2 * 2) + 1

    # batch内两两计算相似度, 得到相似度矩阵(batch_size*batch_size)

    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)

    print(sim)

    print(sim.shape)

    # 将相似度矩阵对角线置为很小的值, 消除自身的影响

    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12

    # 相似度矩阵除以温度系数

    sim = sim / temp

    # 计算相似度矩阵与y_true的交叉熵损失

    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低

    loss = F.cross_entropy(sim, y_true)

    return torch.mean(loss)

    """

    苏神keras源码

    def simcse_loss(y_true, y_pred):

        idxs = K.arange(0, K.shape(y_pred)[0]) #生成batch内句子的编码 [0,1,2,3,4,5]为例子

        idxs_1 = idxs[None, :] # 给idxs添加一个维度，变成： [[0,1,2,3,4,5]]

        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None] # 这个意思就是说，如果一个句子id为奇数，那么和它同义的句子的id就是它的上一句，如果一个句子id为偶数，那么和它同义的句子的id就是它的下一句。 [:, None] 是在列上添加一个维度。初步生成了label。[[1], [0], [3], [2], [5], [4]]

        y_true = K.equal(idxs_1, idxs_2) # equal会让idxs1和idxs2都映射到6*6,idxs1垂直，idxs2水平

        y_true = K.cast(y_true, K.floatx()) # 生成label

        y_pred = K.l2_normalize(y_pred, axis=1) # 对句向量各个维度做了一个L2正则，使其变得各项同性，避免下面计算相似度时，某一个维度影响力过大。

        similarities = K.dot(y_pred, K.transpose(y_pred)) # 计算batch内每句话和其他句子的内积相似度。

        similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12 # 将和自身的相似度变为0(后面的softmax之后)。

        similarities = similarities * 20 # 将所有相似度乘以20，这个目的是想计算softmax概率时，更加有区分度。

        loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)

        return K.mean(loss)

    """

  

def simcse_sup_loss(y_pred, device, lamda=0.05):

    """

    有监督损失函数

    """

    similarities = F.cosine_similarity(y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=2)

    row = torch.arange(0, y_pred.shape[0], 3)

    col = torch.arange(0, y_pred.shape[0])

    col = col[col % 3 != 0]

  

    similarities = similarities[row, :]

    similarities = similarities[:, col]

    similarities = similarities / lamda

  

    y_true = torch.arange(0, len(col), 2, device=device)

    loss = F.cross_entropy(similarities, y_true)

    return loss
```

# 参考文献

1. [SIMCSE算法源码分析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/483453992)
2. [SimCSE论文及源码解读 | Swift's Blog (transformerswsz.github.io)](https://transformerswsz.github.io/2022/05/01/SimCSE%E8%AE%BA%E6%96%87%E5%8F%8A%E6%BA%90%E7%A0%81%E8%A7%A3%E8%AF%BB/)
