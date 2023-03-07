# Attention



# Seq2Seq中的Attention

## 缺陷
在seq2seq这篇文章中详细介绍了seq2seq模型的细节，但是仅仅用一个语义编码c是完全不能够表示编码器的输入的，源的可能含义的数量是无限的。当编码器被迫将所有信息放入单个向量中时，它很可能会忘记一些东西。

不仅编码器很难将所有信息放入一个向量中——这对解码器来说也很困难。解码器只看到源的一种表示。但是，在每个生成步骤中，源的不同部分可能比其他部分更有用。但在目前的情况下，解码器必须从相同的固定表示中提取相关信息——这不是一件容易的事。

![](image/Pasted%20image%2020220727180013.png)

这个时候就需要引入注意力机制了，注意这里的注意力机制和transformer中的self-attention是不一样的。下面详细介绍一下。注意几个名词：注意力得分、注意力权重。其中注意力得分即score的计算有多种方法，权重就是对得分进行softmax归一化。

## attention
注意机制是神经网络的一部分。在每个解码器步骤中，它决定哪些源部分更重要。在此设置中，编码器不必将整个源压缩为单个向量 - 它为所有源标记提供表示（例如，所有 RNN 状态而不是最后一个）。


![](image/Pasted%20image%2020221111180414.png)
步骤：
- 接受注意输入：解码器状态$h_t$以及所有编码器状态$s_1,s_2,\dots,s_m$
- 计算每个编码器状态的注意力分数$s_k$，注意力分数表示它对解码器状态$h_t$的相关性，使用注意力函数，接收一个解码器状态和一个编码器状态并返回一个标量分数，即图中的$score(h_t,s_k)$
- 计算注意力权重：即概率分布- 使用Softmax函数
- 计算注意力输出：具有注意力机制的编码器状态的加权和
![](image/Pasted%20image%2020221111180444.png)

即为如图所示内容为如何计算注意力。

注意我们提到的注意力函数，这里的注意力分数的计算有很多种方法，下面介绍几种比较常见的办法：
- 点积： 最简单的办法。
- 双线性函数
- 多层感知机
![](image/Pasted%20image%2020221111180511.png)

注意后两者都有要优化的参数的，第一个点积是直接运算，因此很简单。
在应用时可以直接将注意力的结果传输到最后的softmax，也可以将原始的$h_t$合并，下面介绍几种变体。

## Bahdanau Model
![](image/Pasted%20image%2020221111180459.png)
- 编码器使用双向的RNN
- 利用上一时刻的隐层状态计算注意力输出c，然后和隐层状态一起作为当前时刻的输入，再得到结果$\hat{y}$。这里再说一下训练的过程中当前步的输入使用的是真实的$y$，测试的时候才会使用上一步的输出作为输入。可以将上下文向量c（也就是注意力输出）与$x$拼接后作为输入。
- 注意力得分使用的是感知机。
这里引用一下李沐大佬的代码：
```python
class Seq2SeqAttentionDecoder(AttentionDecoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,

                 dropout=0, **kwargs):

        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)

        self.attention = d2l.AdditiveAttention(

            num_hiddens, num_hiddens, num_hiddens, dropout)

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.GRU(

            embed_size + num_hiddens, num_hiddens, num_layers,

            dropout=dropout)

        self.dense = nn.Linear(num_hiddens, vocab_size)

  

    def init_state(self, enc_outputs, enc_valid_lens, *args):

        # outputs的形状为(batch_size，num_steps，num_hiddens).

        # hidden_state的形状为(num_layers，batch_size，num_hiddens)

        outputs, hidden_state = enc_outputs

        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

  

    def forward(self, X, state):

        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).

        # hidden_state的形状为(num_layers,batch_size,

        # num_hiddens)

        enc_outputs, hidden_state, enc_valid_lens = state

        # 输出X的形状为(num_steps,batch_size,embed_size)

        X = self.embedding(X).permute(1, 0, 2) # 转换是为了方便后面循环计算。

        outputs, self._attention_weights = [], []

        for x in X:

            # query的形状为(batch_size,1,num_hiddens)

            query = torch.unsqueeze(hidden_state[-1], dim=1) # -1是指在最后一层最后时刻的隐藏状态，作为query

            # context的形状为(batch_size,1,num_hiddens)

            context = self.attention(

                query, enc_outputs, enc_outputs, enc_valid_lens)

            # 在特征维度上连结

            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)

            # 将x变形为(1,batch_size,embed_size+num_hiddens)

            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)

            outputs.append(out)

            self._attention_weights.append(self.attention.attention_weights)

        # 全连接层变换后，outputs的形状为

        # (num_steps,batch_size,vocab_size)

        outputs = self.dense(torch.cat(outputs, dim=0))

        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,

                                          enc_valid_lens]

  

    @property

    def attention_weights(self):

        return self._attention_weights
```
## Luong Model
![](image/Pasted%20image%2020221111180540.png)
这个模型的编码器比较常规，使用当前状态计算注意力输出，然后解码器中将隐层状态与注意力输出做一步结合，这样得到了新的隐层状态，然后再传递得到预测结果。

## 注意力对齐
![](image/Pasted%20image%2020221111180549.png)
可以看到解码器关注的源token。

到此seq2seq中的attention就介绍完毕了，其实还有很多细节，以后遇到了会持续补充。

# Self-attention
移步transformer。
