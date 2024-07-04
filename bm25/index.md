# BM25



# BM25算法

BM25算法，通常用来作搜索相关性平分。一句话概况其主要思想：对Query进行语素解析，生成语素qi；然后，对于每个搜索结果D，计算每个语素qi与D的相关性得分，最后，将qi相对于D的相关性得分进行加权求和，从而得到Query与D的相关性得分。

  

## 原理

BM25一般公式如下：

$$
score(Q,d) = \sum_{i}^nW_iR(q_i, d)
$$
其中Q表示为Query，$q_i$表示Q解析后的一个语素(对中文而言，我们可以把对Query的分词作为语素分析，每个词看成语素)。d表示一个搜索结果文档，$W_i$表示语素$q_i$的权重，$R(q_i, d)$表示语素$q_i$与文档d的相关性得分。

## 权重
下面我们来看如何定义Wi。判断一个词与一个文档的相关性的权重，方法有多种，较常用的是IDF。这里以IDF为例，公式如下：

$$
IDF(q_i) = \log \frac{N-n(q_i)+0.5}{n(q_i)+0.5}
$$

其中，N为索引中的全部文档数，n(qi)为包含了qi的文档数。

根据IDF的定义可以看出，对于给定的文档集合，包含了qi的文档数越多，qi的权重则越低。也就是说，当很多文档都包含了qi时，qi的区分度就不高，因此使用qi来判断相关性时的重要度就较低。

## 相关性得分
下面定义相关性得分$R(q_i,d)$

首先来看BM25中相关性得分的一般形式：

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020220817131807.png)

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020220817131811.png)

其中$k_1, k_2, b$为调节因子，根据经验设置，一般$k_1=2,b=0.75$ ，$f_i$为$q_i$在文档d中出现的频率，$qf_i$为$q_i$在Query中出现的频率，dl为文档d的长度，avgdl为所有文档的平均长度，绝大部分情况下$q_i$在Query中只会出现一次，因此
$qf_i=1$，所以可以简化为：
![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020220817132154.png)

从K的定义中可以看到，参数b的作用是调整文档长度对相关性影响的大小。b越大，文档长度的对相关性得分的影响越大，反之越小。而文档的相对长度越长，K值将越大，则相关性得分会越小。这可以理解为，当文档较长时，包含qi的机会越大，因此，同等fi的情况下，长文档与qi的相关性应该比短文档与qi的相关性弱。

综上，BM25算法的相关性得分公式可总结为：

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020220817132305.png)

从BM25的公式可以看到，通过使用不同的语素分析方法、语素权重判定方法，以及语素与文档的相关性判定方法，我们可以衍生出不同的搜索相关性得分计算方法，这就为我们设计算法提供了较大的灵活性。

## 代码
```python
import math
import jieba
import re

text = '''

自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。

它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。

自然语言处理是一门融语言学、计算机科学、数学于一体的科学。

因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，

所以它与语言学的研究有着密切的联系，但又有重要的区别。

自然语言处理并不是一般地研究自然语言，

而在于研制能有效地实现自然语言通信的计算机系统，

特别是其中的软件系统。因而它是计算机科学的一部分。

'''

  

class BM25(object):

  

    def __init__(self, docs):

        self.D = len(docs) # doc个数

        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D # 每篇平均长度

        self.docs = docs

        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数

        self.df = {} # 存储每个词及出现了该词的文档数量(count)

        self.idf = {} # 存储每个词的idf值，当作权重

        self.k1 = 1.5

        self.b = 0.75

        self.init()

  

    def init(self):

        for doc in self.docs:

            tmp = {}

            for word in doc:

                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数（也可以用defaultdict)

            self.f.append(tmp) # idx为索引，f[idx]为一个dict，dict存储着第idx+1个文档中每个词的出现次数,idx代表第几个文档。

            for k in tmp.keys(): # 如果词k出现在了当前文档中，则df[k]即文档数量加1

                self.df[k] = self.df.get(k, 0) + 1

        for k, v in self.df.items():

            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5) # 计算idf

  

    def sim(self, query, index): # 与单个文档的相似度

        score = 0

        for word in query:

            if word not in self.f[index]:

                continue

            d = len(self.docs[index]) # 当前文档的长度

            score += (self.idf[word]*(self.f[index][word]/d)*(self.k1+1)

                      / ((self.f[index][word]/d)+self.k1*(1-self.b+self.b*d

                                                      / self.avgdl)))

        return score

  

    def simall(self, query):

        scores = []

        for index in range(self.D):

            score = self.sim(query, index)

            scores.append(score)

        return scores

def get_sentences(doc):

    line_break = re.compile('[\r\n]') # 以换行符分割

    delimiter = re.compile('[，。？！；]') # 以中文标点符号分割

    sentences = []

    for line in line_break.split(doc):

        line = line.strip()

        if not line:

            continue

        for sent in delimiter.split(line):

            sent = sent.strip()

            if not sent:

                continue

            sentences.append(sent)

    return sentences

  

if __name__ == '__main__':

    sents = get_sentences(text)

    print(sents)

    doc = []

    for sent in sents:

        words = list(jieba.cut(sent))

        doc.append(words)

    # print(doc)

    s = BM25(doc)

    # print(s.f)

    # print(s.df)

    # print(s.idf)

    print(s.simall(['自然语言', '计算机科学', '领域', '人工智能', '领域']))
```
