# TF-IDF



## What?

TF-IDF(term frequency–inverse document frequency)是一种用于信息检索与数据挖掘的常用加权技术，常用于挖掘文章中的关键词，而且算法简单高效，常被工业用于最开始的文本数据清洗。

TF-IDF有两层意思，一层是"词频"（Term Frequency，缩写为TF），另一层是"逆文档频率"（Inverse Document Frequency，缩写为IDF）。

假设我们现在有一片长文叫做《量化系统架构设计》词频高在文章中往往是停用词，“的”，“是”，“了”等，这些在文档中最常见但对结果毫无帮助、需要过滤掉的词，用TF可以统计到这些停用词并把它们过滤。当高频词过滤后就只需考虑剩下的有实际意义的词。

但这样又会遇到了另一个问题，我们可能发现"量化"、"系统"、"架构"这三个词的出现次数一样多。这是不是意味着，作为关键词，它们的重要性是一样的？事实上系统应该在其他文章比较常见，所以在关键词排序上，“量化”和“架构”应该排在“系统”前面，这个时候就需要IDF，IDF会给常见的词较小的权重，它的大小与一个词的常见程度成反比。

**当有TF(词频)和IDF(逆文档频率)后，将这两个词相乘，就能得到一个词的TF-IDF的值。某个词在文章中的TF-IDF越大，那么一般而言这个词在这篇文章的重要性会越高，所以通过计算文章中各个词的TF-IDF，由大到小排序，排在最前面的几个词，就是该文章的关键词。**

## 步骤

第一步，计算词频：


$$
词频(TF) = \frac{某个词在文章中出现次数}{文章的总词数}
$$
第二步，计算逆文档频率：


$$
逆文档频率(IDF) = log(\frac{语料库的文档总数}{包含该词的文档数+1})
$$

> 1.为什么+1？是为了处理分母为0的情况。假如所有的文章都不包含这个词，分子就为0，所以+1是为了防止分母为0的情况。
>
> 2.为什么要用log函数？log函数是单调递增，求log是为了归一化，保证反文档频率不会过大。
>
> 3.会出现负数？肯定不会，分子肯定比分母大。

第三步，计算TF-IDF：


$$
TF-IDF = 词频(TF) \times 逆文档频率(IDF)
$$

## **优缺点**

TF-IDF的优点是简单快速，而且容易理解。缺点是有时候用**词频**来衡量文章中的一个词的重要性不够全面，有时候重要的词出现的可能不够多，而且这种计算无法体现位置信息，无法体现词在上下文的重要性。如果要体现词的上下文结构，那么你可能需要使用word2vec算法来支持。

## 代码

```python
corpus = ['this is the first document',
            'this is the second second document',
            'and the third one',
            'is this the first document']
words_list = list()
for i in range(len(corpus)):
    words_list.append(corpus[i].split(' '))
```



### 手动实现

```python
def manual():
    
    from collections import Counter
    count_list = list()
    for i in range(len(words_list)):
        count = Counter(words_list[i])
        count_list.append(count)

    import math
    def tf(word, count):
        return count[word] / sum(count.values())


    def idf(word, count_list):
        n_contain = sum([1 for count in count_list if word in count])
        return math.log(len(count_list) / (1 + n_contain))


    def tf_idf(word, count, count_list):
        return tf(word, count) * idf(word, count_list)

    for i, count in enumerate(count_list):
        print("第 {} 个文档 TF-IDF 统计信息".format(i + 1))
        scores = {word : tf_idf(word, count, count_list) for word in count}
        sorted_word = sorted(scores.items(), key = lambda x : x[1], reverse=True)
        for word, score in sorted_word:
            print("\tword: {}, TF-IDF: {}".format(word, round(score, 5)))
```

### gensim

```python
def gensim_work():
    from gensim import corpora
    # 赋给语料库中每个词(不重复的词)一个整数id
    dic = corpora.Dictionary(words_list) # 创建词典
    new_corpus = [dic.doc2bow(words) for words in words_list]
    # 元组中第一个元素是词语在词典中对应的id，第二个元素是词语在文档中出现的次数
    from gensim import models
    tfidf = models.TfidfModel(new_corpus)
    tfidf.save("tfidf.model")
    # 载入模型 
    tfidf = models.TfidfModel.load("tfidf.model")
    # 使用这个训练好的模型得到单词的tfidf值
    res = [tfidf[temp] for temp in new_corpus]
    print(res)
```

### sklearn

```python
def sklearn_work():
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    print(tfidf.toarray()) # 权重
    print(vectorizer.get_feature_names()) # 单词
    print(vectorizer.vocabulary_) # 词典
```
