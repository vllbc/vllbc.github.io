# 文本相似度



# Tf-Idf
参考：[https://www.jb51.net/article/142132.htm](https://www.jb51.net/article/142132.htm)

代码：（使用gensim）

```python
import jieba
from gensim import corpora, models, similarities

doc0 = "我不喜欢上海"
doc1 = "上海是一个好地方"
doc2 = "北京是一个好地方"
doc3 = "上海好吃的在哪里"
doc4 = "上海好玩的在哪里"
doc5 = "上海是好地方"
doc6 = "上海路和上海人"
doc7 = "喜欢小吃"
doc_test = "我喜欢上海的小吃"

all_doc = [doc0, doc1, doc2, doc3, doc4, doc5, doc6, doc7]

all_doc_list = []
for doc in all_doc:
    doc_list = list(jieba.cut(doc))
    all_doc_list.append(doc_list)  # 语料库分词

doc_test_list = [word for word in jieba.cut(doc_test)]  # 测试分词
	
dictionary = corpora.Dictionary(all_doc_list)  # 创建词典

corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]  # 创建语料库
doc_test_vec = dictionary.doc2bow(doc_test_list)  # 测试

tfidf = models.TfidfModel(corpus)  # 根据语料库创建tfidf模型

index = similarities.SparseMatrixSimilarity(
    tfidf[corpus], num_features=len(dictionary.keys()))
sim = index[tfidf[doc_test_vec]]
print(dictionary.token2id)

```



**总结一下使用tf-idf计算文本相似度的步骤：**

1、读取文档
2、对要计算的多篇文档进行分词
3、对文档进行整理成指定格式，方便后续进行计算
4、计算出词语的词频
5、【可选】对词频低的词语进行过滤
6、建立语料库词典
7、加载要对比的文档
8、将要对比的文档通过doc2bow转化为词袋模型
9、对词袋模型进行进一步处理，得到新语料库
10、将新语料库通过tfidfmodel进行处理，得到tfidf
11、通过token2id得到特征数
12、稀疏矩阵相似度，从而建立索引
13、得到最终相似度结果
