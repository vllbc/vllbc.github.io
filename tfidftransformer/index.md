# TfidfTransformer


**TfidfTransformer()**
**输入：词频TF**
**输出：词频逆反文档频率TF-IDF**（即词频TF与逆反文档频率IDF的乘积，IDF的标准计算公式为 ：idf=log[n/(1+df)]，其中n为文档总数，df为含有所计算单词的文档数量，df越小，idf值越大，也就是说出现频率越小的单词意义越大）

1.计算词频TF，通过函数CountVectorizer()来完成，以该文档为输入，并得到词频 tf 输出；
2.计算词频逆反文档频率TF-IDF，通过函数TfidfTransformer()来实现，以第一步的词频 tf 输出为输入，并得到 tf-idf 格式的输出。

```python
cv = CountVectorizer()
tfidf = TfidfTransformer().fit_transform(cv.fit_transform(corpus))
```



将两步简化为一步就是函数 TfidfVectorizer() 所实现的功能了

**TfidfVectorizer()**
**输入：文档**
**输出：该文档的词频逆反文档频率TF-IDF**

**TfidfVectorizer()**.fit_transform(corpus) = **TfidfTransformer()**.fit_transform(**CountVectorizer()**.fit_transform(corpus))


