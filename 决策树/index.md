# 决策树



参考：[https://cuijiahua.com/blog/2017/11/ml_2_decision_tree_1.html](https://cuijiahua.com/blog/2017/11/ml_2_decision_tree_1.html)

《机器学习》周志华

# 决策树

决策树是什么？决策树(decision tree)是一种基本的分类与回归方法。举个通俗易懂的例子，如下图所示的流程图就是一个决策树，长方形代表判断模块(decision block)，椭圆形成代表终止模块(terminating block)，表示已经得出结论，可以终止运行。从判断模块引出的左右箭头称作为分支(branch)，它可以达到另一个判断模块或者终止模块。我们还可以这样理解，分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点(node)和有向边(directed edge)组成。结点有两种类型：内部结点(internal node)和叶结点(leaf node)。内部结点表示一个特征或属性，叶结点表示一个类。蒙圈没？？如下图所示的决策树，长方形和椭圆形都是结点。长方形的结点属于内部结点，椭圆形的结点属于叶结点，从结点引出的左右箭头就是有向边。而最上面的结点就是决策树的根结点(root node)。这样，结点说法就与模块说法对应上了，理解就好。

## 步骤

1.特征选择

特征选择在于选取对训练数据具有分类能力的特征。这样可以提高决策树学习的效率，如果利用一个特征进行分类的结果与随机分类的结果没有很大差别，则称这个特征是没有分类能力的。经验上扔掉这样的特征对决策树学习的精度影响不大。通常特征选择的标准是信息增益(information gain)或信息增益比，为了简单，本文使用信息增益作为选择特征的标准。那么，什么是信息增益？在讲解信息增益之前，让我们看一组实例，贷款申请样本数据表。

![机器学习实战教程（二）：决策树基础篇之让我们从相亲说起](https://cuijiahua.com/wp-content/uploads/2017/11/m_2_2.jpg)

希望通过所给的训练数据学习一个贷款申请的决策树，用于对未来的贷款申请进行分类，即当新的客户提出贷款申请时，根据申请人的特征利用决策树决定是否批准贷款申请。

特征选择就是决定用哪个特征来划分特征空间。比如，我们通过上述数据表得到两个可能的决策树，分别由两个不同特征的根结点构成。

(1).香农熵

在可以评测哪个数据划分方式是最好的数据划分之前，我们必须学习如何计算信息增益。集合信息的度量方式称为香农熵或者简称为熵(entropy)，这个名字来源于信息论之父克劳德·香农。

如果看不明白什么是信息增益和熵，请不要着急，因为他们自诞生的那一天起，就注定会令世人十分费解。克劳德·香农写完信息论之后，约翰·冯·诺依曼建议使用"熵"这个术语，因为大家都不知道它是什么意思。

熵定义为信息的期望值。在信息论与概率统计中，熵是表示随机变量不确定性的度量。如果待分类的事物可能划分在多个分类之中，则符号xi的信息定义为 ：


$$
l(x_i) = -log_{2}p(x_i)
$$
其中p(xi)是选择该分类的概率。有人可能会问，信息为啥这样定义啊？答曰：前辈得出的结论。这就跟1+1等于2一样，记住并且会用即可。上述式中的对数以2为底，也可以e为底(自然对数)。

通过上式，我们可以得到所有类别的信息。为了计算熵，我们需要计算所有类别所有可能值包含的信息期望值(数学期望)，通过下面的公式得到：


$$
H = -\sum_{i=1}^np(x_i)log_2p(x_i)
$$
期中n是分类的数目。熵越大，随机变量的不确定性就越大。

当熵中的概率由数据估计(特别是最大似然估计)得到时，所对应的熵称为经验熵(empirical entropy)。什么叫由数据估计？比如有10个数据，一共有两个类别，A类和B类。其中有7个数据属于A类，则该A类的概率即为十分之七。其中有3个数据属于B类，则该B类的概率即为十分之三。浅显的解释就是，这概率是我们根据数据数出来的。我们定义贷款申请样本数据表中的数据为训练数据集D，则训练数据集D的经验熵为H(D)，|D|表示其样本容量，及样本个数。设有K个类Ck, = 1,2,3,..	.,K,|Ck|为属于类Ck的样本个数，因此经验熵公式就可以写为 ：


$$
H(D) = -\sum_{k=1}^K \frac{|c_k|}{|D|}log_2\frac{|c_k|}{|D|}
$$
根据此公式计算经验熵H(D)，分析贷款申请样本数据表中的数据。最终分类结果只有两类，即放贷和不放贷。根据表中的数据统计可知，在15个数据中，9个数据的结果为放贷，6个数据的结果为不放贷。所以数据集D的经验熵H(D)为：


$$
H(D) = -\frac{9}{15}log_2\frac{9}{15} - \frac{6}{15}log_2\frac{6}{15} = 0.971
$$
(2)信息增益

在上面，我们已经说过，如何选择特征，需要看信息增益。也就是说，信息增益是相对于特征而言的，信息增益越大，特征对最终的分类结果影响也就越大，我们就应该选择对最终分类结果影响最大的那个特征作为我们的分类特征。

在讲解信息增益定义之前，我们还需要明确一个概念，条件熵。

熵我们知道是什么，条件熵又是个什么鬼？条件熵H(Y|X)表示在已知随机变量X的条件下随机变量Y的不确定性，随机变量X给定的条件下随机变量Y的条件熵(conditional entropy)H(Y|X)，定义为X给定条件下Y的条件概率分布的熵对X的数学期望：


$$
H(Y|X) = \sum_{i=1}^np_iH(Y|X = x_i), \\\\
p_i = P(X = x_i)
$$
明确了条件熵和经验条件熵的概念。接下来，让我们说说信息增益。前面也提到了，信息增益是相对于特征而言的。所以，特征A对训练数据集D的信息增益g(D,A)，定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H(D|A)之差，即：


$$
g(D,A) = H(D) - H(D|A)
$$
一般地，熵H(D)与条件熵H(D|A)之差称为互信息(mutual information)。决策树学习中的信息增益等价于训练数据集中类与特征的互信息。

设特征A有n个不同的取值{a1,a2,···,an}，根据特征A的取值将D划分为n个子集{D1,D2，···,Dn}，|Di|为Di的样本个数。记子集Di中属于Ck的样本的集合为Dik，即Dik = Di ∩ Ck，|Dik|为Dik的样本个数。于是经验条件熵的公式可以些为


$$
H(D|A) = \sum_{i=1}^n\frac{|D_i|}{|D|}H(D_i) = -\sum_{i=1}^n\frac{|D_i|}{|D|}\sum_{k=1}^K\frac{|D_{ik}|}{|D_i|}log_2\frac{|D_{ik}|}{|D_i|}
$$


说了这么多概念性的东西，没有听懂也没有关系，举几个例子，再回来看一下概念，就懂了。

以贷款申请样本数据表为例进行说明。看下年龄这一列的数据，也就是特征A1，一共有三个类别，分别是：青年、中年和老年。我们只看年龄是青年的数据，年龄是青年的数据一共有5个，所以年龄是青年的数据在训练数据集出现的概率是十五分之五，也就是三分之一。同理，年龄是中年和老年的数据在训练数据集出现的概率也都是三分之一。现在我们只看年龄是青年的数据的最终得到贷款的概率为五分之二，因为在五个数据中，只有两个数据显示拿到了最终的贷款，同理，年龄是中年和老年的数据最终得到贷款的概率分别为五分之三、五分之四。所以计算年龄的信息增益，过程如下：


![机器学习实战教程（二）：决策树基础篇之让我们从相亲说起](https://cuijiahua.com/wp-content/uploads/2017/11/m_2_13.jpg)

同理，计算其余特征的信息增益g(D,A2)、g(D,A3)和g(D,A4)。分别为：

![机器学习实战教程（二）：决策树基础篇之让我们从相亲说起](https://cuijiahua.com/wp-content/uploads/2017/11/m_2_14_m.jpg)

![机器学习实战教程（二）：决策树基础篇之让我们从相亲说起](https://cuijiahua.com/wp-content/uploads/2017/11/m_2_15.jpg)



最后，比较特征的信息增益，由于特征A3(有自己的房子)的信息增益值最大，所以选择A3作为最优特征。

由于特征A3(有自己的房子)的信息增益值最大，所以选择特征A3作为根结点的特征。它将训练集D划分为两个子集D1(A3取值为"是")和D2(A3取值为"否")。由于D1只有同一类的样本点，所以它成为一个叶结点，结点的类标记为“是”。

对D2则需要从特征A1(年龄)，A2(有工作)和A4(信贷情况)中选择新的特征，计算各个特征的信息增益：

![机器学习实战教程（三）：决策树实战篇之为自己配个隐形眼镜](https://cuijiahua.com/wp-content/uploads/2017/11/ml_3_2-b.jpg)

根据计算，选择信息增益最大的特征A2(有工作)作为结点的特征。由于A2有两个可能取值，从这一结点引出两个子结点：一个对应"是"(有工作)的子结点，包含3个样本，它们属于同一类，所以这是一个叶结点，类标记为"是"；另一个是对应"否"(无工作)的子结点，包含6个样本，它们也属于同一类，所以这也是一个叶结点，类标记为"否"。

这样就生成了一个决策树，该决策树只用了两个特征(有两个内部结点)，生成的决策树如下图所示。

![机器学习实战教程（三）：决策树实战篇之为自己配个隐形眼镜](https://cuijiahua.com/wp-content/uploads/2017/11/ml_3_3.jpg)



## 总结

我们已经学习了从数据集构造决策树算法所需要的子功能模块，包括经验熵的计算和最优特征的选择，其工作原理如下：得到原始数据集，然后基于最好的属性值划分数据集，由于特征值可能多于两个，因此可能存在大于两个分支的数据集划分。第一次划分之后，数据集被向下传递到树的分支的下一个结点。在这个结点上，我们可以再次划分数据。因此我们可以采用递归的原则处理数据集。

构建决策树的算法有很多，比如C4.5、ID3和CART，这些算法在运行时并不总是在每次划分数据分组时都会消耗特征。由于特征数目并不是每次划分数据分组时都减少，因此这些算法在实际使用时可能引起一定的问题。目前我们并不需要考虑这个问题，只需要在算法开始运行前计算列的数目，查看算法是否使用了所有属性即可。

决策树生成算法递归地产生决策树，直到不能继续下去未为止。这样产生的树往往对训练数据的分类很准确，但对未知的测试数据的分类却没有那么准确，即出现过拟合现象。过拟合的原因在于学习时过多地考虑如何提高对训练数据的正确分类，从而构建出过于复杂的决策树。解决这个问题的办法是考虑决策树的复杂度，对已生成的决策树进行简化。

**决策树的一些优点：**

- 易于理解和解释。决策树可以可视化。
- 几乎不需要数据预处理。其他方法经常需要数据标准化，创建虚拟变量和删除缺失值。决策树还不支持缺失值。
- 使用树的花费（例如预测数据）是训练数据点(data points)数量的对数。
- 可以同时处理数值变量和分类变量。其他方法大都适用于分析一种变量的集合。
- 可以处理多值输出变量问题。
- 使用白盒模型。如果一个情况被观察到，使用逻辑判断容易表示这种规则。相反，如果是黑盒模型（例如人工神经网络），结果会非常难解释。
- 即使对真实模型来说，假设无效的情况下，也可以较好的适用。

**决策树的一些缺点：**

- 决策树学习可能创建一个过于复杂的树，并不能很好的预测数据。也就是过拟合。修剪机制（现在不支持），设置一个叶子节点需要的最小样本数量，或者数的最大深度，可以避免过拟合。
- 决策树可能是不稳定的，因为即使非常小的变异，可能会产生一颗完全不同的树。这个问题通过decision trees with an ensemble来缓解。
- 概念难以学习，因为决策树没有很好的解释他们，例如，XOR, parity or multiplexer problems。
- 如果某些分类占优势，决策树将会创建一棵有偏差的树。因此，建议在训练之前，先抽样使样本均衡。

## 代码

```python
import numpy as np
import math

def equalNums(label_list, label):
    """
    函数说明：
        计算标记集中某个标记的数量
    Parameters：
        label_list - 标记集
        label - 某个标记
    Returns：
        num - 某个标记的数量
    """
    return np.sum(label_list == label)
    
def calcShannonEnt(label_list):
    """
    函数说明：
        计算信息熵
        对应公式 Ent(D) = -∑ Pk*log2(Pk) k=1..len(label_set)
    Parameters：
        label_list - 标记集
    Returns：
        shannonEnt - 当前标记集的信息熵
   """
    label_set = set(label_list)
    len_label_list = label_list.size
    shannonEnt = 0.0
    for label in label_set:
        prob = equalNums(label_list, label)/len_label_list
        shannonEnt -= prob * math.log2(prob)
    return shannonEnt

def conditionnalEntropy(feature_list, label_list):
    """
    函数说明：
        计算条件信息熵，对应信息增益公式中的被减项
    Parameters：
        feature_list - sample_list中某一列，表示当前属性的所有值
        label_list - 标记集
    Returns：
        entropy - 条件信息熵
   """
    feature_list = np.asarray(feature_list)
    label_list = np.asarray(label_list)
    feature_set = set(feature_list)
    entropy = 0.0

    for feat in feature_set:
        pro = equalNums(feature_list, feat)/feature_list.size
        entropy += pro * calcShannonEnt(label_list[feature_list == feat])

    return entropy

def calcInfoGain(feature_list, label_list):
    """
    函数说明：
        计算信息增益
    Parameters：
        feature_list - sample_list中某一列，表示当前属性的所有值
        label_list - 标记集
    Returns：
        当前属性的信息增益
   """
    return calcShannonEnt(label_list) - conditionnalEntropy(feature_list, label_list)

def splitDataSet(sample_list, label_list, axis, value):
    """
    函数说明：
        决策树在选好当前最优划分属性之后划分样本集
        依据value选择对应样例，并去除第axis维属性
    Parameters：
        feature_list - sample_list中某一列，表示当前属性的所有值
        label_list - 标记集
    Returns：
        return_sample_list, return_label_list
   """
    # sample_list[sample_list[...,axis] == value] 利用了numpy数组的布尔索引
    filtered_sample_list = sample_list[sample_list[...,axis] == value]
    return_label_list = label_list[sample_list[...,axis] == value]
    # np.hstack 将数组横向拼接，也就是去除第axis维属性
    return_sample_list = np.hstack((filtered_sample_list[...,:axis], filtered_sample_list[...,axis+1:]))
    return return_sample_list, return_label_list

def chooseBestFeatureToSplit(sample_list, label_list):
    """
    函数说明：
        选取最优划分属性
    Parameters：
        sample_list - 样本集
        label_list - 标记集
    Returns：
        bestFeat_index - 最优划分属性的索引值
   """
    numFeatures = sample_list.shape[1]
    bestInfoGain = 0
    bestFeat_index = -1

    for i in range(numFeatures):
        infoGain = calcInfoGain(sample_list[..., i], label_list)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat_index = i
    return bestFeat_index

def createTree(sample_list, label_list, attr_list_copy):
    """
    函数说明：
        生成决策树
    Parameters：
        sample_list - 样本集
        label_list - 标记集
        attr_list_copy - 属性集（之所以加copy是为了删属性的时候是在副本上，防止递归出错）
    Returns：
        myTree - 最终的决策树
   """
    # attr_list 有del操作，不用副本的话递归会出错
    attr_list = attr_list_copy.copy()
    if len(set(label_list)) == 1:       # 如果只有一种标记，直接返回标记
        return label_list[0]
    elif sample_list.size == 0:         # 如果所有属性都被遍历，返回最多的标记
        return voteLabel(label_list)

    # bestFeat_index 最优划分属性的索引值
    # bestAttr 最优划分属性对应的名字
    bestFeat_index = chooseBestFeatureToSplit(sample_list, label_list)
    bestAttr = attr_list[bestFeat_index]
    myTree = {bestAttr: {}}
    del(attr_list[bestFeat_index])
    feat_set = set(sample_list[..., bestFeat_index])
    # 依据最优划分属性进行划分，并向下递归
    for feat in feat_set:
        return_sample_list, return_label_list = splitDataSet(sample_list, label_list, bestFeat_index, feat)
        myTree[bestAttr][feat] = createTree(return_sample_list, return_label_list, attr_list)
    return myTree

def voteLabel(label_list):
    """
    函数说明：
        这个函数是用在遍历完所有特征时，返回最多的类别
    Parameters：
        label_list: 标记列表
    Returns：
        数量最多的标记
   """

    # unique_label_list 是label_list中标记种类列表
    # label_num 是unique_label_list对应的数量列表
    unique_label_list = list(set(label_list))
    label_num_list = []

    for label in unique_label_list:
        label_num_list.append(equalNums(label_list, label))

    # label_num.index(max(label_num))是label_num数组中最大值的下标
    return unique_label_list[label_num_list.index(max(label_num_list))]

def classify(decisionTree, testVec, attr_list):
    """
    函数说明：
        对tesVec进行分类
    Parameters：
        decisionTree - 决策树
        attr_list - 属性名列表
        testVec - 测试向量
    Returns：
        label - 预测的标记
   """

    feature = list(decisionTree.keys())[0]          # feature为决策树的根节点
    feature_dict = decisionTree[feature]            # feature_dict为根节点下的子树
    feature_index = attr_list.index(feature)        # feature_index为feature对应的属性名索引
    feature_value = testVec[feature_index]          # feature_value为测试集中对应属性的值
    label = None
    if feature_value in feature_dict.keys():
        # 如果没有结果就继续向下找
        if type(feature_dict[feature_value]) == dict:
            label = classify(feature_dict[feature_value], testVec, attr_list)
        else:
            label = feature_dict[feature_value]
    return label

def testAccuracy(decisionTree, test_sample_list, test_label_list, attr_list):
    """
    函数说明：
        测试十次得到正确率均值
    Parameters：
        decisionTree - 决策树
        test_sample_list - 测试样本集
        test_label_list - 测试标记集
        attr_list - 属性集
    Returns：
        
   """
    def oneTime(decisionTree, test_sample_list, test_label_list, attr_list):
        rightNum = 0
        predict_label_list = []
        for i in range(len(test_sample_list)):
            predict_label = classify(decisionTree, test_sample_list[i], attr_list)
            predict_label_list.append(predict_label)
            if predict_label == test_label_list[i]:
                rightNum += 1
        accuracy = rightNum/len(test_sample_list)
        return accuracy
    sum = 0.0
    for i in range(10):
        sum += oneTime(decisionTree, test_sample_list, test_label_list, attr_list)
    av_accuracy = sum/10
    return av_accuracy

dataSet = np.array([[0, 0, 0, 0, 'no'],            
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']])
labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
print(createTree(dataSet[:, :-1], dataSet[:, -1], labels))
```

```
{'有自己的房子': {'0': {'有工作': {'0': 'no', '1': 'yes'}}, '1': 'yes'}}
```
