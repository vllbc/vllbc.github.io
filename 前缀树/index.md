# 前缀树


## 什么是前缀树？
前缀树是**N叉树的一种特殊形式**。通常来说，一个前缀树是用来存储字符串的。前缀树的每一个节点代表一个字符串（前缀）。每一个节点会有多个子节点，通往不同子节点的路径上有着不同的字符。子节点代表的字符串是由节点本身的原始字符串，以及通往该子节点路径上所有的字符组成的。
![](image/Pasted%20image%2020220727141408.png)
在上图示例中，我们在节点中标记的值是该节点对应表示的字符串。例如，我们从根节点开始，选择第二条路径 ‘b’，然后选择它的第一个子节点 ‘a’，接下来继续选择子节点 ‘d’，我们最终会到达叶节点 “bad”。节点的值是由从根节点开始，与其经过的路径中的字符按顺序形成的。

值得注意的是，根节点表示空字符串。

前缀树的一个重要的特性是，节点所有的后代都与该节点相关的字符串有着共同的前缀。这就是前缀树名称的由来。

我们再来看这个例子。例如，以节点 “b” 为根的子树中的节点表示的字符串，都具有共同的前缀 “b”。反之亦然，具有公共前缀 “b” 的字符串，全部位于以 “b” 为根的子树中，并且具有不同前缀的字符串来自不同的分支。

前缀树有着广泛的应用，例如自动补全，拼写检查等等。



## 代码
```python
import collections

class TrieNode(object): # 定义节点

    # Initialize your data structure here.

    def __init__(self):

        self.node = collections.defaultdict(TrieNode)
        self.char = ""

        self.is_word = False

    @property

    def data(self):

        return self.node

    def __getitem__(self, key):

        return self.node[key]
        
    def __str__(self) -> str:

        return self.char

    __repr__ = __str__

class Trie(object):

    def __init__(self):

        """

        Initialize your data structure here.

        """

        self.root = TrieNode()

    def insert(self, word):

        """

        Inserts a word into the trie.

        :type word: str

        :rtype: void

        """

        node = self.root

        for chars in word:
			temp = node.char
            node = node[chars] # 因为defaultdict，如果不存在则自动生成键
            node.char = temp + chars # 键可以视为路径上的字符，节点可以视为从根到当前节点表示的前缀或单词。

        node.is_word = True # 这里很重要，用于search判断是否为完整的单词

    def search(self, word):

        """

        Returns if the word is in the trie.

        :type word: str

        :rtype: bool

        """

        node = self.root

        for chars in word:

            if chars not in node.data.keys(): # 因为defaultdict所以不能判断value为None，要判断键是否存在

                return False

            node = node[chars]

        # 判断单词是否是完整的存在在trie树中

        return node.is_word    

    def startsWith(self, prefix):

        """

        Returns if there is any word in the trie that starts with the given prefix.

        :type prefix: str

        :rtype: bool

        """

        node = self.root

        for chars in prefix:

            if chars not in node.data.keys(): # 和上面同理

                return False

            node = node[chars]

        return True
        
    def get_all_words(self): # 获取所有的单词

        q = [self.root]

        while q:

            node = q.pop(0) # 相当于一个队列

            for child in node.data.values():

                if child.is_word:

                    yield child.char

                q.append(child)
```


