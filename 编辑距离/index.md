# 编辑距离



# 编辑距离
## 定义

编辑距离(Edit Distance)是针对两个字符串S1和S2的差异程度进行量化，计算方式是看至少需要多少次的处理才能将S1变成S2（和S2变成S1是等价的），用 EditDis(S1,S2)表示。

其中处理的方式有三种：  
1.插入一个字符  
2.删除一个字符  
3.替换一个字符

这是严格意义上的距离，满足”距离三公理"：

1.对称性，EditDis(S1,S2) = EditDis(S2,S1)
2.非负性，EditDis(S1,S2) >= 0, 当且仅当S1=S2时，等号成立
3.三角不等式，EditDis(S1,S2) + EditDis(S1,S3) >= EditDis(S2,S3)

## 动态规划求解

令 S1.substr(i)表示 S1前i个字符构成的子串，S2.substr(j)表示S2前j个字符构成的子串
$dp[i][j]$表示S1.substr(i)和S2.substr(j)的编辑距离。
注意，这句话的另一层含义是：当我们计算出了EditDis(S1,S2)则默认了S1.substr(i)已经通过三种处理方式变成了S2.substr(j)。
所以，当我们计算$dp[i+1][j+1]$时，我们可以利用$dp[i][j]$，$dp[i+1][j]$，$dp[i][j+1]$的信息。

## 过程

1.确定最后一步：
令，S1的长度为len1，S2的长度为len2
$dp[len1][len2]$有三种方式可以实现，一种是S1.substr(len1-1)插入一个字符使之等于S2（$dp[len1][len2-1] + 1$），一种是S2.substr(len2)删除一个字符使之等于S1（$dp[len1-1][len2] + 1$），另一种是替换最后一个字符使S1和S2相等（$dp[len1-1][len2-1] + 1$）
![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020220829234524.png)

2. 确定转移方程:

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020220829234535.png)

3.确定边界和初始状态  
我们设定Dp二维数组大小是（Len+1） * （Len2+1），第0行代表 S1为空串，第0列代表S2为空串。  
显然，S1变成为空串需要的每次操作是$dp[i][0]=i$  
S2变成为空串需要的每次操作是$dp[0][j] = j$


## 代码

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        import numpy as np
        len1, len2 = len(word1), len(word2)
        dp = [[0]*(len2+1) for _ in range(len1+1)] # 定义  
        for i in range(1,len1+1):
            dp[i][0] = i # 边界
        for j in range(1,len2+1):
            dp[0][j] = j # 边界
        for i in range(1,len1+1):
            for j in range(1,len2+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)
        return int(dp[len1][len2])

```
