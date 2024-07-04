# 课程表（拓扑排序）


> Problem: 

  

## 思路

  

> 注意拓扑排序最好是邻接表（哈系表实现），并用队列处理后续入度为0的点

  

## 解题方法

  

> 描述你的解题方法

  

## 复杂度

  

时间复杂度:

> 添加时间复杂度, 示例： $O(n)$

  

空间复杂度:

> 添加空间复杂度, 示例： $O(n)$

  
  
  

## Code

```Python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        if not prerequisites:

            return True

        def get_zero(numCourses, prerequisites, temp):

            queue = []

            for prerequisite in prerequisites:

                temp[prerequisite[0]] += 1

            for i in range(len(temp)):

                if temp[i] == 0:

                    queue.append(i)

            return queue

        temp = [0 for _ in range(numCourses)]

        queue = get_zero(numCourses, prerequisites, temp)

        if queue == []:

            return False

        from collections import defaultdict

        d = defaultdict(list)

        for prerequisite in prerequisites:

            d[prerequisite[1]].append(prerequisite[0])

        while queue:

            idx = queue.pop(0)

            nexs = d[idx]

            if nexs:

                for nex in nexs:

                    temp[nex] -= 1

                    if temp[nex] == 0:

                        queue.append(nex)

            numCourses -= 1

        return numCourses == 0

```
