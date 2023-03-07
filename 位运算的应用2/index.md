# 位运算的应用（2）



# 位运算的应用（2）

也是leetcode上的一个题目，题目是这样的

[https://leetcode-cn.com/problems/number-of-1-bits/](https://leetcode-cn.com/problems/number-of-1-bits/)

话不多说 直接上代码

```c++
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count = 0;
        while(n>0){
            if((n&1)==1){ //如果末位为1
                count++;
            }
            n=(n>>1); //每次向右移动一位
        }
        return count;
    }

};
```
