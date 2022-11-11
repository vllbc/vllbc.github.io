# 10


# 字符串转换整数 (atoi)

[https://leetcode-cn.com/problems/string-to-integer-atoi/](https://leetcode-cn.com/problems/string-to-integer-atoi/)

```python
#重点是正则表达式

class Solution:
    def myAtoi(s: str):
        import re
        ss = re.findall("^[\+\-]?\d+",s.strip())
        res = int(*ss)
        if res > (2**31-1):
            res = (2**31-1)
        if res < -2**31:
            res = -2**31
        return res
```

WA了四次才整出来，太菜了，以为很简单，没有认真读题，要吸取教训。


