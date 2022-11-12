# 新手时期的一个题

# 新手时期的一个题
前几天在洛谷刚刷的一个题目

~~当时在一些地方踩了坑，写出来吸取教训~~

洛谷链接：https://www.luogu.com.cn/problem/P1055



当时获取输入的时候是用%d来获取的，后来发现数据十分异常，通过测试发现

> -号被识别成了符号，所以获取的输入异常

经过很久的思考

> <u>然后我恍然大悟，用%c获取就好了啊！</u>

先贴上代码

```c++
#include <iostream>
#include <stdlib.h>
#include <cstdio>
using namespace std;

int main(){
    char temp;
    char num[9];
    scanf("%c-%c%c%c-%c%c%c%c%c-%c",&num[0],&num[1],&num[2],&num[3],&num[4],&num[5],&num[6],&num[7],&num[8],&temp);
    int sum = 0;
    int X = 'X';
    for(int n =1;n<=9;n++){
        sum=sum+((num[n-1]-48)*n);
    }
    
    if(sum%11 == 10){
        if(temp=='X'){
            printf("Right");
        }
        else
        {
            printf("%c-%c%c%c-%c%c%c%c%c-%c",num[0],num[1],num[2],num[3],num[4],num[5],num[6],num[7],num[8],'X');
        }
        
    }
   else if(sum%11 == (int)temp-48){
        printf("Right");
   }
   else if(sum%11 != (int)temp-48)
   {
       printf("%c-%c%c%c-%c%c%c%c%c-%c",num[0],num[1],num[2],num[3],num[4],num[5],num[6],num[7],num[8],(char)((sum%11)+48));
   }
   
    system("pause");
    return 0;
}

```

虽然我写的很臃肿，但至少AC了

后来也没想着优化代码。

总之我就是通过字符获取输入然后转换成整数型在进行判断

总体的难点就在这，普及题的难度，后面就很简单了！

我获得的收获如下：

1. `获取输入时要灵活判断是字符还是整数`
2. `灵活使用数组`
3. `要及时优化代码`
4. `向大佬学习！`


