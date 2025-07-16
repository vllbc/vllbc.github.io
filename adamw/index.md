# AdamW

AdamW相对与Adam的改动十分简单，其将权重衰减项从梯度的计算中拿出来直接加在了最后的权重更新步骤上（图1，式12）。其提出的动机在于：原先Adam的实现中如果采用了 [L2权重衰减](https://zhida.zhihu.com/search?content_id=231119964&content_type=Article&match_order=1&q=L2%E6%9D%83%E9%87%8D%E8%A1%B0%E5%87%8F&zhida_source=entity)，则相应的权重衰减项会被直接加在loss里，从而导致动量的一阶与二阶滑动平均均考虑了该权重衰减项，而这影响了Adam的优化效果，而将权重衰减与梯度的计算进行解耦能够显著提升Adam的效果。目前，AdamW现在已经成为[transformer](https://zhida.zhihu.com/search?content_id=231119964&content_type=Article&match_order=1&q=transformer&zhida_source=entity)训练中的默认优化器了。

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250712002850.png)

参考：[# Adam和AdamW](https://zhuanlan.zhihu.com/p/643452086)
