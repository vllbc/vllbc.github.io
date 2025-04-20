# einsum

>einops.einsum calls einsum operations with einops-style named axes indexing, computing tensor products with an arbitrary number of tensors. Unlike typical einsum syntax, here you must pass tensors first, and then the pattern.

>Also, note that rearrange operations such as `"(batch chan) out"`, or singleton axes `()`, are not currently supported.

爱因斯坦求和

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250111211532.png)
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250111211543.png)

