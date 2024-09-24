# jupyter_pdf



# 浅谈jupyter转pdf问题

​	老师要求要把之前做的实验打印出来，但是由于学院的系统环境问题，没有办法直接保存为pdf，因此我采用了别的方法，我先把系统上的文件保存为.ipynb格式，然后在自己的电脑环境上打开，再下载了pandoc和MiKTex后，安装所需要的依赖后成功保存为pdf格式，但是有个问题就是中文无法显示，在网上搜索过后，在生成的.tex文件里面在适当的位置加入如下代码

```latex
\usepackage{fontspec, xunicode, xltxtra}
\setmainfont{Microsoft YaHei}
\usepackage{ctex}
```

然后在命令行中输入`xelatex **.tex`即可，也可以直接在jupyter中保存为.tex文件，然后再进行上述操作显示中文。

不过一般来说可以对工具进行设置从而不需要每次都要添加这些东西。目前也没有搜索到什么内容。

