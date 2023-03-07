# git



## 常用命令
### diff
`git diff`：当工作区有改动，临时区为空，diff的对比是“工作区”与最后一次commit提交的仓库的共同文件”；当工作区有改动，临时区不为空，diff对比的是“工作区”与“暂存区”的共同文件”。
`git diff --cached`：显示暂存区与最后一次commit的改动。
`git diff` <分支1> <分支2> 显示两个分支最后一次commit的改动。

### init
`git init`初始化一个仓库


### add 
 `git add .`将除`.gitignore`中声明的所有文件加入暂存区
 也可以git add 特定文件，将文件加入暂存区。

### commit
`git commit -m "提交说明"` 提交到工作区

`git commit --amend` 修改上次的提交记录

### push
`git push` 提交到远程库。可以指定分支提交
比如：`git push origin main` 指定为main分支

### pull

`git pull` 将改动拉倒本地库，并合并，默认为Merge合并
如果加上`rebase`参数则合并方式变为rebase合并。

### log
`git log` 查看提交历史
`git log --oneline`简洁输出


### branch

`git branch xxx` 建立分支xxx
然后用`git checkout xxx` 切换到分支xxx
或者用`git checkout -b xxx` 完成同样的操作


### checkout
`git checkout` 回退版本即head分离 但master不变
checkout可以切换分支，也可以head分离，但是分支的位置不变，但后面的reset和revert都会改变分支

### reset

`git reset --hard HEAD^`回退一个版本（master也改变）
`git reset --hard HEAD~n` 回退n个版本
`git reset --hard  xxxxxx`  xxxxxx为版本号 即为git log显示的每一个版本号 一般为前六位

reset的参数有三种，其作用如下：
![](image/Pasted%20image%2020220830163125.png)

最危险但最常用的就是hard。

### revert
`git revert`命令用于撤消对仓库提交历史的更改。其他“撤消”命令，例如 git checkout 和 git reset，将HEAD和分支引用指针移动到指定的提交。git revert也需要一个指定的提交，但是，它并不会将 ref 指针移动到这个提交。revert 操作将采用反转指定的提交的更改，并创建一个新的“还原提交”。然后更新 ref 指针以指向新的还原提交，使其成为分支的HEAD。

### merge
 `git merge` 比如在master分支里 执行git merge xxx 将xxx分支合并到master中，一般项目开发，一人一个分支，最后提交的时候合并再提交。不过更推荐用git rebase方法，这样合并后的分支更加直观

### branch
最常用的就是创建和删除分支

`git branch -f master~3`将分支强制回退三个版本，但head不动


### cherry-pick
当需要另一个分支的所有改动时，用`git merge`，但当需要部分改动时候，要用`git cherry-pick xxx`   xxx为哈希值或者分支名，指定为分支名时候，将分支的最新改动合并过来

### rebase
当不知道提交的哈希值时，可以用`git rebase -i HEAD~x` 来可视化管理，可以调整提交的顺序，可以删除不想要的提交，或合并提交

`git rebase xx1 xx2`将xx2分支上的提交记录放到xx1后面


### fetch
`git fetch`获取远程仓库的数据，不会改变你本地仓库的状态，不会更新你的master,也没有更改你的本地磁盘文件，可以理解为单纯的下载操作。而`git pull`相当于`git fetch + git merge`即抓取后合并

### reflog
`git reflog` 查看操作记录，这个操作可以撤销不小心用`git reset`回退版本的操作


## 底层内容

这几天系统学习了一下git的底层内容，通透了很多，记录一些。贴一下原博客：[https://www.lzane.com/tech/git-internal/](https://www.lzane.com/tech/git-internal/)
首先看一个图

![](image/Pasted%20image%2020221113190157.png)
首先要明确有四种object，第一种是记录文件内容，第二种是记录目录结构，第三种是记录提交信息，第四种是记录tag信息，第四种无关紧要。

从下面开始看，最下面记录的文件内容，注意只记录文件内容，不包括文件名等其它内容。是一个blob类型的节点，将文件的内容信息经过SHA1哈希算法得到对应的哈希值作为这个object在Git仓库中的唯一身份证。

然后再往上的三角形记录的是仓库的目录结构，它将当前的目录结构打了一个快照。从它储存的内容来看可以发现它储存了一个目录结构（类似于文件夹），以及每一个文件（或者子文件夹）的权限、类型、对应的身份证（SHA1值）、以及文件名。

再往上就是记录的提交的信息，它储存的是一个提交的信息，包括对应目录结构的快照tree的哈希值，上一个提交的哈希值，提交的作者以及提交的具体时间，最后是该提交的信息。

还有分支的信息和Head。HEAD、分支和普通的Tag可以理解为一个指针，指向对应commit的sha1值。

仓库有三个分区：工作目录、index索引区域、Git仓库。

当文件被修改后，只是工作目录发生了改变，其余两个是没有任何变化的。当运行`git add xxx`命令后，即将xxx文件加入了索引区域，此时新建了一个blob object，并且将原来指向xxx指向了新建的blob Object，记住索引索引的是add的所有文件，这时运行`git commit`，会生成一个tree object，然后创建commit object，将分支等信息指向新的commit。注意每次commit都是储存的全新的文件快照而不是变更部分。
