# one_test



# 用命令行实现博客自动更新，类似于网站后端

需要用到的库有`os`标准库

2020 12/21 更新：冬至日，完成了图像界面。下面的代码

需要的命令有:

1. `os.system("")` 用来模仿dos命令，主要用来生成站点，和git push
2. `os.listdir("")`获取当前目录下所有文件，用来复制site目录的文件到个人网站项目上
3. `yaml`库，用来操作mkdocs的配置文件,即`mkdocs.yml`
4. `shutil`库，用来复制并替换文件
5. 配置文件，`config.yml` 用来获取各需要的文件路径

大概代码如下

又更新了一次，给这个程序加入了命令行参数，使其更方便

## 旧代码：（命令行实现）

```python
import os
import sys
import yaml
import shutil
import argparse
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger('my_test')


parser = argparse.ArgumentParser()
parser.add_argument('-new', type=str,help='是否为新文件')
parser.add_argument('-type', type=str,help='类型')
parser.add_argument('-name', type=str,help='名字')
parser.add_argument('-folder', type=str,help='路径')
parser.add_argument('-messege', type=str,help='提交信息')
args = parser.parse_args()
def copyFiles(sourceDir,targetDir):
    if sourceDir.find("exceptionfolder")>0:
        return
    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,file)
        targetFile = os.path.join(targetDir,file)
        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) !=
 os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
                print(targetFile+ " copy succeeded")
        if os.path.isdir(sourceFile):
            copyFiles(sourceFile, targetFile)

with open(r"config.yml",'r',encoding='utf-8') as fp:
    res = fp.read()
    data = yaml.load(res,Loader=yaml.FullLoader)
    mkdocs_yaml = data['mkdocs_yaml']
    mkdocs_work = data['mkdocs_work_folder']
    githubpage = data['githubpage_folder']
with open(mkdocs_yaml,'r',encoding='utf-8') as fp:
    result = fp.read()
    data = yaml.load(result,Loader=yaml.FullLoader)
    print("注意在执行前要确保单词拼对，而且有对应文件!!!!!")
    clas = args.new
    if clas == "yes":
        types = args.type
        names = args.name
        dirs = args.folder+'.md'
        messge = args.messege
    else:
        messge = args.messege
        os.system(f"cd {mkdocs_work} && mkdocs build --clean")
        copyFiles(f"{mkdocs_work}\site",githubpage)
        if messge != "":
            os.system(f'cd {githubpage} && git add . && git commit -m "{messge}" && git push origin master')
            print("all works OK!!")
            sys.exit()
    for index,name in enumerate(data['nav']):
        if name.get(types):
            print('ok')
            data['nav'][index][types].append({names:dirs})
    with open(mkdocs_yaml,'w',encoding='utf-8') as fp:
        yaml.dump(data,fp,allow_unicode=True)


os.system(f"cd {mkdocs_work} && mkdocs build --clean")
#替换文件


copyFiles(f"{mkdocs_work}\site",githubpage)
if messge != "":
    os.system(f'cd {githubpage} && git add . && git commit -m "{messge}" && git push origin master')
print("all works OK!!")
```

代码非常简单，唯一的技术点就是复制一个目录的所有文件以及子文件到另一个文件夹，用到了递归。

本博客就是通过这个工具提交的！

## 新代码：（图形界面）

```python
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import os
import yaml

config = {
    '算法相关':'sf',
    'Python':'python',
    'Cookbook':'Cookbook',
    '爬虫':'spider',
    'Pandas':'pandas',
    'Pytorch':'pytorch',
    'AI学习':'AI_learn',
    'flask':"flask",
    '其他':'others'
}

with open(r"config.yml",'r',encoding='utf-8') as fp:
    res = fp.read()
    data = yaml.load(res,Loader=yaml.FullLoader)
    mkdocs_yaml = data['mkdocs_yaml']
    mkdocs_work = data['mkdocs_work_folder']
    githubpage = data['githubpage_folder']
    

def copyFiles(sourceDir,targetDir):
    if sourceDir.find("exceptionfolder")>0:
        return
    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,file)
        targetFile = os.path.join(targetDir,file)
        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) !=
 os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
                print(targetFile+ " copy succeeded")
        if os.path.isdir(sourceFile):
            copyFiles(sourceFile, targetFile)


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):  #控件函数
        self.cb=QComboBox()
        self.cb.addItem("请选择类型")
        self.cb.addItem("算法相关")
        self.cb.addItem("Python")
        self.cb.addItem("Cookbook")
        self.cb.addItem("爬虫")
        self.cb.addItem("Pandas")
        self.cb.addItem("AI学习")
        self.cb.addItem("flask")
        self.cb.addItem("其他")
        self.cb.activated.connect(self.showfolder)
        #标签
        self.is_new = QLabel('是否为新文件')
        self.types = QLabel('类型')
        self.name = QLabel('名字')
        self.folder = QLabel('路径')
        self.message = QLabel('提交信息') 
        #文本框
        self.cb_isnew = QComboBox()
        self.cb_isnew.addItem("yes")
        self.cb_isnew.addItem("no")
        self.nameEdit = QLineEdit()
        # self.folderEdit = QLineEdit()
        self.cbfolder = QComboBox()
        self.messageEdit = QTextEdit()
        #按钮
        self.commit_but = QPushButton('确认提交')
        self.commit_but.clicked.connect(self.mains)
        #布局
        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.is_new, 1, 0)
        grid.addWidget(self.cb_isnew, 1, 1)


        grid.addWidget(self.types, 2, 0)
        grid.addWidget(self.cb,2,1)
        # grid.addWidget(typeEdit, 2, 1)
    
        grid.addWidget(self.name, 3, 0)
        grid.addWidget(self.nameEdit, 3, 1)
        
        grid.addWidget(self.folder,4,0)
        grid.addWidget(self.cbfolder,4,1)

        grid.addWidget(self.message,5,0)
        grid.addWidget(self.messageEdit,5,1)

        grid.addWidget(self.commit_but)

        self.setLayout(grid) 
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('MY_BLOG_SERVER')
        self.show()
        
    def mains(self):
        is_new = self.cb_isnew.currentText()
        types = self.cb.currentText()
        names = self.nameEdit.text()
        folder = self.cbfolder.currentText()  
        message = self.messageEdit.toPlainText()
        #如果不是新文件
        if is_new != 'yes':
            os.system(f"cd {mkdocs_work} && mkdocs build --clean")
            copyFiles(f"{mkdocs_work}\site",githubpage)
            if message != "":
                os.system(f'cd {githubpage} && git add . && git commit -m "{message}" && git push origin master')
                QMessageBox.information(self,"恭喜!","所有工作完成!",QMessageBox.Yes | QMessageBox.No)
                sys.exit()
        #如果是新文件
        with open(mkdocs_yaml,'r',encoding='utf-8') as fp:
            result = fp.read()
            data = yaml.load(result,Loader=yaml.FullLoader)
            for index,name in enumerate(data['nav']):
                if name.get(types):
                    data['nav'][index][types].append({names:f"{config[types]}/{folder}"})
    
        with open(mkdocs_yaml,'w',encoding='utf-8') as fp:
            yaml.dump(data,fp,allow_unicode=True)

        os.system(f"cd {mkdocs_work} && mkdocs build --clean")

        copyFiles(f"{mkdocs_work}\site",githubpage)
        
        if message != "":
            os.system(f'cd {githubpage} && git add . && git commit -m "{message}" && git push origin master')
        QMessageBox.information(self,"恭喜!","所有工作完成!",QMessageBox.Yes | QMessageBox.No)
        sys.exit()

    def showfolder(self):
        self.cbfolder.addItem("请选择文件")
        docs_folder = mkdocs_yaml.strip('mkdocs.yml')+f'docs/{config[self.cb.currentText()]}'
        for file in os.listdir(docs_folder):
            self.cbfolder.addItem(file)
if  __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
```
