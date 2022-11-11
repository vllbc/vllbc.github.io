# pyqt


# 对pyqt5的学习

```markdown
QtCore:包含了核心的非GUI功能。此模块用于处理时间、文件和目录、各种数据类型、流、URL、MIME类型、线程或进程。
QtGui包含类窗口系统集成、事件处理、二维图形、基本成像、字体和文本。
qtwidgets模块包含创造经典桌面风格的用户界面提供了一套UI元素的类。
QtMultimedia包含的类来处理多媒体内容和API来访问相机和收音机的功能。
Qtbluetooth模块包含类的扫描设备和连接并与他们互动。描述模块包含了网络编程的类。这些类便于TCP和IP和UDP客户端和服务器的编码，使网络编程更容易和更便携。
Qtpositioning包含类的利用各种可能的来源，确定位置，包括卫星、Wi-Fi、或一个文本文件。
Enginio模块实现了客户端库访问Qt云服务托管的应用程序运行时。
Qtwebsockets模块包含实现WebSocket协议类。
QtWebKit包含一个基于Webkit2图书馆Web浏览器实现类。
Qtwebkitwidgets包含的类的基础webkit1一用于qtwidgets应用Web浏览器的实现。
QtXml包含与XML文件的类。这个模块为SAX和DOM API提供了实现。
QtSvg模块提供了显示SVG文件内容的类。可伸缩矢量图形（SVG）是一种描述二维图形和图形应用的语言。
QtSql模块提供操作数据库的类。
QtTest包含的功能，使pyqt5应用程序的单元测试
```

最简单的创建窗口：

```python
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon


class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.initUI() #界面绘制交给InitUi方法
        
        
    def initUI(self):
        #设置窗口的位置和大小
        self.setGeometry(300, 300, 300, 220)  
        #设置窗口的标题
        self.setWindowTitle('Icon')
        #设置窗口的图标，引用当前目录下的web.png图片
        self.setWindowIcon(QIcon('web.png'))        
        
        #显示窗口
        self.show()
        
        
if __name__ == '__main__':
    #创建应用程序和对象
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_()) 
```



**创建一个按钮**，并绑定退出事件



```python
import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication
from PyQt5.QtCore import QCoreApplication
 
 
class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
        
    def initUI(self):               
        
        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(sys.exit) #和sys.exit事件绑定
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 50)       
        
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Quit button')    
        self.show()
        
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
```

# 定位

```python
import sys
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
 #绝对定位
 
class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
        
    def initUI(self):
        
        lbl1 = QLabel('Zetcode', self)
        lbl1.move(15, 10)
 
        lbl2 = QLabel('tutorials', self)
        lbl2.move(35, 40)
        
        lbl3 = QLabel('for programmers', self)
        lbl3.move(55, 70)        
        
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Absolute')    
        self.show()
        
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
```

一般布局要`QGridLayout()`

```python
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon

class My_blog_control(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        
        self.commit_btn = QPushButton("确定")
        # self.commit_btn.clicked.connect(sys.exit)
        self.cel_btn = QPushButton("退出")

        grid = QGridLayout()
        names = ['Cls', 'Bck', '', 'Close',
                 '7', '8', '9', '/',
                '4', '5', '6', '*',
                 '1', '2', '3', '-',
                '0', '.', '=', '+']
        position = [(i,j) for i in range(5) for j in range(4)]
        for positi,name in zip(position,names):
            if name == '':
                continue
            button = QPushButton(name)
            grid.addWidget(button,*positi)

        self.setLayout(grid)
        # self.setGeometry(300, 300, 850, 520)
        self.move(300,150)
        self.setWindowTitle("MY_BLOG_CONTROL")
        self.show()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mbc = My_blog_control()
    sys.exit(app.exec_())
```




