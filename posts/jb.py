#-*- coding : utf-8-*-
import pathlib
import os
import re
import urllib.parse
import shutil

root = pathlib.Path('算法题')
files = list(root.glob('**/*.md'))
count = 0
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        content = content.replace('sf', '算法题')

    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

    


        