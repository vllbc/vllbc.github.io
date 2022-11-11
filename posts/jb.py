#-*- coding : utf-8-*-
import pathlib
import os
import re
import urllib.parse
import shutil

root = pathlib.Path('.')
files = list(root.glob('**/*.md'))
for file in files:
    print(file)
    img_path = file.parent.parent / 'image'
    imgs = list(img_path.glob('*'))
    # print(img_l)
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        for img_name in imgs[:]:
            name = urllib.parse.quote(img_name.name)
            if name in content:
                shutil.move(str(img_name), str(file.parent / 'image'))
                imgs.remove(img_name)