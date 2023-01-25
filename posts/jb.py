import pathlib
import os
import datetime
import re
import urllib.parse
import shutil

tranf = lambda timestamp: datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
root = pathlib.Path(__file__).parent
files = list(root.glob('**/*.md'))
count = 0
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        mdate = tranf(file.stat().st_mtime)
        content = re.sub(r'lastmod: .*', 'lastmod: ' + mdate, content)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)