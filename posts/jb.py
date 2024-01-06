import pathlib
import datetime
import re

tranf = lambda timestamp: datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
root = pathlib.Path(__file__).parent
files = list(root.glob('**/*.md'))
counts = 0
for file in files[:]:
    
    with open(file, 'r', encoding='utf-8') as f:
        flag = 0
        content = f.read()
        if not re.findall(r'markup: pdc\nweight: \d+', content):
            temp = re.findall(r'---\ntitle: .*\ndate: .*\nlastmod: .*\ncategories: .*\ntags: .*\nauthor: "vllbc"\nmathjax: true\nmarkup: pdc\n*?\n---', content)
            if not temp:
                content = '---\ntitle: .*\ndate: .*\nlastmod: .*\ncategories: .*\ntags: .*\nauthor: "vllbc"\nmathjax: true\nmarkup: pdc\n---\n' + content   # 要确保没有format

        
        cdate = tranf(file.stat().st_ctime)
        year = cdate.split('-')[0]
        if int(year) >= 2023:
            content = re.sub(r'date: .*', 'date: ' + cdate, content)
            
        mdate = tranf(file.stat().st_mtime)
        content = re.sub(r'lastmod: .*', 'lastmod: ' + mdate, content)
        print(file)
        tags = re.findall(r'.*posts\\(.*)(\\.*)?\\.*\.md', str(file))[0][0].split('\\')
        content = re.sub(r'categories: .*', 'categories: ' + str(tags[:-1]), content)
        content = re.sub(r'tags: .*', 'tags: ' + str(tags), content)
        
        title = str(file).split('\\')[-2]
        content = re.sub(r'title: .*', f'title: "{title}"', content)

    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
