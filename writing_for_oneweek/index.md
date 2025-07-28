# 

```dataview
Table file.mtime as "最后修改时间"
WHERE !todo and date(today) - file.mtime <= dur(7 days) and tags[0] != "Reading" and title
Sort file.mtime desc
```

