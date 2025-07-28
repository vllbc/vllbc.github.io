# 

```dataview
Table file.mtime as "最后修改时间"
WHERE tags[0] = "Reading" and date(today) - file.mtime <= dur(7 days)
Sort file.mtime desc
```

