# 


```dataview
Table rows.file.link as filename
WHERE todo and title != "<% tp.file.title %>"
Sort file.ctime desc
GROUP BY tags[1] as "category"
```







