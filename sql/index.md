# sql



# **sql学习**

> 1. 连接数据库 mysql -u root -p 输入密码即可
>
> 2. 显示所有的数据库 show databases;
>
> 3. use databasename 使用某个数据库
>
> 4. CREATE DATABASE xxx; 创建数据库
>
> 5. drop database xxx;  删除数据库
>
> 6. show tables; 显示这个数据库下的所有数据表
>
> 7. CREATE TABLE  'xxx'  (
>
>    ​	'columns'  type,
>    ​	.....
>
>    );
>
> 8. DROP TABLE xxx;
>
> 9. INSERT  INTO xxx (columns,....) VALUES (values,....);
>
> 10. SELECT  *  from tablename;选择数据表中所有数据，具体查询方法以后写
>
> 11. DELETE from tablename;删除表中的所有数据，也可以加where限制条件
>
> 12. UPDATE  table  SET ...... where  .......;
>
> 13. CASE WHEN  ... THEN ...  ELSE ...  END;
>
> 14. SQL JOIN 子句用于把来自两个或多个表的行结合起来，基于这些表之间的共同字段。分为内连接，外连接，左连接，右连接
>
> 15. ORDER BY查询的时候用来排序  ASC升序,DESC为降序.
>
> 16. WHERE的一些用法:
>     where xx  in ('','')类似于python的in
>     where xxx BETWEEN  a AND b 从a到b
>
> 17. like 的一些用法：
>     like '%x' %匹配任意多的字符
>     like '_x%'  _匹配任意单个字符
>
> 18. concat() 连接字符函数
>
> 19. REPLACE(a,b,c)将a中的b替换为c
>
> 20. `limit y` 分句表示: 读取 y 条数据
>
> 21. `limit x, y` 分句表示: 跳过 x 条数据，读取 y 条数据
>
> 22. `limit y offset x` 分句表示: 跳过 x 条数据，读取 y 条数据
>
> 23. SELECT distinct 去重查询 也可以用在一些聚合函数里
>
> 24. GROUP BY 分组
>
> 25. HAVING 解决WHERE 无法和聚合函数一起用
>

附：

sql各连接![avatar](sqllj.jpg)
