# Markedown用法笔记
**Markdown Preview Enhanced**是完善预览功能的*插件*

**Maridown pdf**将.md文件转换成其他格式

## 1. 标题

可使用1到6个"#"可表示1-　6级标题

例如：##  ###  ####

---
## 2. 分割线
三个及其以上的 - or * 表示例如：--- ***

---
## 3.内容强调
+ 斜体　\*斜体\*=*斜体*
+ 粗体　\*粗体\*=**粗体**
+ 又斜又粗　\*\*\*又斜又粗\*\*\*=***又斜又粗***
+ 删除线 \~\~内容\~\~=~~内容~~
+ 下划线　　<\u>下划线例子<\/u>=<u>下划线例子</u>
+ 高亮显示　\`高亮显示\`=`高亮显示`
+ 上下标 上标：X<\sub>2<\/sub> =X<sub>2</sub>　　O<\sup>3<\sup>=O<sup>3</sup>
---
## 4.超链接
\[百度\]\(http://www.baidu.com\)

例如：[百度](http://www.baidu.com)

---
## 5.列表
`无序列表`
* c++
- python
+ javascript

`有序列表`
1. apple
2. orange
3. banana

`嵌套列表`
* 第一层
   * 第二层
      1. 第三层1
      2. 第三层2

* 第一层
   + 第二层
      - 第三层

---
## 6.插入图片
\! \[pic] \( url)

例子：
![](https://github.com/hmzbox/Study-notes/blob/master/Markdown/images/markdown.jpg)

## 7.文字引用
\>第一层

\>>第二层

\>>>第三层

\>> 这还是第三层

\>换行 就可以跳出去

>第一层
>>第二层
>>>第三层
>> 这还是第三层

>换行 就可以跳出去

---
## 8. 代码

使用tab然后输入代码块
```
    public class helloword
    {
        public statc void main(String [] args)
        {
            system.out.println("hello world");
        }
    }
```

***问题：为什么不显示代码块行号***

---
## 9.表格
表头头1\|表头头2\|表头头3

\:--------|:-------:|--------:

居左　　\|

----------\|居中　\|

居左　　\|居中　\|居右边

---
表头头头头1|表头头头2|表头头头头3
:---------|:-------:|----------:
居左      |
----------|居中     |
居左      |居中     |居右边

---
10.emoji

:simle:
:smilley:
:smirk:
:worried:
:expressionless: