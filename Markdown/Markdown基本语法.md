# Markedown用法笔记
**Markdown Preview Enhanced**是完善预览功能的*插件*

**Maridown pdf**将.md文件转换成其他格式

**空格**：切换全角，直接输入space

**换行**：连续两下回车

**\`\`\`**:　　表示代码块
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

例：[百度](http://www.baidu.com)

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
\! \[pic\]\( url\)

\!\[\]\(https://github.com/hmzbox/Study-notes/blob/master/Markdown/images/markdown.jpg\)
![](https://github.com/hmzbox/Study-notes/blob/master/Markdown/images/markdown.jpg)

\<div align=center>\<img src="./images/3.1Reducing_bias_and_variance.png" width="500">\</div>

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
## 10.表情（emoji）
    :smile::smiley::smirk::worried::expressionless:
:smile:　:smiley:　:smirk:　:worried:　:expressionless:

---
## 11 数学公式
* 行内公式：将公式插入到本行内 \$xyz\$ 　　　　$xyz$
* 独行公式：将公式插入到新的一行内，并且居中 \$\$xyz\$\$ 　　　$$xyz$$
- 上标符号，符号：^，如：\$x\^4\$　　　　　　　　　 $x^4$
- 下标符号，符号：_，如：\$x_1\$ 　　　　　　　　　$x_1$
- 多个下标，符号：{} ，\$x_{1ba}\$ 　　　　　　　　　$x_{1ba}$
+ 加法运算，符号：+，如：\$x+y=z\$　　　　　　　　$x+y=z$
+ 减法运算，符号：-，如：\$x-y=z\$　　 　　　　　　$x-y=z$
+ 乘法运算，符号：\times，如：\$x \times y=z\$　　　$x \times y=z$
+ 点乘运算，符号：\cdot，如：\$x \cdot y=z\$　　　　$x \cdot y=z$
+ 根号运算，符号：\sqrt,如：\$ \sqrt{1+2+3}\$　　　　　　$\sqrt{1+2+3}$
+ 大于等于运算，符号：\geq，如：$x+y \geq z$
+ 小于等于运算，符号：\leq，如：$x+y \leq z$
---
+ 向右推出，符号：\Rightarrow,如：$\Rightarrow$
+ 上下除法显示，符号：\frac,如：$\frac {2ac}{b^2}$
+ 省略号，符号：\cdots，如：$\cdots$
+ 空，符号：\varnothing，如：P={$\varnothing$}
+ 属于，符号：\in ,如：$a\in b$
+ 不属于，符号：\notin ,如：$a\notin b$
+ 并，符号：\cup ,如：$ x \cup y $
+ 交，符号：\cap ,如：$ x \cap y $
---
, 符号：,如：
+ 矢量, 符号：\vec,如：$\vec{a}$
+ 积分, 符号：\int_0^1 x^2{\rm d}x , 如：$\int_0^1x^2{\rm d}x $
+ 极限, 符号：\lim_{n\rightarrow+\infty}\frac{1}{n(n+1)}, 如：$\lim_{n\rightarrow+\infty}\frac{1}{n(n+1)}$

+ 累加, 符号: \sum_1^ n\frac{1}{x^2} ,如：$\sum_1^n\frac{1}{x^2}$
+ 累乘, 符号：\prod_{i=0}^n,如 $\prod_{i=0}^n\frac{1}{x^2}$
---
**希腊字母**
+ $\infty$  \infty
+ $\alpha$  \alpha
+ $\beta$   \beta
+ $\gamma$  \gamma
+ $\Gamma$  \Gamma
+ $\delta$  \delta
+ $\theta$  \theta
+ $\varepsilon$ \varepsilon
+ $\lambda$ \lambda
+ $\mu$   \mu
+ $\sigma$  \sigma
+ $\pi$  \pi
+ $\varphi$ \varphi
+ $\Phi$  \Phi
+ $\chi$  \chi
+ $\omega$  \omega
+ $\Omega$  \Omega



---
## 12矩阵

$$
\left[
 \begin{matrix}
   a & b & c & d & e\\
   f & g & h & i & j \\
   k & l & m & n & o \\
   p & q & r & s & t
  \end{matrix} 
\right]
$$
```
$$
\left[
 \begin{matrix}
   a & b & c & d & e\\
   f & g & h & i & j \\
   k & l & m & n & o \\
   p & q & r & s & t
  \end{matrix} 
\right]
$$
```

## 13字体颜色
$\color{red}红字$ \color{red}红字
$\color{blue}蓝字$ \color{red}蓝字