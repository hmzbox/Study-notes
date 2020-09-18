- [第2章 图像编码与基本嵌入方法](#第2章-图像编码与基本嵌入方法)
  - [2.1 RGB、YUV和YC<sub>b</sub>C<sub>r</sub>空域图像编码格式](#21-rgbyuv和ycsubbsubcsubrsub空域图像编码格式)
  - [2.2 JPEG 变换域图像编码流程](#22-jpeg-变换域图像编码流程)
  - [2.3 二元嵌入 LSBR（least significant bit replacement）与三元嵌入LSBM（LSB matching）](#23-二元嵌入-lsbrleast-significant-bit-replacement与三元嵌入lsbmlsb-matching)
    - [二元嵌入LSBR](#二元嵌入lsbr)
    - [三元嵌入LSBM](#三元嵌入lsbm)
  - [2.4 调色板隐写](#24-调色板隐写)
    - [基于亮度排序的调色板](#基于亮度排序的调色板)
    - [基于距离排序的最佳奇偶分配（optimum parity assignment，OPA）的调色板隐写](#基于距离排序的最佳奇偶分配optimum-parity-assignmentopa的调色板隐写)
  - [2.5量化调制与格（略）](#25量化调制与格略)
- [第3章 隐写分布特性的保持](#第3章-隐写分布特性的保持)
  - [3.1 分布特性的识别（$\chi$<sup>2</sup>统计量：识别LSBR)](#31-分布特性的识别chisup2sup统计量识别lsbr)
  - [3.2 基于预留补偿区的分布恢复的方法（OutGuess隐写方案/软件）](#32-基于预留补偿区的分布恢复的方法outguess隐写方案软件)
  - [3.3基于统计模型的方法（Model-based，MB）](#33基于统计模型的方法model-basedmb)
  - [3.4基于调整修改方式的隐写(F3/F4/F5)](#34基于调整修改方式的隐写f3f4f5)
    - [F3隐写](#f3隐写)
    - [F4隐写](#f4隐写)


---
## 第2章 图像编码与基本嵌入方法

---
### 2.1 RGB、YUV和YC<sub>b</sub>C<sub>r</sub>空域图像编码格式

**RGB(模型)**：RGB888是依据人眼识别的颜色定义出的空间。它的细节难以进行数字化的调整。

RGB的取值范围为：[0,255]、[0,255]、[0,255]

**YUV(模型)**: 
YUV的取值范围为：[0,255]、[负整数,正整数]、[负整数,正整数]
1. 相对于RGB，将亮度和色度分离开，从而适合于图像处理领域
2. 相对于RGB，兼容彩色电视与黑白电视问题，只有Y则是灰度图像。

**Y**表示**亮度(Luminance component/Luma)** Y = 0.299R + 0.587G + 0.114B，只有Y信号=**灰度图像**

**U、V**用来表示**色度(Chrominace/chroma)**，**U、V**是构成彩色的两个分量。

此时，我们设 U = 0.492(B - Y)，V=0.877(R-Y) 则可以求得：

U = - 0.147R- 0.289G + 0.436B　　V = 0.615R - 0.515G - 0.100B
即：

![](https://github.com/hmzbox/Study-notes/blob/master/Stego/images/YUV%20mode.png)

**YC<sub>b</sub>C<sub>r</sub>**: 为了统一YUV模型的取值范围为[0,255]

![](https://github.com/hmzbox/Study-notes/blob/master/Stego/images/YCbCr.png)


**总结2.1**：YUV色彩模型来源于RGB模型，YCbCr模型来源于YUV模型，还未涉及压缩，压缩是JPEG通过YUV411或YUV422**采样**实现。YUV为分离亮度和色度，及实现黑白彩色电视兼容提出，YCbCr为改变YUV取值范围为[0,255]

---
### 2.2 JPEG 变换域图像编码流程
**有损压缩**

因人眼对亮度变换的敏感度要比对色彩变换的敏感度高出很多，所以，可以认为Y分量要比C<sub>b</sub>和C<sub>r</sub>分量重要的多。

通常采用两种采样方式：YUV411和YUV422，它们所代表的意义是Y,C<sub>b</sub>和C<sub>r</sub>三个分量的数据取样比例一般是4：1：1或者4：2：2（在2x2的单元中，本应分别有4个Y，4个U，4个V值，用**12**个字节存储。经4:1:1采样处理后，每个单元中的值分别有4个Y、1个U、1个V，只用**6**个字节。

![](https://github.com/hmzbox/Study-notes/blob/master/Stego/images/JPEG.png)

1. **YC<sub>b</sub>C<sub>r</sub>格式输入**。若输入不是YC<sub>b</sub>C<sub>r</sub>格式，则转换成YC<sub>b</sub>C<sub>r</sub>;
2. **图像切分成8×8和采样**。若H和W不是8的倍数，则需要扩充到8的倍数（Y通常是嵌入域），并对C<sub>b</sub>和C<sub>r</sub>进行YUV411或YUV422采样；（例如16×16的块，包含4个Y分块和1个C<sub>b</sub>和1个C<sub>r</sub>）
3. **DCT II变换**。将输入分块均-128,范围 [0,255] 到 [-128,127] 以满足DCT输入条件（偶函数）；每个8×8分块都通过DCT II变换得到8×8的浮点数DCT系数矩阵 M<sub>dct</sub>（最左上角的点为直流系数DC，其余为交流系数AC）。
4. **量化**=M<sub>dct</sub> / Q<sub>i</sub>，然后取整。通过质量因子QF(如80)选取8×8的量化表 Q<sub>i</sub>，QF的取值范围为（70-100）（新的量化后矩阵就会出现很多0）
5. **无损压缩**。64个DCT矩阵排成一维序列，**DC部分**采用差分编码（DPCM）只记录相邻块的DC系数差，**AC系数部分**采用游程编码RLEC，最后将DPCM和RLEC进行熵编码，也就是哈夫曼编码(Huffman),最后加上头文件。

---
### 2.3 二元嵌入 LSBR（least significant bit replacement）与三元嵌入LSBM（LSB matching）

#### 二元嵌入LSBR
* **应用**：**Jsteg**隐写软件，在JPEG量化DCT系数采用LSBR
* **原理**：**奇小偶大**，替换最小bit位，最小位若是0（**偶数**）变1（**+1**），最小位若是1（**奇数**）变0（**-1**）
* **缺陷**：**值对**现象出现，使得相邻像素的值接近，如：2i与2i+1趋于相等，如图：
![](https://github.com/hmzbox/Study-notes/blob/master/Stego/images/LSBRHistogram.png)
* **优势**：在JPEG量化系数-1、0、1不适用，通常最小值对是2与3、是-2与-3，**奇小偶大**可以使得1与2、-1与-2是最小值对，从分利用了分布较密的1与-1。
#### 三元嵌入LSBM

* **提出原因**：解决LSBR值对问题
* **原理**：![](https://github.com/hmzbox/Study-notes/blob/master/Stego/images/LSBM.png)

如：2=(010)<sub>2</sub>，**if** s=0，(010)<sub>2</sub>不变；**if** s=1，(010)<sub>2</sub>+1=(011)<sub>2</sub>=3 or (010)<sub>2</sub>-1=(001)<sub>2</sub>=1
* **性能分析**：![](https://github.com/hmzbox/Study-notes/blob/master/Stego/images/LSBMHistogram.png)
**总结2.3**：三元嵌入LSBM解决二元嵌入LSBR值对问题，提高安全性。

---
### 2.4 调色板隐写

#### 基于亮度排序的调色板
* **应用**：**EzStego**亮度排序调色板，再对索引值进行LSBR
* **隐写原理**： 亮度序号奇偶性就是颜色的奇偶性，奇偶性代表了隐写值
* **缺陷**：亮度排序**不**准确，EzStego软件隐写后图片色彩有跳跃性

#### 基于距离排序的最佳奇偶分配（optimum parity assignment，OPA）的调色板隐写
* **OPA的作用**：使得$c_i$与最近距离$s_i$的奇偶性必定不同
* **颜色距离的定义**： $d[i,j]=d_{RGB} (c_i,c_j)=\sqrt{(r_i+r_j)^2+(g_i+g_j)^2+(b_i+b_j)^2}$
* **嵌入原理**：使用颜色三个分量和的奇偶性$P(c_i)=(r_i+g_i+b_i)mod 2$表示隐写消息的比特
* **OPA隐写原理** 
设调色颜色共有N个,令$P$={空},$c_i$奇偶性用$P(c_i)$（**0/1**分别代表二元奇偶性）表示。**（注意，$P$和$P(c_i)$不是一个东西，$P(c_i)$中是Parity）**
   1. 计算并排序全部$d[i,j]$，得到**非递减序列**(因为有相等距离存在)$D=\cdots d[u,v]\leq d[k,l]$，Notes：计算**N!**次；
   2. 得到**唯一序列D**。例如，对相等距离的d，按照颜色索引值大小进行排序。**Notes**：**递减**序列
   3. 因为是**非递减唯一序列**，从小的距离$d[i,j]$开始分配奇偶
      * $if c_i \notin P$集合且$c_j \notin P$集合，分配相反的奇偶性给$c_i$与$c_j$,$P集合=P\cup${$c_i$}$\cup${$c_j$}
      * $if c_i \notin P$集合且$c_j \in P$集合，则将$P(c_i)=1-P(c_j)$**（奇偶性相反，$P(c)$代表的是奇偶性！）**，$P集合=P\cup${$c_i$}
      * $if c_i \in P$集合且$c_j \notin P$集合，则将$P(c_j)=1-P(c_i)$，$P集合=P\cup${$c_j$}
   (因为从小距离开始，所以最短距离先分配)

**总结2.4：**距离排序相比于亮度排序更加准确，最佳奇偶分配OPA使得颜色$c_i$与最近距离的颜色奇偶性一定不同。

### 2.5量化调制与格（略）

---

## 第3章 隐写分布特性的保持

**统计特征保持**：隐写中**尽可能的维持**载体的**统计分布**，并非要求完全一致。（一般指一阶载体分布的保持，尚未出现一阶以上的载体分布保持）


### 3.1 分布特性的识别（$\chi$<sup>2</sup>统计量：识别LSBR)

**回顾第二章-LSBR特性**：奇值只减，偶值只加

*例*：$h(2i)$表示采样点$2i$的直方图值，$h^x(2i)$表示为隐写后的值。不失一般性，**因**$h(2i)>h^x(2i)$，**所以**更多$h(2i)$变成了$2i+1$。
$\Rightarrow|h(2i)-h(2i+1)|\geq|h^x(2i+1)-h^x(2i)|$  *(造成对值分布更接近的原因)*

**提示：**
$|h(2i)+h(2i+1)|=|h^x(2i+1)+h^x(2i)|$*(值对**不外流**)*。
如 011-010互相改变，但不失一般性，原始的010的数量比原始的011数量多，所以，更多010变成了011，较少的011变成了010，造成了值对
$\Rightarrow$(教材P20)总有一个数量的增加多余减少

---
**$\chi^2$统计量**:服从$N(0,1)$高斯（正态）分布变量**平方和**的分币。

**记：**
+ $$y^x(i)=\frac {(h^x(2i)+h^x(2i+1))}{2};$$
其中，$y^x(i)$是**固定值**;
+ $$y(i)=h^x(2i)$$ 
其中，$y(i)$是**偶数修改后的值**。

**因为**在JPEG系数中0和1经常不用，**所以**$i$从1开始，有$i=1,2,...,d-1$。


<div align=center><img src="https://github.com/hmzbox/Study-notes/blob/master/Stego/images/3.1t.png" width="800"></div>

**Notes：**
1. $y^x(i)=y^*(i)$
2. $h^x(i)=h^*(i)$
3. $f(t)$为统计特征$t-\chi^2(d-1)$；**因为**在JPEG系数中0和1经常不用，**所以**$i$从1开始，有$i=1,2,...,d-1$。

---

**$\chi^2$分布特性分析**：$t$越小越可能存在隐写（因LSBR隐写算法会造成值对），设计阈值$\gamma$假设检验：

**漏检率**：$P_{\rm MD}(\gamma)=\int_\gamma^\infty f(t){\rm d}t$

**误检率**：$P_{\rm FA}(\gamma)=\int_0^\gamma f(t){\rm d}t$
$${正确率=1-\frac {漏检率+误检率}{2}}$$
由于含密载体$t$值非常小，因此，在实际中可以简单地用以下**统计量**完成检测:
$$\color{red}{p=\int_T^\infty f(t){\rm d}t=1-\int_0^T f(t){\rm d}t}$$

  其中，$T$表示根据输入图像计算得到的$t$值（$t$服从$\chi^2分布$），若$p$接近1，则存在隐写。


<div align=center><img src="https://github.com/hmzbox/Study-notes/blob/master/Stego/images/3.2p.png" width="400"></div>

$\color{red}{优点：}$对于**连续LSBR嵌入**的隐写非常有效，甚至可以**估计嵌入长度**

$\color{red}{缺点：}$对**不连续、非满负载LSBR**的分析性能**下降**很快。因此**Provos[3.1]**提出了“**滑动窗口$\chi^2$分析方法**”。

滑动窗口$\chi^2$分析方法：**随机选择**LSBR的**局部**进行$\chi^2$分析；$\chi^2$分析对局部密集嵌入区域仍然有效，使得对隐密载体的总体检测情况仍然呈现一定的异常。
[3.1]Provos N. Defending Against Statistical Steganalysis[C]//Usenix security symposium. 2001, 10: 323-336.

---

### 3.2 基于预留补偿区的分布恢复的方法（OutGuess隐写方案/软件）
**预留补偿区**：指不嵌入秘密信息的区域，此区域用于进行相应的补偿处理，从而恢复载体的分布特性。

**Outguess：** 是Provos设计的**JPEG图像隐写软件**，采用一阶统计特征保持方法，即LSBR嵌入后，调整**预留补偿区的LSB**(未嵌入秘密信息的区域)修复直方图。
+ 采用流密码RC4加密（异或）秘密信息，密钥双方已共享。


**OutGuess统计保持算法策略**
+ 针对一对邻值上的（总体）分布，算法不急着立即这么做修改邻值，它对各个值上修改次数进行记录，允许暂时在一定的次数(𝛽)内不修改，$\color{red}{目的是希望等待值对上修改需求的相互抵消}$；
+ 只有需要修改的次数超过设置的次数(𝛽)，才调用 𝑒𝑥𝑐ℎ𝐷𝐶𝑇函数基于前面区域进行修正；
+  但在修正失败的情况下，继续增加记录的修改次数；逐个系数
考察完毕后，最后对记录的需要修改次数再进行一轮处理$\color{red}不确保实现完全的修正$

$\color{red}OutGuess缺点：$预留补偿区域的分布恢复方法显著降低了嵌入效率，实际也更严重影响了二阶及以上阶的统计特性

源码地址（C++）：https://github.com/crorvick/outguess

---

### 3.3基于统计模型的方法（Model-based，MB）

**因为**基于预留补偿区域的分布恢复方法显著降低了嵌入效率，实际也更严重影响了二阶及以上阶的统计特性。

**所以**P.Salle提出了**基于模型（Model-based，MB）的隐写**，不会引起嵌入效率下降的问题，甚至还有提高。

**基本思想：**
+ 将载体信号建模为由两部分组成的随机变量**X=(Xdet, Xindet)**，其中Xdet和Xindet分别表示**确定的和非确定的部分**。隐写时，将只更改Xindet，从而保持它的分布不变，而且将确保隐密对象的非确定部分服从一定的分布模型。(在LSBR中，Xindet表示最后一个比特，非0即1)
+ 计算确定部分Xdet的概率分别，并根据假设的模型，计算非确定部分Xindet相对确定部分Xdet的条件概率。
+ 用熵解码器把均匀分布的秘密信息比特解码成服从上述条件概率分布的数据。
+ 用得到的数据替换Xindet，得到隐密对象。

基于MB模型隐写的嵌入过程如图:
<div align=center><img src="https://github.com/hmzbox/Study-notes/blob/master/Stego/images/3.4MB Embeding.png" width="600"></div>
提取过程如图所示：
<div align=center><img src="https://github.com/hmzbox/Study-notes/blob/master/Stego/images/3.5MB Extracting.png" width="600"></div>

Sallee将MB隐写框架应用于JPEG图像，提出了基于广义Cauchy模型的JPEG隐写，简称为Cauchy MB隐写。由于该算法是基于MB隐写框架设计的第一个隐写算法，也被称为MB1隐写。

与常用的高斯分布、广义拉普拉斯分布等相比，柯西分布能够更好地拟合AC系数直方图。尤其在柯西分布的尾部，能够对直方图进行非常好的拟合。
柯西分布具有封闭形式的概率分布函数，这使得能够方便地计算每个直方图方条的概率。

（2，2）的AC系数直方图及其拟合的柯西曲线如图所示：
<div align=center><img src="https://github.com/hmzbox/Study-notes/blob/master/Stego/images/3.6MB AC Cauchy.png" width="500"></div>

为了保持分组内各系数值所占的比例与用柯西分布拟合所得到的一致，MB1隐写将**秘密信息、每个系数值在分组内的相对位置及其条件概率传至熵解码器**，以解码得到隐写后的每个系数值在分组内的相对位置，从而得到隐密系数。
为了避免只改变图像的一部分而引起的与未含秘密信息部分的视觉差异，用于隐写信息的系数顺序将随机确定。

嵌入过程：
1) 给定JPEG格式的载体图像，加密后的信息。得到载体图像AC系数值的低精度直方图（系数值分组长度大于1）作为Xdet；
2) 采用最大似然拟合每个直方图的模型参数和；
3) 由系数值在各自分组内的偏移组成Xindet, 利用模型的条件概率密度函数计算每个系数的值所在的分组内所有可能的偏移相对其分组的条件概率；
4) 选取一个伪随机置乱，以确定系数顺序；
5) 以4)中确定的顺序将秘密信息、系数偏移、3)中计算得到的条件概率传至一个非自适应算术解码器。解码得到含秘密信息的偏移；
6) 根据系数值所在的分组和解码得到的系数值在分组内的偏移，得到隐写后的系数。
提取过程：
1)-4)与嵌入过程相同。
5) 以4)中确定的顺序将系数偏移、3)中计算得到的偏移概率传至非自适应算术编码器。编码得到秘密信息。

之后，考虑到MB1隐写将增大图像8×8块间的不连续性，即分块效应，Phil Sallee只在至多一半的非0系数上进行MB1隐写，然后对剩下的非0系数值进行调整，以减小分块效应，从而提出MB2隐写。

源码地址（matlab）：http://www.pudn.com/Download/item/id/1748692.html

---
### 3.4基于调整修改方式的隐写(F3/F4/F5)
#### F3隐写
论文（2-F5-A steganographic algorithm High capacity despite better steganalysis(2001,IH,CR1118)
为了克服Jsteg隐写不能抵抗卡方分析的缺陷，F3隐写对JSteg隐写进行了改进，具体嵌入策略如下：
**嵌入过程：**
1. 信息嵌入时，若JPEG系数的LSB与要嵌入的秘密信息比特**相同**，**则不做改动**；**否则**，JPEG系数的**绝对值减1**；
2. 秘密信息嵌入在非0的JPEG系数上，为0的系数不嵌入信息。当在绝对值为1的系数上嵌入比特0时，会产生新的0系数，则此次嵌入无效，在下一个系数中重新嵌入。

**提取过程：**
对F3隐写后的秘密信息提取时，只要将**非0的JPEG系数的LSB提取**。

嵌入方式如图所示：
<div align=center><img src="https://github.com/hmzbox/Study-notes/blob/master/Stego/images/3.7 F3.png" width="600"></div>

嵌入前和隐写后对比图：
<div align=center><img src="https://github.com/hmzbox/Study-notes/blob/master/Stego/images/3.8 F3 Histogram.png" width="400"></div>

**F3隐写的特点：**
1. 保持了JPEG系数统计分布关于0的对称性；
2. 嵌入时，当绝对值为1的JPEG系数修改为0时，由于提取算法不能分辨这个0是消息嵌入产生的0和未使用的0，所以嵌入算法要往后寻找一个非0偶数，或寻找一个奇数并将其绝对值减1修改成非0偶数；
3. 嵌入了更多的零，这一异常可以被利用于隐写分析。

源码地址为（python）：https://github.com/uuuup/F3_stegano
（matlab）http://www.pudn.com/Download/item/id/3190169.html

---
#### F4隐写