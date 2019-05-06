# learn_modeling

一、监督算法：

1.1 分类算法(线性和非线性)

encoder{  
        感知机（神经网络）  
        CNN  
        RNN  
        }

概率{  
        朴素贝叶斯（NB）  
        Logistic Regression（LR）
        最大熵MEM（与LR同属于对数线性分类模型）  
    }
    
[GLM与逻辑回归](https://www.jianshu.com/p/9c61629a1e7d)

广义线性模型，y=ax+随机变量，y也是一个随机变量。如果y服从正态分布，那么E(y)可以用x的线性回归来表示。如果y服从伯努利分布（0或者1），那么E(y)可以用x的逻辑回归来表示。
    
支持向量机(SVM)

[work/svm.txt](https://github.com/arfu2016/work/blob/master/svm.txt)

SVM:

1. 损失函数与logistic regression的损失函数有类似之处：把log函数变成线性分段函数

2. maximal margin and support vectors

画两条平行线分别过两个类别的点，使得平行线之间的距离最大，两条平行线中间的线（middle line）就是分界线

3. outlier的处理

如果有个别outlier出现在了middle line的另一侧（和大多数同一类的点不在分界线的同一侧），可以用该outlier到middle line的距离做惩罚（或者用该outlier到本方margin line的距离做惩罚），之前乘一个惩罚因子。

决策树(ID3、CART、C4.5)

[nlp/nlp_models/decision_tree/](https://github.com/arfu2016/nlp/tree/master/nlp_models/decision_tree)

assembly learning{  
        Boosting{  
            Gradient Boosting{  
                GBDT  
                xgboost（传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）；xgboost是Gradient Boosting的一种高效系统实现，并不是一种单一算法。）
            }  
            AdaBoost  
        }     
        Bagging{  
            随机森林  
        }  
        Stacking  
    }  

1.2 概率图模型

假设你有许多小明同学一天内不同时段的照片，从小明提裤子起床到脱裤子睡觉各个时间段都有（小明是照片控！）。现在的任务是对这些照片进行分类。比如有的照片是吃饭，那就给它打上吃饭的标签；有的照片是跑步时拍的，那就打上跑步的标签；有的照片是开会时拍的，那就打上开会的标签。问题来了，你准备怎么干？一个简单直观的办法就是，不管这些照片之间的时间顺序，想办法训练出一个多元分类器。就是用一些打好标签的照片作为训练数据，训练出一个模型，直接根据照片的特征来分类。例如，如果照片是早上6:00拍的，且画面是黑暗的，那就给它打上睡觉的标签;如果照片上有车，那就给它打上开车的标签。这样可行吗？

乍一看可以！但实际上，由于我们忽略了这些照片之间的时间顺序这一重要信息，我们的分类器会有缺陷的。举个例子，假如有一张小明闭着嘴的照片，怎么分类？显然难以直接判断，需要参考闭嘴之前的照片，如果之前的照片显示小明在吃饭，那这个闭嘴的照片很可能是小明在咀嚼食物准备下咽，可以给它打上吃饭的标签；如果之前的照片显示小明在唱歌，那这个闭嘴的照片很可能是小明唱歌瞬间的抓拍，可以给它打上唱歌的标签。所以，为了让我们的分类器能够有更好的表现，在为一张照片分类时，我们必须将与它相邻的照片的标签信息考虑进来。

![bayes_markov](images/bayes_markov.png)

分词

python: jieba

c++: baidu / lac, 哈工大HIT-SCIR / ltp

java: hankcs / HanLP, 中科院NLPIR-team / NLPIR

naive bayes assumption: 分词后的句子的概率等于第一个词的概率乘以剩余sequence的概率，剩余sequence的概率用递归来算。之所以说是naive bayes assumption，是因为假设剩余sequence的概率与第一个词是什么是无关的，也就是独立性假设（实际上是有关的，哪怕是近似的markov assumption，也胜过naive bayes assumption，但naive bayes assumption在建模时最为简单）。

词的概率就是在语料中数词频，和计算tf-idf中的idf比较类似。语料越大，词频越准。需要注意的是，由于语料总是有限，某些词的词频统计出来会是0，但实际上却有很小的概率，这时候需要用smoothing的技术来估算这个很小的概率。

简单讲，naive bayes的语言模型是每个词的概率相乘。

拼写错误的纠正

方法一：用word window来看，用词向量的bag of words向量来预测中间的词，看预测值和写出来的词是否一致，看写出来的词的概率有多高，如果很低，就是写错了。

方法二：用贝叶斯法则，p(c|w) = p(w|c)乘以p(c)除以p(w)，对于任何的c，p(w) 都是定值。p(c)可以从正确语料中统计出来，p(w|c)需要从人工修改的语料中统计。

HMM

HMM是贝叶斯网络，是单向网络。白色的节点是要标记的序列，灰色的节点是观测到的字符序列。HMM模型很漂亮，但是，参数较少，建模和预测能力较弱，假设也太强，只有当前字符和上一个字符才会直接影响当前标记。

markov assumption: 某个标记是什么，只与前一个标记有关，与再之前的标记没有关系。

简单讲，n-gram（比如2-gram）的语言模型是每个词的条件概率（在固定前一个词的前提下）相乘。

HMM需要估算一个label到另一个label的转移参数，也要估算一个label下出现某个词的概率参数，使用（测试集）时是用维特比算法来求出全局最优的sequence（概率最大）.

对于求sequence的概率，笨办法是每一步都看n个词，如果有m步的话，总的复杂度就是n的m次方，得到全局最优，但复杂度太高；维特比算法是运用动态规划来求全局最优，但只能针对特定的模型，比如HMM模型，复杂度没n的m次方那么高；Beam search是局部最优，每次只看最优的几个词，比如最优的5个词，最后从中选出局部最优，每一步看的都是5个词（从之前的25个路径当中选最优的5个），复杂度是5乘以n；贪婪算法是Beam search的特例，每次只看最优的一个词，最后得到一个序列，复杂度是n，最后一步也只要看n个词就好，之前的路径已经确定。

MEMM（最大熵马尔科夫）

MEMM能引入多个特征，建模能力相比HMM得到提高，假设也减弱，整个序列上的字符都能直接影响当前标记。MEMM是马尔科夫网络，只是归一化的设计有问题，会导致在预测标记时，更倾向于选择下一个状态更加集中的状态，而这种选择显然是没有道理的，所以，在这种情况下容易发生标记错误。

CRF（命名实体识别）

CRF是马尔科夫网络，是双向网络（或者说是无向图）。节点相互之间都有影响，两个方向的影响系数可能会有不同。双向网络比单向网络自由度更高，建模能力更强。Linear-chain CRF是不成环的。马尔科夫性是是保证或者判断概率图是否为概率无向图的条件。马尔科夫随机场指的就是马尔科夫网络。

对于CRF，白色节点是某个token_i的与label有关的一组特征，可以认为白色节点也是关于多个特征的马尔科夫网络。

这里，我们的特征函数仅仅依靠当前单词的标签和它前面的单词的标签对标注序列进行评判，这样建立的CRF也叫作线性链CRF，这是CRF中的一种简单情况。也就是说，token_i的label只会和token_(i-1)的label以及token_(i+1)的label发生关系，所以是线性的。

CRF长句和短句输入进去没有太大区别，因为模板中一般只看周围几个词，也就是看word window，不像lstm，每一层都要训练系数，太长的话，再做back propagation的时候，前面的系数很容易梯度消失导致不再拟合。但是，在预测的时候，用的维特比算法，句子越长，要考虑的情况越多，句子太长的话，内存可能爆掉。

CRF的建模公式是：

![CRF_formula](images/CRF_formula.png)

其中，O是给定的观测序列，比如，一句话的词组成的序列。I是需要做的标记，比如BIO标记。i表示当前关注的token，也就是关注当前label和上一个label的那个token。k表示当前特征，一共M个特征，每个特征执行一定的限定作用。Z(o)是用来归一化的，为什么？想想LR（logistic regression）以及softmax为何有归一化呢，一样的嘛，形成概率值。需要注意的是，这里的归一化是类似softmax的归一化，是全局归一化，对i的求和以及对k的求和一起归一化。

给定一个序列s，有多种标记的方法l。每种l的得分公式就是上面CRF建模公式的分子：

![CRF_numerator](images/crf_numerator.png)

分母用来对多种l序列进行归一化：

![CRF_dominator](images/crf_dominator.png)
    
几个特征函数的例子（此处的标注是词性）:

当l_i是“副词”并且第i个单词以“ly”结尾时，我们就让f1 = 1，其他情况f1为0。不难想到，f1特征函数的权重λ1应当是正的。而且λ1越大，表示我们越倾向于采用那些把以“ly”结尾的单词标注为“副词”的标注序列。f1只和l_i以及s中的一个位置（此处碰巧是第i个位置）有关，与l_(i-1)无关。在crf++中用unigram来表示。

如果i=1，l_i=动词，并且句子s是以“？”结尾时，f2=1，其他情况f2=0。同样，λ2应当是正的，并且λ2越大，表示我们越倾向于采用那些把问句的第一个单词标注为“动词”的标注序列。f2只和l_i以及s中的一个位置(此处是句子的最后一个位置)有关，与l_(i-1)无关。在crf++中用unigram来表示。

当l_i-1是介词，l_i是名词时，f3 = 1，其他情况f3=0。λ3也应当是正的，并且λ3越大，说明我们越认为介词后面应当跟一个名词。f3只和l_i以及l_(i-1)有关。在crf++中用bigram来表示。

如果l_i和l_i-1都是介词，那么f4等于1，其他情况f4=0。这里，我们应当可以想到λ4是负的，并且λ4的绝对值越大，表示我们越不认可介词后面还是介词的标注序列。f4只和l_i以及l_(i-1)有关。在crf++中用bigram来表示。

CRF++

在CRF++中，是先定义模板，然后模板自动产生大量特征。每一条模板将在每一个token处生产若干个特征函数。CRF++的模板（template）有U系列（unigram）、B系列(bigram)，U系列只关注当前位置的label，B系列关注当前位置和之前位置的label.

    U00:%x[-2,0]
    U01:%x[-1,0]
    U02:%x[0,0]
    U03:%x[1,0]
    U04:%x[2,0] 
    U05:%x[-2,0]/%x[-1,0]/%x[0,0]
    U06:%x[-1,0]/%x[0,0]/%x[1,0]
    U07:%x[0,0]/%x[1,0]/%x[2,0]
    U08:%x[-1,0]/%x[0,0]
    U09:%x[0,0]/%x[1,0] 

    B

从U00到U04表示，对于每个token，都要关注之前两个字符以及之后两个字符（是或的关系）对当前label的影响。而U05到U09关注的是多个字符（是且的关系）对当前label的影响。B关注的是上面的规则再加上之前的标记对当前标记的影响。从U00到U04，看某一个位置的影响，更类似边缘分布，从U05到U09，看两个或多个位置联合起来的影响（不只是某一个位置是什么，同时对其他位置有要求），更类似联合分布。

对于unigram模板，经常只看之前和之后两个字符，在标记联合分布式，经常就是标U05到U09这个样子，看两个或者三个位置的联合影响。对于bigram模板，经常只写一个B，就是unigram模板产生的特征在多看一下前一个位置的label。

对于unigram模板，The number of feature functions generated by a template amounts to (L * N), where L is the number of output classes (labels) and N is the number of unique string expanded from the given template.

对于bigram模板，this type of template generates a total of (L * L * N) distinct features, where L is the number of output classes and N is the number of unique features generated by the templates. When the number of classes is large, this type of templates would produce a tons of distinct features that would cause inefficiency both in training/testing.
    
1.3 回归预测

线性回归

神经网络

树回归

[Imylu/work5/](https://github.com/arfu2016/Imylu/tree/master/work5)

Ridge岭回归

Lasso回归

二、非监督：

2.1 聚类

(1) 基础聚类  
    
K—means  
    
[nlp/nlp_models/kmeans/](https://github.com/arfu2016/nlp/tree/master/nlp_models/kmeans)   
    
二分k-means   
K中值聚类  
GMM聚类  
    
(2) 层次聚类

(3) 密度聚类

(4) 谱聚类

2.2 主题模型

pLSA

LDA隐含狄利克雷分析

2.3 关联分析 (correlation analysis)，对于关联分析，都是x，没有y

Person correlation coefficient

Apriori算法

FP-growth算法    
    
2.4 降维

PCA算法

[nlp/nlp_models/pca/](https://github.com/arfu2016/nlp/tree/master/nlp_models/pca)

SVD算法

LDA线性判别分析

LLE局部线性嵌入    

2.5 异常检测

2.6 collaborative filtering (协同过滤)

三、半监督学习：仅有少数种子样本

bootstrapping用于关系抽取

四、迁移学习与复杂的深度学习模型

五、模型的内在机制和物理意义

周志华arxiv论文：learning with interpretable structure from rnn

六、统计

t-test

[work/statistics.txt](https://github.com/arfu2016/work/blob/master/statistics.txt)
[t-test.txt](https://github.com/arfu2016/nlp/blob/master/nlp_models/algorithm/statistics/t-test.txt)

anova

[anova.txt](https://github.com/arfu2016/nlp/blob/master/nlp_models/algorithm/statistics/anova.txt)

贝叶斯统计

[beysian.txt](https://github.com/arfu2016/nlp/blob/master/nlp_models/algorithm/statistics/beysian.txt)

pearson correlation

[nlp/nlp_models/pearson/](https://github.com/arfu2016/nlp/tree/master/nlp_models/pearson)

confusion_matrix

[nlp/nlp_models/confusion_matrix/](https://github.com/arfu2016/nlp/tree/master/nlp_models/confusion_matrix)

七、问题

[25_fun_questions](https://github.com/arfu2016/nlp/tree/master/nlp_models/algorithm/machine_learning/questions)

[feature_scaling](https://github.com/arfu2016/nlp/tree/master/nlp_models/feature_scaling)

[gradient_descent](https://github.com/arfu2016/nlp/tree/master/nlp_models/gradient_descent)

[lagrange_interpolation](https://github.com/arfu2016/nlp/tree/master/nlp_models/lagrange_interpolation)


