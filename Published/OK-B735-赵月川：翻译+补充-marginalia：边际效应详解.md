# B735：翻译+补充-marginalia：边际效应详解

> 编译：赵月川 (中国社会科学院大学)    
> 邮箱：<zhaoyuechuan@ucass.edu.cn>    

&emsp;

> 本文主要翻译自如下主义，并进行了适当的补充和调整.      
> **Source：** https://www.andrewheiss.com/blog/2022/05/20/marginalia/

&emsp;

## 1. 如何定义“边际”？

自19世纪70年代经济学 **“边际革命”** （Marginal Revolution）以来，新古典经济学乃至现代经济学逐步走向理性主义、实证主义和科学主义。

在经济学学习中，我们经常遇到 **marginal**（边际） 这个词，但要准确区分和说明诸如边际均值（**marginal means**）、边际斜率（**marginal slopes**）、条件边际效应（**conditional marginal effects**）以及 均值处的边际效应（**marginal effects at the mean**）等相关概念，往往并不容易。

微观经济中⼀个常⻅的问题是如何在预算约束下实现个人效⽤的最⼤化。边际效用则表示“增加一个单位商品所带来的效用变化”。在统计与回归分析中，边际效应可理解为某一解释变量变化所带来的被解释变量的即时变化率，通常以偏导数形式呈现。

> 若因变量 $y$ 关于自变量 $x$ 的函数为：$y = f(x)$，则边际定义为：$\frac{\partial y}{\partial x} = f'(x)$

当$y = -0.5x^2 + 5x + 5$，很容易计算出$\frac{\partial y}{\partial x} = -x +5$。下图给出了该函数部分点的切线斜率。若将该函数视为某种效用函数, $x=0$ 处的边际效用为 5，意味着在此点增加一单位的x，其他条件不变的情况下，效用将增加 5 单位。随着 x 增加，边际效用逐渐递减。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefinedB735-%E8%BE%B9%E9%99%85%E6%96%9C%E7%8E%87-%E8%B5%B5%E6%9C%88%E5%B7%9D.png)

然⽽，在实际应⽤中，“边际效用”这⼀术语的含义远不⽌于偏导数。例如，在随机对照试验中，处理组和对照组的均值之差通常被称为边际效应（有时也称为条件效应，甚⾄条件边际效应）。接下来，我将基于作者（Andrew Heiss）的例子详细介绍“边际”的计算，帮助我们更好地理解“边际”。

## 2. 统计学中如何计算“边际”？

作者基于V-Dem（Varieties of Democracy）和世界银行的数据研究两个问题：
> 尊重个⼈⼈权的国家是否往往腐败程度更低?
> 腐败⼜如何影响⼀个国家在选举透明度⽅⾯的选择？

其中，V-Dem（Varieties of Democracy）是一个多国数据，可以通过R命令获取。首次加载时可能需要等待几秒。

```r
remotes::install_github("vdeminstitute/vdemdata")
library(vdemdata) #必须手动将 vdem 数据载入
data("vdem") 
```
- 核心变量为**腐败程度指数**，是一个[0,1]范围内的连续变量。
- 被解释变量为**选举透明度**，作者将其处理为一个二元变量，0表示选举不透明，1表示选举存在不同程度的披露。
- 其他关键变量：**民主指数**（civil liberaties index）。其余变量描述可参看链接原文。

（另外，数据的清洗过程也可参看链接原文，我在此仅放入计算“边际”的相关代码。）

### 2.1 OLS回归

以OLS回归作为参照，这里给出民主指数和腐败程度指数的线性拟合。模型设定为：

$$
E (腐败程度指数|民主指数) = \beta_0 + \beta_1 民主指数 （1）
$$

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefinedB735-ols1-%E8%B5%B5%E6%9C%88%E5%B7%9D.png)

通过以下命令，我们可以计算出腐败程度指数对民主指数的边际效应（$\beta_1$）为-0.812，即民主指数每增加一个单位，腐败程度会**平均**下降0.812。因为数据在实时更新，所以我重新运行命令得到的结果与链接原文存在些许差异。

```r
model_simple <- lm(public_sector_corruption ~ civil_liberties,
+                    data = corruption)
> tidy(model_simple)
# A tibble: 2 × 5
  term            estimate std.error statistic  p.value
  <chr>              <dbl>     <dbl>     <dbl>    <dbl>
1 (Intercept)      102.       5.34        19.2 3.38e-44
2 civil_liberties   -0.812    0.0732     -11.1 7.66e-22
```

进而在原模型中引入民主指数的平方项：

$$
E (腐败程度指数|民主指数) = \beta_0 + \beta_1 民主指数 + \beta_2 民主指数^2 （2）
$$

拟合图形显示如下，可以发现此时的斜率会受到民主指数取值的影响。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefinedB735-ols2-%E8%B5%B5%E6%9C%88%E5%B7%9D.png)

对公式（2）求偏导可以得到：

$$
\frac{\partial 腐败程度指数}{\partial 民主指数}  = \beta_1 + \beta_2 民主指数 （3）
$$

通过以下命令可以得到参数取值，并给民主指数赋值为25、55、80，可以得到切点斜率分别为0.594、-0.587、-1.572。这种情况下，我们解释系数的经济含义时需要更加严谨。

```r
> model_sq <- lm(public_sector_corruption ~ civil_liberties + I(civil_liberties^2), data = corruption)
> tidy(model_sq)
# A tibble: 3 × 5
  term                 estimate std.error statistic      p.value
  <chr>                   <dbl>     <dbl>     <dbl>        <dbl>
1 (Intercept)           53.0     10.0          5.29 0.000000375 
2 civil_liberties        1.28     0.376        3.40 0.000849    
3 I(civil_liberties^2)  -0.0178   0.00314     -5.65 0.0000000667

 # Extract the two civil_liberties coefficients
> civ_lib1 <- tidy(model_sq) |> filter(term == "civil_liberties") |> pull(estimate)
> civ_lib2 <- tidy(model_sq) |> filter(term == "I(civil_liberties^2)") |> pull(estimate)

> # Make a little function to do the math
> civ_lib_slope <- function(x) civ_lib1 + (2 * civ_lib2 * x)
> civ_lib_slope(c(25, 55, 80))
[1]  0.3885366 -0.6766313 -1.5642711
> ## [1]  0.594 -0.587 -1.572

> model_sq |> 
+   slopes(newdata = datagrid(civil_liberties = c(25, 55, 80)),
+          eps = 0.001) # 通过这个命令可以获得切点斜率以及相应的p值
 civil_liberties Estimate Std. Error      z Pr(>|z|)    S   2.5 % 97.5 %
              25    0.389     0.2230   1.74   0.0814  3.6 -0.0485  0.826
              55   -0.677     0.0715  -9.47   <0.001 68.2 -0.8167 -0.537
              80   -1.564     0.1486 -10.53   <0.001 83.7 -1.8556 -1.273
Term: civil_liberties
Type: response
Comparison: dY/dX

> model_sq |> 
+   emtrends(~ civil_liberties, var = "civil_liberties",
+            at = list(civil_liberties = c(25, 55, 80)),
+            delta.var = 0.001) |> 
+   test() # 通过这个命令也可以获得切点斜率以及相应的p值
 civil_liberties civil_liberties.trend     SE  df t.ratio p.value
              25                 0.389 0.2230 168   1.743  0.0831
              55                -0.677 0.0715 168  -9.469  <.0001
              80                -1.564 0.1490 168 -10.492  <.0001
```

需要注意的是，虽然从这个例子上看，`slopes`命令和`emtrends`命令的结果接近一致，但两者的计算逻辑是不同的。

**`slopes`命令计算的是平均边际效应（Average marginal effects）。**下图给出了数据处理过程。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefinedB735-slope-%E8%B5%B5%E6%9C%88%E5%B7%9D.png)

此图可以与以下命令对应分析。`slopes`命令是先估计了每行数据的边际效应，然后计算了全样本的边际效应平均值。这里作者强调，这里计算的结果并不与前文一致，仅是在探索平均边际效应的计算过程。

```r
> mfx_sq <- slopes(model_sq)
> head(mfx_sq)

 Estimate Std. Error      z Pr(>|z|)     S 2.5 % 97.5 %
    -1.20     0.0964 -12.47   <0.001 116.2 -1.39  -1.01
    -1.81     0.1882  -9.60   <0.001  70.0 -2.17  -1.44
    -2.14     0.2449  -8.75   <0.001  58.7 -2.62  -1.66
    -2.11     0.2388  -8.82   <0.001  59.7 -2.58  -1.64
    -1.97     0.2161  -9.13   <0.001  63.6 -2.40  -1.55
    -1.67     0.1655 -10.07   <0.001  76.8 -1.99  -1.34

Term: civil_liberties
Type: response
Comparison: dY/dX

> mfx_sq |> 
+   group_by(term) |> 
+   summarize(avg_slope = mean(estimate))
# A tibble: 1 × 2
  term            avg_slope
  <chr>               <dbl>
1 civil_liberties     -1.17

> avg_slopes(model_sq)
 Estimate Std. Error     z Pr(>|z|)     S 2.5 % 97.5 %
    -1.17     0.0922 -12.7   <0.001 119.9 -1.35 -0.988
Term: civil_liberties
Type: response
Comparison: dY/dX
```

**`emtrends`命令计算的则是均值边际效应（marginal effects at the mean）。**下图同样给出了数据处理过程。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefinedB735-trend-%E8%B5%B5%E6%9C%88%E5%B7%9D.png)

结合以下命令，我们就可以理解`emtrends`命令的计算逻辑，它是通过先对协变量求平均值，再计算协变量平均值出的边际效应。

```r
> avg_civ_lib <- mean(corruption$civil_liberties)
> avg_civ_lib
[1] 68.85497

> civ_lib_fitted <- model_sq |> 
+   augment(newdata = tibble(civil_liberties = c(avg_civ_lib, avg_civ_lib + 0.001)))
> civ_lib_fitted
# A tibble: 2 × 2
  civil_liberties .fitted
            <dbl>   <dbl>
1            68.9    56.7
2            68.9    56.7

> (civ_lib_fitted[2,2] - civ_lib_fitted[1,2]) / 0.001
    .fitted
1 -1.168578

> model_sq |> 
+   avg_slopes(newdata = "mean")
 Estimate Std. Error     z Pr(>|z|)     S 2.5 % 97.5 %
    -1.17     0.0922 -12.7   <0.001 119.9 -1.35 -0.988
Term: civil_liberties
Type: response
Comparison: dY/dX
```

需要说明的是，不论是边际效应的平均值还是平均值的边际效应，我们都很难在现实生活中对应起来，毕竟很少样本刚刚好处于平均值处。这种**平均边际**也仅在OLS回归里存在。如果更换成logit模型，“边际”就不会是这样的计算过程了。

### 2.2 logit回归

回到作者关心的研究问题：腐败⼜如何影响⼀个国家在选举透明度⽅⾯的选择？由于被解释变量是一个二元变量，因而设定logit模型。我们画出两个变量的拟合图。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefinedB735-logit-%E8%B5%B5%E6%9C%88%E5%B7%9D.png)

我们通过以下命令估计模型并计算AME和MEM。

```r
> model_logit <- glm(
There were 50 or more warnings (use warnings() to see the first 50)
+   disclose_donations ~ public_sector_corruption,
+   family = binomial(link = "logit"),
+   data = corruption
+ )
> 
> tidy(model_logit)
# A tibble: 2 × 5
  term                     estimate std.error statistic  p.value
  <chr>                       <dbl>     <dbl>     <dbl>    <dbl>
1 (Intercept)                1.82     0.370        4.91 8.88e- 7
2 public_sector_corruption  -0.0637   0.00929     -6.86 6.66e-12

> model_logit |> 
+   avg_slopes()
 Estimate Std. Error     z Pr(>|z|)     S    2.5 %   97.5 %
 -0.00816    0.00028 -29.1   <0.001 617.6 -0.00871 -0.00762
Term: public_sector_corruption
Type: response
Comparison: dY/dX

> model_logit |> 
+   emtrends(~ public_sector_corruption, 
+            var = "public_sector_corruption", 
+            regrid = "response")
 public_sector_corruption public_sector_corruption.trend      SE  df asymp.LCL asymp.UCL
                     46.3                        -0.0117 0.00159 Inf   -0.0148   -0.0086
Confidence level used: 0.95 
```

通过计算，可以发现在logit回归中，腐败程度指数的系数为-0.0637，AME为-0.00816，MEM为-0.0117。我们可以解释这一结果为：一个国家的腐败程度指数每增加一个单位，选举透明的概率将平均下降0.816%，而在均值处的选举透明概率将下降1.17%。这与OLS结果不同的是，此时AME和MEM相差巨大。且AME的置信区间比MEM的置信区间小很多，这是因为AME的是根据全样本边际效应平均得到的，而MEM则只计算了协变量均值处的边际效应。

进而我们在logit模型中引入腐败程度指数的平方项，观察结果会有怎样的变化。

```r
> model_logit_fancy <- glm(
+   disclose_donations ~ public_sector_corruption + I(public_sector_corruption^2) + 
+     polyarchy + log_gdp_percapita + public_sector_corruption * region,
+   family = binomial(link = "logit"),
+   data = corruption
+ )

> model_logit_fancy |> 
+   avg_slopes()
                     Term                                                           Contrast Estimate Std. Error      z Pr(>|z|)   S    2.5 %   97.5 %
 log_gdp_percapita        dY/dX                                                               0.03143    0.03758  0.836  0.40306 1.3 -0.04224  0.10509
 polyarchy                dY/dX                                                               0.00446    0.00167  2.668  0.00763 7.0  0.00118  0.00774
 public_sector_corruption dY/dX                                                              -0.00455    0.00175 -2.595  0.00946 6.7 -0.00799 -0.00111
 region                   Asia and Pacific - Eastern Europe and Central Asia                 -0.08372    0.07610 -1.100  0.27127 1.9 -0.23288  0.06543
 region                   Latin America and the Caribbean - Eastern Europe and Central Asia  -0.18232    0.08590 -2.123  0.03379 4.9 -0.35067 -0.01397
 region                   Middle East and North Africa - Eastern Europe and Central Asia     -0.12767    0.10600 -1.204  0.22842 2.1 -0.33542  0.08009
 region                   Sub-Saharan Africa - Eastern Europe and Central Asia               -0.15246    0.11651 -1.309  0.19066 2.4 -0.38081  0.07588
 region                   Western Europe and North America - Eastern Europe and Central Asia -0.21933    0.07319 -2.997  0.00273 8.5 -0.36278 -0.07589
Type: response

> model_logit_fancy |> 
+   emtrends(~ public_sector_corruption, 
+            var = "public_sector_corruption", 
+            regrid = "response")
NOTE: Results may be misleading due to involvement in interactions
 public_sector_corruption public_sector_corruption.trend      SE  df asymp.LCL asymp.UCL
                     46.3                        -0.0064 0.00256 Inf   -0.0114  -0.00139
Results are averaged over the levels of: region 
Confidence level used: 0.95 

> model_logit_fancy |> 
+   avg_slopes(newdata = "mean") ##通过设定均值，我们通过slope命令得到MEM
                     Term                                                           Contrast Estimate Std. Error      z Pr(>|z|)   S     2.5 %   97.5 %
 log_gdp_percapita        dY/dX                                                               0.05456    0.07922  0.689  0.49100 1.0 -1.01e-01  0.20983
 polyarchy                dY/dX                                                               0.00775    0.00395  1.962  0.04975 4.3  8.33e-06  0.01549
 public_sector_corruption dY/dX                                                              -0.00283    0.00378 -0.749  0.45405 1.1 -1.02e-02  0.00458
 region                   Asia and Pacific - Eastern Europe and Central Asia                 -0.28085    0.19956 -1.407  0.15932 2.6 -6.72e-01  0.11028
 region                   Latin America and the Caribbean - Eastern Europe and Central Asia  -0.26663    0.15978 -1.669  0.09517 3.4 -5.80e-01  0.04653
 region                   Middle East and North Africa - Eastern Europe and Central Asia     -0.33400    0.21500 -1.553  0.12031 3.1 -7.55e-01  0.08740
 region                   Sub-Saharan Africa - Eastern Europe and Central Asia               -0.22071    0.17160 -1.286  0.19836 2.3 -5.57e-01  0.11561
 region                   Western Europe and North America - Eastern Europe and Central Asia -0.44846    0.15100 -2.970  0.00298 8.4 -7.44e-01 -0.15250
Type: response
```

从这个结果来看，腐败程度指数的AME为-0.0045，而MEM（根据slope命令设定均值得到）为-0.00283且不显著。

## 3. 其他情况的“边际”

### 3.1 群体平均边际效应

为了让研究更加丰富，我们并不局限在AME和MEM。在这个例子中，如果想要知道不同地区国家的边际效应，应该怎么做呢？在AME的基础上，引入**群体平均边际效应（Group average marginal effects）**，根据群体特征计算平均边际效应，逻辑如下图所示：

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefinedB735-GME-%E8%B5%B5%E6%9C%88%E5%B7%9D.png)

通过以下命令可以计算GME。

```r
mfx_logit_fancy <- model_logit_fancy |> 
  slopes(variables = "public_sector_corruption")

head(mfx_logit_fancy)

mfx_logit_fancy |> 
  group_by(region) |> 
  summarize(region_ame = mean(estimate))

model_logit_fancy |> 
  slopes(variables = "public_sector_corruption",
         by = "region")

model_logit_fancy |> 
  emtrends(~ public_sector_corruption + region,
           var = "public_sector_corruption", regrid = "response")

> model_logit_fancy |> 
+   avg_slopes(variables = "public_sector_corruption",
+              newdata = datagrid(region = levels(corruption$region)),
+              by = "region")

                           region Estimate Std. Error      z Pr(>|z|)   S   2.5 %   97.5 %
 Eastern Europe and Central Asia  -0.01068    0.00626 -1.705   0.0883 3.5 -0.0230  0.00160
 Latin America and the Caribbean  -0.00285    0.00431 -0.660   0.5093 1.0 -0.0113  0.00561
 Middle East and North Africa     -0.00828    0.00721 -1.148   0.2510 2.0 -0.0224  0.00586
 Sub-Saharan Africa               -0.00283    0.00378 -0.749   0.4540 1.1 -0.0102  0.00458
 Western Europe and North America -0.00104    0.00278 -0.374   0.7086 0.5 -0.0065  0.00442
 Asia and Pacific                 -0.01279    0.00627 -2.041   0.0412 4.6 -0.0251 -0.00051

Term: public_sector_corruption
Type: response
Comparison: dY/dX
```

因篇幅有限，我这里仅展示最后一行命令的结果，GME的好处在于我们得到了不同地区的边际效应。例如，东欧和中亚（Eastern Europe and Central Asia）地区的边际效应为-0.01068，即在该地区，腐败程度指数每增加一个单位，选举透明的概率会下降0.01068。

### 3.2 特定值处的边际效应

更具体地，如果我们想要知道**特定值处边际效应（Marginal effects at user-specified or representative values）**，我们可以通过以下三个步骤进行计算。

> 第一步：构造“数据网格”用于模拟不同值的场景

这里我不再汇报结果，而仅是提供命令并作出解释。

```r
regions_to_use <- c("Western Europe and North America", 
                    "Latin America and the Caribbean",
                    "Middle East and North Africa")

expand_grid(public_sector_corruption = c(20, 80),
            region = regions_to_use,
            polyarchy = mean(corruption$polyarchy),
            log_gdp_percapita = mean(corruption$log_gdp_percapita))

modelr::data_grid(corruption,
                  public_sector_corruption = c(20, 80),
                  region = regions_to_use,
                  .model = model_logit_fancy)

datagrid(model = model_logit_fancy,
         public_sector_corruption = c(20, 80),
         region = regions_to_use)

ref_grid(model_logit_fancy,
         at = list(public_sector_corruption = c(20, 80),
                   region = regions_to_use))
```

其中，`expand_grid()` 手动组合变量取值；`data_grid()` 自动补全模型变量，适合预测；`datagrid()` 为 `marginaleffects` 专用，自动适配模型、支持边际效应与反事实分析；`ref_grid()` 用于 `emmeans`，生成特定值下的参考网格，适合趋势与对比分析。

> 第二步：插入模型并使用 `slopes` 或 `emtrends` 计算边际效应

当通过第一步获得网格数据后，我们所计算的边际效应更类似于MEM，因为并没有用到全样本。以下命令直接给出了边际效应的结果，即切点斜率。

```r
> model_logit_fancy |> 
+   slopes(variables = "public_sector_corruption",
+          newdata = datagrid(public_sector_corruption = c(20, 80),
+                             region = regions_to_use))
 public_sector_corruption                           region  Estimate Std. Error      z Pr(>|z|)   S     2.5 %   97.5 %
                       20 Western Europe and North America -1.74e-02   1.34e-02 -1.303    0.192 2.4 -0.043633 0.008780
                       20 Latin America and the Caribbean  -1.04e-03   6.21e-03 -0.168    0.867 0.2 -0.013208 0.011126
                       20 Middle East and North Africa     -1.60e-02   1.71e-02 -0.937    0.349 1.5 -0.049462 0.017472
                       80 Western Europe and North America -1.07e-05   6.53e-05 -0.164    0.869 0.2 -0.000139 0.000117
                       80 Latin America and the Caribbean  -2.78e-03   2.61e-03 -1.065    0.287 1.8 -0.007884 0.002333
                       80 Middle East and North Africa     -7.28e-04   1.91e-03 -0.381    0.703 0.5 -0.004475 0.003019
Term: public_sector_corruption
Type: response
Comparison: dY/dX

> model_logit_fancy |> 
+   emtrends(~ public_sector_corruption + region, var = "public_sector_corruption",
+            at = list(public_sector_corruption = c(20, 80),
+                      region = regions_to_use),
+            regrid = "response", delta.var = 0.001) 
 public_sector_corruption region                           public_sector_corruption.trend       SE  df asymp.LCL asymp.UCL
                       20 Western Europe and North America                      -1.74e-02 1.34e-02 Inf -0.043632  0.008780
                       80 Western Europe and North America                      -1.07e-05 6.53e-05 Inf -0.000139  0.000117
                       20 Latin America and the Caribbean                       -1.04e-03 6.21e-03 Inf -0.013207  0.011125
                       80 Latin America and the Caribbean                       -2.78e-03 2.61e-03 Inf -0.007886  0.002335
                       20 Middle East and North Africa                          -1.60e-02 1.71e-02 Inf -0.049460  0.017470
                       80 Middle East and North Africa                          -7.28e-04 1.91e-03 Inf -0.004475  0.003019
Confidence level used: 0.95 
```

> 第三步：计算拟合值

实际问题中，我们更关心拟合值。通过以下命令，我们给出了特定腐败程度指数为20或80的不同地区的选举透明概率的拟合值，可以发现此时结果汇报的是prob而不是边际效应。例如：东欧和中亚地区，当腐败程度指数为20时，选举透明的概率为0.179。

```r
model_logit_fancy |> 
  predictions(newdata = datagrid(public_sector_corruption = c(20, 80),
                                 region = regions_to_use))

model_logit_fancy |> 
  emmeans(~ public_sector_corruption + region, var = "public_sector_corruption",
          at = list(public_sector_corruption = c(20, 80),
                    region = regions_to_use),
          regrid = "response") 

> model_logit_fancy |> 
+   emmeans(~ public_sector_corruption + region, var = "public_sector_corruption",
+           at = list(public_sector_corruption = c(20, 80),
+                     region = regions_to_use),
+           regrid = "response") 
 public_sector_corruption region                               prob       SE  df asymp.LCL asymp.UCL
                       20 Western Europe and North America 1.79e-01 0.200000 Inf   -0.2129   0.57012
                       80 Western Europe and North America 7.25e-05 0.000496 Inf   -0.0009   0.00104
                       20 Latin America and the Caribbean  2.44e-01 0.146000 Inf   -0.0434   0.53058
                       80 Latin America and the Caribbean  8.70e-02 0.113000 Inf   -0.1349   0.30884
                       20 Middle East and North Africa     4.73e-01 0.337000 Inf   -0.1873   1.13249
                       80 Middle East and North Africa     7.85e-03 0.026700 Inf   -0.0445   0.06024

Confidence level used: 0.95 
```

进而我们根据以下命令将拟合值的结果可视化。

```r
ggplot(logit_predictions, aes(x = public_sector_corruption, y = prob, color = region)) +
  geom_line(linewidth = 1) +
  labs(x = "Public sector corruption", y = "Predicted probability of having\na campaign finance disclosure law", color = NULL) +
  scale_y_continuous(labels = percent_format()) +
  scale_color_manual(values = c(clrs, "grey30")) +
  theme_mfx() +
  theme(legend.position = "bottom")
```

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefinedB735-%E7%89%B9%E5%AE%9A%E5%80%BC%E5%A4%84%E8%BE%B9%E9%99%85%E6%95%88%E5%BA%94%E5%8F%AF%E8%A7%86%E5%8C%96-%E8%B5%B5%E6%9C%88%E5%B7%9D.png)

该图展示了不同地区中，随着腐败程度指数，选举透明的概率是如何变化的。

### 3.3 反事实值处的边际效应

3.2中所计算的边际效用其实是所有协变量处于均值处得到的，但是这与真实世界不一致。因而我们需要保留其他变量的原始值，计算**反事实值处的边际效应（Average marginal effects at counterfactual user-specified values）**。

```r
cfct_data <- datagrid(model = model_logit_fancy,
                      public_sector_corruption = c(20, 80),
                      grid_type = "counterfactual")

# Specify variables = "public_sector_corruption" so that it doesn't calculate
# slopes and contrasts for all the other covariates
mfx_cfct <- model_logit_fancy |> 
  slopes(newdata = datagrid(public_sector_corruption = c(20, 80),
                            grid_type = "counterfactual"),
         variables = "public_sector_corruption")

head(mfx_cfct)

mfx_cfct |> 
  group_by(public_sector_corruption) |> 
  summarize(avg_slope = mean(estimate))

model_logit_fancy |> 
  avg_slopes(newdata = datagrid(public_sector_corruption = c(20, 80),
                                grid_type = "counterfactual"),
             variables = "public_sector_corruption",
             by = "public_sector_corruption")

mfx_cfct |> 
  filter(region %in% regions_to_use) |> 
  group_by(public_sector_corruption, region) |> 
  summarize(avg_slope = mean(estimate))

> model_logit_fancy |> 
+   avg_slopes(newdata = datagrid(public_sector_corruption = c(20, 80),
+                                 grid_type = "counterfactual"),
+              variables = "public_sector_corruption",
+              by = c("public_sector_corruption", "region"))

 public_sector_corruption                           region  Estimate Std. Error       z Pr(>|z|)   S    2.5 %   97.5 %
                       20 Eastern Europe and Central Asia  -5.15e-03   0.006011 -0.8566   0.3916 1.4 -0.01693 0.006632
                       20 Latin America and the Caribbean  -1.13e-03   0.006589 -0.1711   0.8641 0.2 -0.01404 0.011787
                       20 Middle East and North Africa     -1.06e-02   0.014891 -0.7107   0.4773 1.1 -0.03977 0.018602
                       20 Sub-Saharan Africa               -3.77e-04   0.005132 -0.0735   0.9414 0.1 -0.01044 0.009681
                       20 Western Europe and North America -2.71e-02   0.028026 -0.9657   0.3342 1.6 -0.08200 0.027865
                       20 Asia and Pacific                 -1.56e-02   0.010203 -1.5242   0.1275 3.0 -0.03555 0.004446
                       80 Eastern Europe and Central Asia  -8.20e-03   0.004482 -1.8301   0.0672 3.9 -0.01699 0.000582
                       80 Latin America and the Caribbean  -4.64e-03   0.004813 -0.9649   0.3346 1.6 -0.01408 0.004789
                       80 Middle East and North Africa     -4.35e-04   0.001126 -0.3863   0.6992 0.5 -0.00264 0.001772
                       80 Sub-Saharan Africa               -2.20e-03   0.002354 -0.9333   0.3507 1.5 -0.00681 0.002417
                       80 Western Europe and North America -8.99e-05   0.000555 -0.1619   0.8714 0.2 -0.00118 0.000998
                       80 Asia and Pacific                 -9.92e-04   0.001899 -0.5224   0.6014 0.7 -0.00471 0.002730

Term: public_sector_corruption
Type: response
Comparison: dY/dX
```

根据以上命令，同样我们给出了特定腐败程度指数为20或80的不同地区的选举透明概率的边际效应。进而我们给出与3.2结果的对比图。可以发现结果相差不大。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefinedB735-%E5%8F%8D%E4%BA%8B%E5%AE%9E%E8%BE%B9%E9%99%85%E6%95%88%E5%BA%94-%E8%B5%B5%E6%9C%88%E5%B7%9D.png)


## 4. 相关推文

- [刘潍嘉](https://www.lianxh.cn/search.html?s=刘潍嘉), 2024, [偏态分布数据的回归模型选择](https://www.lianxh.cn/details/1405.html), 连享会 No.1405.
- [匡宇驰](https://www.lianxh.cn/search.html?s=匡宇驰), 2024, [log-0 问题：零值太多如何取对数？](https://www.lianxh.cn/details/1455.html), 连享会 No.1455.
- [吴煜铭](https://www.lianxh.cn/search.html?s=吴煜铭), [郑浩文](https://www.lianxh.cn/search.html?s=郑浩文), 2021, [内生性！内生性！解决方法大集合](https://www.lianxh.cn/details/579.html), 连享会 No.579.
- [安超](https://www.lianxh.cn/search.html?s=安超), 2024, [论文推介：考虑样本自选择偏误的内生性随机前沿分析模型](https://www.lianxh.cn/details/1464.html), 连享会 No.1464.
- [张少鹏](https://www.lianxh.cn/search.html?s=张少鹏), 2021, [Stata：两部模型实现命令-twopm](https://www.lianxh.cn/details/742.html), 连享会 No.742.
- [张少鹏](https://www.lianxh.cn/search.html?s=张少鹏), 2021, [Stata：样本选择偏误与两部模型-twopm-L121](https://www.lianxh.cn/details/714.html), 连享会 No.714.
- [张蛟蛟](https://www.lianxh.cn/search.html?s=张蛟蛟), 2022, [Stata：无需IV的自选择模型-egregsel](https://www.lianxh.cn/details/143.html), 连享会 No.143.
- [张蛟蛟](https://www.lianxh.cn/search.html?s=张蛟蛟), 2024, [fqreg：在零处扎堆的分位数回归](https://www.lianxh.cn/details/1461.html), 连享会 No.1461.
- [徐云娇](https://www.lianxh.cn/search.html?s=徐云娇), [甘徐沁](https://www.lianxh.cn/search.html?s=甘徐沁), 2020, [Stata：分位数回归中的样本自选择问题](https://www.lianxh.cn/details/469.html), 连享会 No.469.
- [徐婷](https://www.lianxh.cn/search.html?s=徐婷), 2020, [多层级Tobit模型及Stata应用](https://www.lianxh.cn/details/421.html), 连享会 No.421.
- [李琼琼](https://www.lianxh.cn/search.html?s=李琼琼), 2020, [Stata: 一文读懂 Tobit 模型](https://www.lianxh.cn/details/12.html), 连享会 No.12.
- [李琼琼](https://www.lianxh.cn/search.html?s=李琼琼), 2020, [Stata：双栏模型简介-(Double-hurdle-model)](https://www.lianxh.cn/details/380.html), 连享会 No.380.
- [李琼琼](https://www.lianxh.cn/search.html?s=李琼琼), 2021, [Stata：自选择偏误之双栏模型简介-(Double-hurdle-model)](https://www.lianxh.cn/details/803.html), 连享会 No.803.
- [牛坤在](https://www.lianxh.cn/search.html?s=牛坤在), 2021, [xtheckmanfe：面板Heckman模型的固定效应估计](https://www.lianxh.cn/details/688.html), 连享会 No.688.
- [王恩泽](https://www.lianxh.cn/search.html?s=王恩泽), 2023, [Stata：一文读懂两部模型-twopm](https://www.lianxh.cn/details/1170.html), 连享会 No.1170.
- [秦利宾](https://www.lianxh.cn/search.html?s=秦利宾), 2020, [Heckman 模型：你用对了吗？](https://www.lianxh.cn/details/153.html), 连享会 No.153.
- [章青慈](https://www.lianxh.cn/search.html?s=章青慈), 2022, [Stata：广义Heckman两步法-gtsheckman](https://www.lianxh.cn/details/1082.html), 连享会 No.1082.
- [连享会](https://www.lianxh.cn/search.html?s=连享会), 2021, [FAQs答疑-2021寒假-Stata高级班-Day2-连玉君-面板门槛-Heckman-Tobit](https://www.lianxh.cn/details/537.html), 连享会 No.537.
- [连玉君](https://www.lianxh.cn/search.html?s=连玉君), 2024, [antrho与Fisher转换：heckman和treareg等命令中参数含义](https://www.lianxh.cn/details/1358.html), 连享会 No.1358.
- [连玉君](https://www.lianxh.cn/search.html?s=连玉君), 2024, [antrho与Fisher转换：heckman和treareg等命令中参数含义](https://www.lianxh.cn/details/1361.html), 连享会 No.1361.
- [郭佳佳](https://www.lianxh.cn/search.html?s=郭佳佳), 2023, [内生性之应对（上）：原理篇--遗漏变量-反向因果-测量误差-自选择](https://www.lianxh.cn/details/1264.html), 连享会 No.1264.
- [郭佳佳](https://www.lianxh.cn/search.html?s=郭佳佳), 2023, [内生性之应对（下）：方法篇--遗漏变量-反向因果-测量误差-自选择](https://www.lianxh.cn/details/1266.html), 连享会 No.1266.
