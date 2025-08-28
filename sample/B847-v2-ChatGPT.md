
# B847：R语言-aggTrees 包


> **作者**：郭皑馨 (华南理工大学)
> **邮箱**：<valerie_guo@163.com>

> **Source:** Di Francesco, R. (2024). Aggregation Trees (Version 1). arXiv. [Link](https://doi.org/10.48550/arXiv.2410.11408) (rep), [PDF](https://arxiv.org/pdf/2410.11408.pdf), [Google](<https://scholar.google.com/scholar?q=Aggregation Trees (Version 1)>).

&emsp; 

> [ChatGPT 对话过程](https://chatgpt.com/share/68a6e702-bf84-8005-af84-e6ed9a35df2f)。我使用 ChatGPT 另外生成了一个版本的推文初稿，你可以在此基础上修改，也可以将你已经写好的内容合并进来。 

**注意**：修改要点

- ChatGPT 生成的内容仅供参考，里面有些表述可能是错误的，也有些地方详略不当，你需要根据原文酌情调整。
- 此外，ChatGPT 给出的参考文献链接也有错误，你需要认真校对。 
- R 实操部分，目前只放入了 R codes，但未提供执行后的结果和解释。你需要自行运行代码并对结果进行分析。


&emsp;


- **Title**: R语言-aggTrees 包
- **Keywords**: R语言, aggTrees, 数据分析

&emsp; 

----

## 1. 简介

在处理异质性效应估计时，我们通常会在三类策略之间做选择：报告总体平均处理效应 (ATE)，绘制条件平均处理效应曲线 (CATE)，或报告若干子组的平均效应 (GATE)。在实际研究与政策评估中，许多学者越来越重视第三类策略，因为它能提供**更具解释力的结论**，例如“对谁最有效”“在哪些群体中无效”，这些信息往往比“平均而言有效”更具有政策价值。

本文将介绍一篇聚焦于 GATE 构造的最新方法论文：**Di Francesco (2024) 提出的 Aggregation Trees (aggTrees)**。该方法为我们提供了一条清晰且可执行的路径：在不事先设定分组规则的前提下，利用机器学习工具构建一套**嵌套的、最优的、可解释的异质性分组序列**，并对每组的处理效应进行有效推断。

### 面向读者

这篇推文特别面向以下读者群体：

* 具备一定计量经济学基础的博士研究生、高年级硕士生；
* 对 Causal Forest、Causal Tree 等机器学习方法有所了解但未曾深入实操的教师；
* 期望在政策评估、医疗经济学、发展经济学等场景中识别异质性效应，并对其进行具备统计意义的推断的研究人员。

我们假设读者已经理解**选择于可观测 (Unconfoundedness) **这一识别前提，并了解如双重稳健 (Doubly Robust) 估计、交叉拟合 (Cross-fitting) 、Honest Splitting 等常见现代因果推断框架。

### 学完本推文，你将掌握：

* 为什么要从 ATE/CATE 过渡到 GATE；
* 如何通过 aggTrees 实现自适应分组并估计每组处理效应；
* 树形聚合 (aggregation trees) 的建模逻辑、估计流程与推断策略；
* 在 R 中完整复现该方法的工作流程，理解每一个核心函数；
* 如何解读输出结果，并将其用于机制识别或政策建议。

### 内容安排概览

* 第 2 节将介绍 ATE、CATE、GATE 三者的联系与局限；
* 第 3 节详细讲解 aggTrees 的三步法，包括树的生长与剪枝策略；
* 第 4 节说明该方法如何在观测研究下实现有效推断；
* 第 5 节展示 R 端实操流程与代码模板；
* 第 6 节汇总结果可视化与机制解读建议；
* 第 7 节提供适用情境与常见误用提示；
* 第 8 节附录中整理伪代码与流程图，便于教学引用；
* 文末提供完整参考文献、代码资源与数据来源。

本文将尽量以清晰可读的中文讲义方式呈现，力求**把方法讲清楚，把推理讲透彻，把代码讲明白**，让不同背景的读者都能从中受益。

## 2. 问题动机与方法全景 

在现代应用中，政策制定者和实证研究者越来越关注异质性：不仅想知道某项干预“有没有用” (ATE)，更想知道“对谁最有用” (CATE 或 GATE)。然而，直接报告 CATE 函数往往面临解释困难与过拟合风险，如何构建**既具有可解释性，又能有效利用数据的信息**的处理效应分组，成为一个重要问题。

### 2.1 从 ATE 到 CATE，再到 GATE

我们先从最基础的定义出发：

* **ATE (平均处理效应)**：

  $$
  \mathrm{ATE} = \mathbb{E}[Y(1) - Y(0)]
  $$

  是最常见的总体效应衡量指标，简洁、稳定，但往往掩盖了群体间的差异性。

* **CATE (条件平均处理效应)**：

  $$
  \tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]
  $$

  捕捉了个体异质性，但该函数往往很复杂，实证中难以解释；机器学习方法虽可估算 $\hat\tau(x)$，但如何将其用于学术与政策叙事，仍是挑战。

* **GATE (分组平均处理效应)**：

  $$
  \mathrm{GATE}_g = \mathbb{E}[Y(1) - Y(0) \mid X \in G_g]
  $$

  是对 $\tau(x)$ 的分段逼近，其中 $G_g$ 为第 $g$ 个子组。关键在于，如何**客观、公正地构造分组 $G_1, G_2, ..., G_K$**，以兼顾信息量与解释性。

**aggTrees** 正是为了构建这样一族“最优且嵌套”的 GATE 分组应运而生。

### 2.2 聚合的动机：既要差异，又要可解释

Di Francesco 指出一个常被忽视的张力：

* 在机器学习下，我们可以用 causal forest 等方法精确估计 $\hat\tau(x)$；
* 但该函数往往不便于解释、复现与交流，特别是当协变量维度很高时；
* 而预设分组 (例如 quartile、性别、教育) 又存在 **“事后选择”的 p-hacking 风险**；
* 因此我们需要一个数据驱动但有结构的策略：**对 $\hat\tau(x)$ 进行“有监督的解释性聚合”**。

这就是 aggTrees 的本质目标：**以聚合树的方式，将 $\hat\tau(x)$ 转化为一族嵌套的可解释分组**。

### 2.3 aggTrees 方法三步法概览

论文提出的方法由以下三步构成，每一步都具有明确目的和算法基础。

**第一步：估计个体处理效应 $\hat\tau(x)$**

* 可使用任意方法，如 Causal Forest (via `grf`) 、T-Learner、X-Learner、BART 等；
* 该步骤完全在训练样本中进行，输出的是一个连续函数估计 $\hat\tau(x)$。

**第二步：构建决策树，对 $\hat\tau(x)$ 进行聚合建模**

* 用回归树拟合 $\hat\tau(x)$，目的是寻找一棵能划分数据、揭示异质性结构的“深树”；
* 与 Athey & Imbens (2016) 的 causal tree 不同，这里的分裂准则是**最小化叶节点内部 $\hat\tau(x)$ 的方差**；
* 分裂过程具有自然解释性，如 “当母亲受教育年限 > 12 时 $\hat\tau(x)$ 明显下降”，则形成一个分组节点。

**第三步：通过剪枝构建一族嵌套分组**

* 使用 cost-complexity pruning 方法，对深树进行逐步合并；
* 每一步合并的两个叶子是“合并后对整体 $\hat\tau(x)$ 方差增加最小”的那一对；
* 最终得到一个嵌套分组序列 $\mathcal{G}^{(1)} \subset \mathcal{G}^{(2)} \subset \dots \subset \mathcal{G}^{(K)}$；
* 用户可以选择颗粒度 (组数) $K$，以平衡模型复杂度与异质性解释力。

这一流程兼顾了灵活性与可控性，是目前 GATE 文献中最系统的聚合方法之一。

### 2.4 可视化示意 (流程图) 

我们可将整个三步法以流程图的形式理解如下：

```mermaid
flowchart TD
    A[输入：观测数据 (Y, D, X)] --> B[估计个体处理效应 \n \hat{τ}(x)]
    B --> C[用回归树对 \hat{τ}(x) 做聚合建模]
    C --> D[得到一棵深树 \n 各叶节点为初始分组]
    D --> E[剪枝形成嵌套分组序列 \n G^{(1)} ⊂ G^{(2)} ⊂ ... ⊂ G^{(K)}]
```

我们将在下一节介绍该方法的建树算法细节、分裂与剪枝策略，以及与现有方法如 causal tree 的对比。

---

**延伸阅读与参考文献**

* Di Francesco, R. (2024). *Aggregation Trees*. [arXiv:2410.11408](https://doi.org/10.48550/arXiv.2410.11408), [PDF](http://sci-hub.ren/10.48550/arXiv.2410.11408), [github](https://riccardo-df.github.io/aggTrees/)
* Athey, S., & Imbens, G. (2016). Recursive Partitioning for Heterogeneous Causal Effects. *PNAS*, 113(27), 7353–7360. [Link](https://doi.org/10.1073/pnas.1510489113)

下一节，我们将具体进入树的构建、聚合逻辑与剪枝算法，解释如何在 $\hat\tau(x)$ 基础上构建一棵“可解释的异质性树”。


## 3. 树的构建与聚合算法 

上一节我们介绍了 aggTrees 的方法框架，核心是将个体效应 $\hat\tau(x)$ 聚合为可解释的分组序列。实现这一目标的关键步骤，是以回归树为基础构建聚合结构，再借助剪枝算法形成嵌套分组。与传统的 Causal Tree 最大不同之处，在于 **aggTrees 的建树目标并非直接拟合 $Y$ 或 $Y(1)-Y(0)$，而是**有监督地压缩 $\hat\tau(x)$ 的变异性。

本节将详细介绍该方法的建树逻辑、剪枝机制，并对比相关方法。

### 3.1 用回归树聚合个体效应估计值

aggTrees 方法的输入是训练样本上的 $\hat\tau(x)$，即每个观测值 $i$ 的估计处理效应 $\hat\tau_i$，以及对应的特征向量 $X_i$。目标是构建一棵回归树 $\mathcal{T}$，该树将整个样本划分为若干叶子节点 (即组别)，使得**每个叶子内 $\hat\tau_i$ 的变异尽可能小**。

具体过程如下：

* 分裂准则采用 CART 样式的平方误差损失：

  $$
  \text{Loss}(\mathcal{T}) = \sum_{g=1}^{K} \sum_{i \in G_g} \left( \hat\tau_i - \bar\tau_{G_g} \right)^2
  $$

  其中 $G_g$ 表示第 $g$ 个叶子组，$\bar\tau_{G_g}$ 为该组内的 $\hat\tau_i$ 均值。

* 贪心分裂每一步选择**能使总损失下降最多的变量与阈值对**；

* 与传统 CART 最大不同的是，这里建树的目标是逼近 $\hat\tau(x)$，而非 $Y$ 本身。

这种策略有两个直接好处：

1. 可以排除仅影响 $Y$ 水平、不影响效应的变量；
2. 在高维设定中，能够更稳定地发现**处理效应异质性最显著的变量组合**。

### 3.2 剪枝：构造嵌套分组序列

建成深树后，aggTrees 使用成本复杂度剪枝 (cost-complexity pruning) 算法生成一系列更简单的子树。该算法广泛用于 CART 方法，其核心思想是逐步合并两组，并在每一步选择**“合并后造成误差增加最小”的两组**。

操作流程如下：

1. 初始深树包含 $K$ 个叶子；
2. 每一步，选择一对叶子组 $G_a$, $G_b$，合并后新组为 $G_{ab}$，计算：

   $$
   \Delta_{ab} = \sum_{i \in G_a \cup G_b} (\hat\tau_i - \bar\tau_{ab})^2 - \left[ \sum_{i \in G_a} (\hat\tau_i - \bar\tau_a)^2 + \sum_{i \in G_b} (\hat\tau_i - \bar\tau_b)^2 \right]
   $$
3. 选择 $\Delta_{ab}$ 最小的一对，合并为新叶子；
4. 重复该过程，直到仅剩一个组。

最终我们得到一系列嵌套的分组划分：

$$
\mathcal{G}^{(1)} \subset \mathcal{G}^{(2)} \subset \dots \subset \mathcal{G}^{(K)}
$$

用户可以据此选择最合适的颗粒度 $K$。通常推荐同时报告多个 $K$ 下的结果，体现方法的 **“层级嵌套”** 特性。

### 3.3 与 Causal Tree 方法的异同

我们将 aggTrees 与 Athey & Imbens (2016) 提出的 Causal Tree 方法进行简要对比：

| 特征     | Causal Tree                         | Aggregation Trees                     |
| -------- | ----------------------------------- | ------------------------------------- |
| 分裂目标 | 最大化处理效应差异 (ATE diff)       | 最小化 $\hat\tau(x)$ 方差           |
| 分裂依据 | 原始 $Y, D$ 数据                  | 已估计的 $\hat\tau(x)$              |
| 分裂策略 | honest splitting + cross-validation | 回归树 + 剪枝生成序列                 |
| 可解释性 | 一棵树对应一个分组                  | 一系列嵌套树对应多个分组              |
| 稳健性   | 分裂变量可能包含仅影响 $Y$ 的变量 | 更稳健，仅压缩 $\hat\tau(x)$ 的变异 |

aggTrees 的优势在于 **模块化结构**：可以配合任意 $\hat\tau(x)$ 的生成方式；而在某些高维协变量设定下，它对 “噪声变量” 的鲁棒性也更高。

### 3.4 一个示例：从深树到三层分组

假设我们在模拟数据中已得到了 $\hat\tau(x)$，初始建树得到如下深树 (8 组)：

```python
Level-0: All data
├── Income < 30000
│   ├── Age < 35  → G1
│   └── Age ≥ 35  → G2
└── Income ≥ 30000
    ├── Education < 12 → G3
    └── Education ≥ 12
        ├── Gender = Male → G4
        └── Gender = Female
            ├── Age < 45 → G5
            └── Age ≥ 45 → G6
```

剪枝后形成如下嵌套结构：

* 6 组时：G5 与 G6 合并；
* 4 组时：G3 与 G4 合并；
* 2 组时：Income < 30k 与其余合并；
* 1 组时：所有样本合并。

我们将看到，这种逐层分组能很好地服务于“对谁最有效”的分析任务，特别是政策制定中按人群定制干预策略时。

---

## 4. 在观测数据中进行有效推断 

上一节我们介绍了如何用回归树聚合 $\hat{\tau}(x)$ 得到嵌套的异质性分组序列。本节关注一个更重要的环节：**如何对这些分组进行有效的处理效应推断？**

我们特别聚焦于观测研究背景下的方法。这类研究面临的最大挑战是**处理组和对照组并非随机分配**，从而使得直接的组内均值比较无法得到有效的 GATE。为了克服这一问题，aggTrees 采用了“双重稳健打分 + honesty + 交叉拟合”的一整套推断流程。

### 4.1 双重稳健估计与 AIPW 打分函数

在非实验数据中，估计某组内的处理效应 $\mathbb{E}[Y(1)-Y(0) \mid X \in G_g]$ 的常见做法是使用**调整后的差值**，其中最常用的是 AIPW (Augmented Inverse Probability Weighting) 打分函数。其形式为：

$$
\psi(W; \eta) = \mu_1(X) - \mu_0(X) + \frac{D \cdot (Y - \mu_1(X))}{e(X)} - \frac{(1 - D) \cdot (Y - \mu_0(X))}{1 - e(X)}
$$

其中：

* $W = (Y, D, X)$ 是观测数据；
* $\mu_d(X) = \mathbb{E}[Y \mid D = d, X]$ 是条件均值函数；
* $e(X) = \mathbb{P}(D = 1 \mid X)$ 是倾向得分函数。

aggTrees 采用的是这样的思路：

* 使用训练样本估计 $\mu_0(X), \mu_1(X), e(X)$；
* 用这些函数值，在诚实样本上计算 $\psi_i$；
* 最终，在诚实样本中，对 $\psi_i$ 进行组别回归 (分组 dummy 回归)，以得到每个分组的 GATE 估计。

这一方法的最大优点是：**若 $\mu_d(X)$ 或 $e(X)$ 中任意一项估计正确，GATE 估计就是一致的** (双重稳健性)。

### 4.2 Honest Splitting：确保推断不被训练污染

在因果推断中，“事后选分组，再进行组内推断”容易引发偏差。为防止这一问题，aggTrees 引入了所谓的 **honest splitting**：

* 样本在建模初期就被分为两个不重叠的部分：

  * 训练样本：用于估计 $\hat\tau(x)$ 与构建聚合树；
  * 诚实样本：用于评估 GATE 与估计标准误；
* 分组结构仅由训练样本决定，不依赖于诚实样本；
* 推断过程不再“事后选择组别”，而是以“已知分组”的方式回归打分函数。

这一思想源于 Athey & Imbens 的 causal tree 构建，也是现代因果机器学习方法的基本原则。

### 4.3 交叉拟合与有效区间估计

为了进一步提高效率与稳健性，aggTrees 支持**交叉拟合 (cross-fitting) **的策略，即：

1. 将样本分为多折；
2. 在每一折上用其它折估计 $\mu_d(X)$ 与 $e(X)$；
3. 对于第 $k$ 折样本，使用其自身的数据估计 GATE 与标准误。

这种策略能减少高维回归中模型误差对打分函数的偏移，确保以下性质：

* GATE 估计满足 **$\sqrt{n}$ 收敛**；
* 每组的区间估计可通过正态近似给出；
* 不同组之间的差异检验也可直接进行。

在 R 包中，`inference_aggtree()` 函数将自动执行这些步骤，并返回：

* 每个分组的估计值与标准误；
* 所有两两组之间的差异估计与 Holm 调整后的 $p$ 值；
* 每个组的平均协变量特征，用于机制分析。

### 4.4 小结：从建树到推断的整合性流程

我们将整个流程简要总结如下：

| 步骤   | 操作                     | 使用样本 | 工具函数              |
| ------ | ------------------------ | -------- | --------------------- |
| Step 1 | 估计 $\hat\tau(x)$     | 训练集   | `grf::causal_forest`  |
| Step 2 | 建树并剪枝               | 训练集   | `build_aggtree()`     |
| Step 3 | 构造打分函数 $\psi_i$ | 诚实集   | `dr_scores()`         |
| Step 4 | 分组 dummy 回归          | 诚实集   | `inference_aggtree()` |

整个流程类似于对“模型选择”与“推断”进行物理隔离，最大程度避免因 overfitting 或 data snooping 带来的偏差。

---

## 5. R 实操流程 

在介绍了 aggTrees 的理论框架和推断方法后，我们现在进入最关键的部分：**如何在 R 中完整实现该方法**。这一节提供从安装到运行的全流程示例，既能用于教学演示，也能直接嵌入到研究论文的复现部分。

### 5.1 安装与加载

aggTrees 已经正式发布在 CRAN 上，可以直接安装：

```r
# 安装一次即可
install.packages("aggTrees")

# 加载依赖包
library(aggTrees)
library(grf)      # 用于估计 CATE
library(rpart)    # 回归树
library(dplyr)    # 数据处理
```

在后续流程中，我们还会用到 `ggplot2` 来绘制结果图。

### 5.2 数据准备

在教学中，常见有两类数据：

1. **模拟数据** (便于快速演示算法逻辑)；
2. **论文应用数据** (例如母亲吸烟与新生儿体重的研究)。

这里先展示模拟数据流程，后续你可以替换为实证数据。

```r
set.seed(123)

# 模拟数据
n <- 1000
X1 <- rnorm(n)
X2 <- rbinom(n, 1, 0.5)
tau <- 2 * (X1 > 0) + X2  # 真正的异质性效应
D <- rbinom(n, 1, 0.5)
Y <- 1 + X1 + 0.5*X2 + tau * D + rnorm(n)

dat <- data.frame(Y, D, X1, X2)
```

### 5.3 样本切分：训练与诚实集

```r
# 将数据划分为训练集和诚实集
splits <- sample_split(Y = dat$Y, D = dat$D, X = dat[,c("X1","X2")])
Y_tr <- splits$Y_tr; D_tr <- splits$D_tr; X_tr <- splits$X_tr
Y_hon <- splits$Y_hon; D_hon <- splits$D_hon; X_hon <- splits$X_hon
```

### 5.4 估计个体处理效应 CATE

```r
# 使用 causal forest 估计 tau(x)
cf <- causal_forest(X_tr, Y_tr, D_tr)
cates_tr <- predict(cf, X_tr)$predictions
cates_hon <- predict(cf, X_hon)$predictions
```

### 5.5 建树与生成聚合分组

```r
# 构建聚合树
agg_model <- build_aggtree(Y_tr, D_tr, X_tr,
                           Y_hon, D_hon, X_hon,
                           cates_tr, cates_hon,
                           method = "aipw")

# 绘制聚合分组序列
plot(agg_model, sequence = TRUE)
```

该函数会返回一系列嵌套分组，从一组到多组，供用户选择。

### 5.6 在诚实集上做推断

```r
# 在不同颗粒度下进行推断，例如选择 3 组
res <- inference_aggtree(agg_model, n_groups = 3)

# 输出结果
res$group_effects         # 各组的 GATE 估计值与区间
res$group_differences     # 各组间差异及 Holm 调整后的 p 值
res$avg_characteristics   # 各组平均协变量特征
```

### 5.7 输出结果与可视化

```r
# 绘制分组效应估计
ggplot(res$group_effects, aes(x = group, y = estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2) +
  ylab("Estimated GATE") +
  xlab("Group")
```

这张图清楚展示了每组的效应估计与置信区间，非常适合在论文或课堂展示中使用。

### 5.8 实践建议与调参要点

* **树深度控制**：在 `rpart.control` 中可设置 `maxdepth`、`minbucket` 等参数，以防止过拟合；
* **样本数限制**：每个叶子节点建议至少包含 30–50 个观测值，否则推断结果不稳定；
* **稳健性检查**：使用不同随机种子、多次样本切分，检查结果是否一致；
* **重叠性检查**：在每组内检验倾向得分分布是否良好，避免极端权重。

---
## 6. 结果呈现与解读 

完成估计与推断后，研究者的下一步工作就是如何**将结果组织成表格与图形**，并在论文或课堂中做出清晰解释。aggTrees 的设计理念不仅是为了得到数值结果，更是为了帮助研究者发现、展示并解释“谁受益最多”的异质性模式。

### 6.1 树结构与分组图示

首先应输出一张“树结构图”，展示建树与剪枝的路径。这类图形的要点包括：

* 每个分裂节点显示分裂变量与阈值 (例如“Income ≥ 30000”)；
* 每个叶子显示分组编号 (G1, G2, …)；
* 若可能，叶子旁标注该组的平均 $\hat\tau(x)$。

这种图能帮助读者快速理解：哪些变量主导了异质性划分，不同人群是如何被聚合到一起的。

### 6.2 分组效应估计表

核心结果是各组 GATE 的数值与置信区间，推荐整理成如下表格：

| 组别 | GATE | 标准误 | 95% CI         | 组内样本数 |
| ---- | ---- | ------ | -------------- | ---------- |
| G1   | 2.10 | 0.45   | \[1.22, 2.98]  | 150        |
| G2   | 0.85 | 0.30   | \[0.27, 1.43]  | 180        |
| G3   | 0.10 | 0.25   | \[-0.39, 0.59] | 170        |

解读要点：

* G1 显著大于 0，说明该群体对处理反应强烈；
* G3 效应接近 0，且区间跨 0，说明该群体几乎无效应。

### 6.3 组间差异检验

除了各组自身的效应估计，还应展示“组间差异是否显著”。aggTrees 的 `inference_aggtree()` 提供了组间差异的估计与多重比较调整后的 p 值。例如：

| 差异比较 | 差异估计 | 标准误 | 调整后 p 值 |
| -------- | -------- | ------ | ----------- |
| G1 – G2  | 1.25     | 0.55   | 0.04        |
| G1 – G3  | 2.00     | 0.50   | 0.00        |
| G2 – G3  | 0.75     | 0.40   | 0.08        |

若差异显著，说明不同组间的确存在异质性反应，这比单纯报告“有异质性”更有说服力。

### 6.4 协变量均值表 (机制分析) 

每个分组的平均协变量特征，可以帮助研究者回答“为什么该组的效应大/小”。例如：

| 组别 | 平均收入 | 平均教育年限 | 吸烟率 |
| ---- | -------- | ------------ | ------ |
| G1   | 20000    | 10.5         | 0.70   |
| G2   | 35000    | 12.3         | 0.45   |
| G3   | 50000    | 14.1         | 0.20   |

结合这些结果，可以推测：“低收入、低教育水平群体的干预效应最大”，这为政策提供了直接线索。

### 6.5 可视化展示

在学术写作中，除了表格外，还应配合图形呈现：

* **效应估计图**：不同分组的点估计与区间 (forest plot 样式)；
* **分组数 vs 异质性解释力图**：横轴为组数，纵轴为树分裂解释的 $\hat\tau(x)$ 方差比例，直观呈现选择合适 $K$ 的依据；
* **协变量分布图**：展示各组在关键变量上的分布差异。

这些图形能直观传达异质性模式，并提升结果的可读性。

---

## 7. 适用场景、边界条件与常见陷阱 

aggTrees 的提出旨在为学术研究和政策分析提供“可解释的异质性发现工具”。但和任何方法一样，它既有优势，也存在局限。正确理解其适用情境与边界条件，对避免误用至关重要。

### 7.1 适用场景

* **政策评估**：例如补贴、税收、教育干预，研究者希望找出对哪些群体最有效；
* **医疗经济学**：比较不同病人群体对某种治疗的响应差异；
* **发展经济学**：在随机试验或自然实验中识别异质性；
* **劳动与教育研究**：理解不同人口子群在干预中的收益差异。

### 7.2 不适用或需谨慎的情境

* **弱重叠问题**：若某些分组的倾向得分分布严重偏斜，估计会不稳健；
* **过小的组样本量**：叶子节点太小会导致标准误过大；
* **强选择偏差**：若不可观测变量决定处理分配，则即便 AIPW 也难以矫正；
* **过度调参**：若用户人为调节树参数使分组更“好看”，就违背了 honesty 原则。

### 7.3 与其他方法的关系

* **Causal Trees**：直接以处理效应差异作为分裂准则，得到一棵树；而 aggTrees 是先学 $\hat\tau(x)$，再用树来解释与聚合。
* **Causal Forests**：适合预测个体效应 $\tau(x)$，但解释性弱；aggTrees 可以用它的输出作为输入，从而得到可解释的分组。
* **Policy Trees**：目标是找到最优政策分配，而非揭示异质性；aggTrees 的定位更接近“异质性解释工具”。
* **分组回归/交互项模型**：需要事先设定分组，而 aggTrees 是数据驱动的，避免了 p-hacking。

---

## 8. 总结与扩展 

本文系统介绍了 aggTrees 方法：

* 从 **ATE → CATE → GATE** 的逻辑出发，说明了构建分组平均处理效应的重要性；
* 介绍了 **三步法**：估计 $\hat\tau(x)$、用树聚合、剪枝形成嵌套分组；
* 展示了 **双重稳健推断** 与 **honesty + 交叉拟合** 的技术路径；
* 提供了 **R 实操代码**，涵盖从安装到推断的完整流程；
* 讨论了 **结果呈现与解释**，并强调了适用场景与边界条件。

aggTrees 的意义在于，它为我们提供了一种**可解释的、层级嵌套的异质性发现工具**，既能避免过度依赖黑箱模型，又能在学术研究和政策评估中提供可信的异质性证据。

未来的研究方向包括：

* 将 aggTrees 与 **因果中介分析**结合，探索不同路径下的异质性；
* 与 **机器学习预测方法** (如 Lasso、XGBoost) 结合，用于高维数据中的效应探索；
* 扩展至 **动态处理效应** 或 **网络干预效应** 的分组发现。

---

## 参考文献

1. Di Francesco, R. (2024). Aggregation Trees. *arXiv*. [Link](https://doi.org/10.48550/arXiv.2410.11408), [PDF](http://sci-hub.ren/10.48550/arXiv.2410.11408), [Google](https://scholar.google.com/scholar?q=Aggregation+Trees+Di+Francesco), [github](https://riccardo-df.github.io/aggTrees/).
2. Di Francesco, R. (2022). Aggregation Trees. *CEIS Research Paper, 546*. [Link](https://doi.org/10.2139/ssrn.4304256), [PDF](http://sci-hub.ren/10.2139/ssrn.4304256), [Google](https://scholar.google.com/scholar?q=Aggregation+Trees+Di+Francesco).
3. Athey, S., & Imbens, G. (2016). Recursive Partitioning for Heterogeneous Causal Effects. *PNAS*, 113(27), 7353–7360. [Link](https://doi.org/10.1073/pnas.1510489113), [PDF](http://sci-hub.ren/10.1073/pnas.1510489113), [Google](https://scholar.google.com/scholar?q=Recursive+Partitioning+for+Heterogeneous+Causal+Effects).
4. Semenova, V., & Chernozhukov, V. (2021). Debiased Machine Learning of Conditional Average Treatment Effects and Other Causal Functions. *The Econometrics Journal*, 24(2), 264–289. [Link](https://doi.org/10.1093/ectj/utaa020), [PDF](http://sci-hub.ren/10.1093/ectj/utaa020), [Google](https://scholar.google.com/scholar?q=Debiased+Machine+Learning+of+Conditional+Average+Treatment+Effects).

