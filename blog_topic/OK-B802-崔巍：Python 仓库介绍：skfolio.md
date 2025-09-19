# B802：Python 仓库介绍：skfolio

- https://github.com/skfolio/skfolio
- [website](https://skfolio.org/)

## 1. skfolio 的定位与主要功能

`skfolio` 是一个专为 **投资组合优化**（portfolio optimization）、**绩效评估** 和 **风险分析**设计的 Python 包。它灵感来源于 `scikit-learn` 的 API 规范，兼容 `pandas` 和 `numpy`，支持与机器学习管道集成。其主要目标是让用户能够方便地进行多资产投资组合建模、优化、回测和分析。

主要特点包括：

* **投资组合建模**：提供多种投资组合模型（如均值-方差、Black-Litterman、风险平价等）和优化器。
* **集成机器学习管道**：支持与 scikit-learn 流水线（pipeline）无缝集成，可结合特征工程、时间序列预测等机器学习流程。
* **丰富的风险/收益指标**：内置多种投资组合绩效评估和风险度量工具（如夏普比率、索提诺比率、最大回撤等）。
* **自动化多资产回测**：支持资产收益率输入，自动执行优化、权重分配及绩效分析。

## 2. 安装方法

```bash
pip install skfolio
```

## 3. 基本使用流程

### 3.1 数据准备

通常需要一个多资产的收益率时间序列（可以是 DataFrame 格式，每列为一个资产）。

```python
import pandas as pd
returns = pd.read_csv('your_returns.csv', index_col=0, parse_dates=True)
# 每一列为不同资产的日收益率
```

### 3.2 投资组合模型与优化

以经典均值-方差（Mean-Variance）模型为例：

```python
from skfolio import MeanVariance, PortfolioOptimizer

# 创建一个均值-方差模型
model = MeanVariance()

# 创建优化器（最大化夏普比率）
optimizer = PortfolioOptimizer(model=model, objective="sharpe")

# 拟合优化器并获得最优投资组合
optimizer.fit(returns)
optimal_portfolio = optimizer.get_optimal_portfolio()
print(optimal_portfolio.weights)  # 输出各资产权重
```

### 3.3 绩效评估与风险指标

```python
from skfolio.metrics import sharpe_ratio, max_drawdown

# 直接调用绩效指标
sr = sharpe_ratio(optimal_portfolio)
md = max_drawdown(optimal_portfolio)
print(f"Sharpe Ratio: {sr}, Max Drawdown: {md}")
```

### 3.4 与机器学习管道集成

`skfolio` 支持与 scikit-learn 兼容的流水线，比如结合特征选择或回归预测模型：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skfolio import MeanVariance, PortfolioOptimizer

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('optimizer', PortfolioOptimizer(model=MeanVariance(), objective="sharpe"))
])

pipe.fit(returns)
```

## 4. 支持的主要模型与功能

* 均值-方差模型（Mean-Variance, Markowitz）
* Black-Litterman 模型
* 风险平价（Risk Parity）
* 等权重（Equal Weight）
* 多种目标函数（最大化夏普比率、最小化方差、目标收益、目标风险等）
* 约束条件设置（如最大/最小持仓，行业暴露约束等）
* 时间序列回测与可视化

## 5. 与同类包的对比

| 包名           | 优势                                             | 劣势                                    |
| -------------- | ------------------------------------------------ | --------------------------------------- |
| skfolio        | sklearn 风格，API 友好，机器学习集成好，约束丰富 | 功能较新，社区资料有限                  |
| PyPortfolioOpt | 经典投资组合优化全，文档多                       | 主要偏向 Markowitz 和 Black-Litterman   |
| bt             | 强大的回测系统，支持复杂策略设计                 | 学习曲线略陡峭，投资组合建模灵活性一般  |
| riskfolio-lib  | 风险分散策略丰富，支持多种风险模型               | 机器学习集成弱，API 不如 skfolio 现代化 |

## 6. 官方资料与文档

* [skfolio 官方文档](https://skfolio.org/)
* [skfolio GitHub](https://github.com/skfolio/skfolio)

## 7. 参考文献

如需系统阅读投资组合优化和相关理论，推荐以下经典文献：

* Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77–91. [Link](https://doi.org/10.2307/2975974), [PDF](http://sci-hub.ren/10.2307/2975974), [Google](https://scholar.google.com/scholar?q=Portfolio+Selection+Markowitz)
