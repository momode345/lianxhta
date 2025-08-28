# FinanceDatabase：Python获取30万+全球金融数据的新工具

> 作者：吕卓阳
>
> 邮箱：lvzy20@163.com
>
> Keywords：FinanceDatabase，Python，金融数据，股票筛选，ETF分析，量化投资 

## 1.FinanceDatabase数据库简介

近期，由 JerBouma 团队开发了一个开源的数据库：**全球金融工具数据库** (FinanceDatabase, **FD**)。**FD** 是一个开源、持续更新的金融工具分类数据库。通过整合来自全球主要交易所和金融数据源，FD 构建了覆盖 **300,000+** 个金融工具的分类信息。这些数据涵盖股票、ETF、基金、加密货币、指数等多类资产，并包含详细的行业、地理位置和投资策略分类。

这一数据集覆盖了从传统股票到新兴加密货币的全谱金融工具，提供了金融产品分类和筛选的全面视角。它可以为研究人员提供跨资产类别、跨国比较和投资策略构建的基础工具。

后文内容结构如下：(2) 为何使用FinanceDatabase数据库？(3)  如何使用FinanceDatabase？(4)  拓展分析 ；（5）总结。

## 2.为何使用FinanceDatabase数据库？

FinanceDatabase团队之所以开发这个新的数据库，主要源于现有金融工具查询普遍存在的一些问题和局限：

1. **信息碎片化的限制**：虽然各大交易所提供了实时数据，但跨市场、跨资产类别的统一分类有限。
2. **产品发现的困难**：现有的金融平台通常存在产品发现困难的问题，且分类标准不够统一。
3. **数据整合的高成本与不一致性**：跨来源数据的协调与整合是一个复杂的过程，容易产生不一致性并且成本较高。

这些问题导致研究人员和投资者常常难以系统性地发现和分类金融工具，限制了跨资产分析的广泛性和投资策略的多样性。

数据特点

- **7 大资产类别**：包括股票、ETF、基金、指数、货币、加密货币、货币市场。
- **300,000+ 金融工具**：涵盖从历史到现代的全球金融产品数据。
- **多维度分类**：包括地理位置、行业分类、投资策略、市值等级等分类维度。
- **实时更新**：通过自动化数据管道，定期集成来自主要数据源的最新信息。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/lvzhuoyang_B785_Fig00.png)
**项目地址**：https://github.com/JerBouma/FinanceDatabase

## 3. 如何使用FinanceDatabase？

### 3.1 安装与基本调用

首先，可以在终端中安装financedatabase库:

```python
# cmd中安装financedatabase库
pip install financedatabase
```
然后，在python中导入financedatabase库，并初始化需要用到的数据库。
```python
# 导入使用
import financedatabase as fd

# 初始化七个数据库
equities = fd.Equities()        # 股票数据库
etfs = fd.ETFs()                # ETF数据库  
funds = fd.Funds()              # 基金数据库
indices = fd.Indices()          # 指数数据库
currencies = fd.Currencies()    # 货币数据库
cryptos = fd.Cryptos()          # 加密货币数据库
moneymarkets = fd.MoneyMarkets() # 货币市场数据库
```

### 3.2 基础筛选功能

以股票资产为例，初始化股票数据库后，可以使用多种筛选条件查询股票。FinanceDatabase提供了两种主要查询方式：

（1）条件筛选 - select() 方法

show_option可以显示可以获取的全部的可选项目:

```
fd.show_options("equities")
```

返回如下可选项，包括不同的货币类型、不同行业等信息

```
{'currency': array(['ARS', 'AUD', 'BRL', 'CAD', 'CHF', 'CLP', 'CNY', 'COP', 'CZK',
    'DKK', 'EUR', 'GBP', 'HKD', 'HUF', 'IDR', 'ILA', 'ILS', 'INR',
    'ISK', 'JPY', 'KES', 'KRW', 'LKR', 'MXN', 'MYR', 'NOK', 'NZD',
    'PEN', 'PHP', 'PLN', 'QAR', 'RUB', 'SAR', 'SEK', 'SGD', 'THB',
    'TRY', 'TWD', 'USD', 'ZAC', 'ZAR'], dtype=object),
 'sector': array(['Communication Services', 'Consumer Discretionary',
    'Consumer Staples', 'Energy', 'Financials', 'Health Care',
    'Industrials', 'Information Technology', 'Materials',
    'Real Estate', 'Utilities'], dtype=object),
 'industry_group': array(['Automobiles & Components', 'Banks', 'Capital Goods',
    'Commercial & Professional Services',
    'Consumer Durables & Apparel', 'Consumer Services',
    'Diversified Financials', 'Energy', 'Food & Staples Retailing',
    'Food, Beverage & Tobacco', 'Health Care Equipment & Services',
    'Household & Personal Products', 'Insurance', 'Materials',
    'Media & Entertainment',
    'Pharmaceuticals, Biotechnology & Life Sciences', 'Real Estate',
    'Retailing', 'Semiconductors & Semiconductor Equipment',
    'Software & Services', 'Technology Hardware & Equipment',
    'Telecommunication Services', 'Transportation', 'Utilities'],
       dtype=object)}
```

基于上述可选项，可以使用select功能筛选需要的股票列表：

```python
# 获取荷兰保险公司
equities.select(
    country='Netherlands',
    industry='Insurance',
)

# 多条件组合筛选
equities.select(
    country=['Netherlands', 'United States'],
    industry='Insurance',
    market=['Euronext Amsterdam', 'Nordic Growth Market', 'OTC Bulletin Board',
        'New York Stock Exchange', 'NASDAQ Global Select', 'NYSE MKT',
        'NASDAQ Capital Market']
)
```

**输出示例：**

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/lvzhuoyang_B785_Fig01.png)

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/lvzhuoyang_B785_Fig02.png)

除了使用 `select()` 方法进行精确的分类筛选外，当我们需要根据公司简介、名称等文本信息进行更灵活的查询时，`FinanceDatabase` 提供了第二种主要查询方式：

**(2) 关键词搜索 - search() 方法**

该方法允许你对数据库中的任意列进行关键词匹配。当预设的分类无法满足你的需求，或者你想寻找特定业务概念的投资标的时，`search()` 将是你的最佳选择。

例如，我们希望在法兰克福交易所中，寻找业务涉及“机器人”或“教育”的“设备”类公司，可以执行以下代码：

```
equities.search(
    summary=["Robotics", "Education"],
    industry_group="Equipment",
    market="Frankfurt",
    index=".F"
)
```

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/lvzhuoyang_B785_Fig03.png)

### 3.3 与财务数据匹配

FinanceDatabase本身只提供分类信息，要获取实际财务数据需要与Finance Toolkit匹配或者其他包含历史交易数据的数据库进行匹配：

### 申请Finance Toolkit API密钥

1. 注册FinancialModelingPrep账户

   - 访问：https://financialmodelingprep.com/
   - 注册免费账户
   - 获取API密钥

2. 免费账户限制

   - 每日250次API调用
   - 5年历史数据
   - 仅限美国交易所上市公司

   通过此番集成，我们现在便可以调用 Finance Toolkit 的强大功能，来获取这些公司的各项重要财务指标了。让我们从一个简单的例子开始，获取它们的历史股价数据。

```
# 第一步：筛选目标公司
dutch_insurance_companies = equities.select(
    country="Netherlands",
    industry="Insurance",
    market="Euronext Amsterdam",
)
print(f"荷兰保险公司: {len(dutch_insurance)} 家")

# 第二步：集成财务工具包
API_KEY = "your key"  # 替换为你的API密钥
toolkit = dutch_insurance.to_toolkit(
    api_key=API_KEY
)

# 第三步：获取历史价格数据
historical_data = toolkit.get_historical_data()
print("历史价格数据获取完成")

# 查看数据示例（查看开盘价）
print(historical_data['Open'])
```

得到返回：

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/lvzhuoyang_B785_Fig04.png)



## 4. 拓展分析

基于本数据库，我们可以做进一步的分析：

### 4.1 全球股票市场分布

```python
# 分析主要市场股票分布
major_markets = ["United States", "China", "Japan", "Germany", "United Kingdom"]
market_stats = {}

for country in major_markets:
    stocks = equities.select(country=country)
    market_stats[country] = len(stocks)
    print(f"{country}: {len(stocks)} 家上市公司")

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(market_stats.keys(), market_stats.values(), color='steelblue')
plt.title('Number of Listed Companies in Major Markets')
plt.ylabel('Number of Listed Companies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

得到返回：

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/lvzhuoyang_B785_Fig05.png)

### 4.2 ETF投资主题分析

```python
print("--- ETF 投资主题分析 ---")
themes = ['Technology', 'Healthcare', 'ESG', 'Real Estate', 'Emerging']
theme_analysis = {}

for theme in themes:
    theme_etfs = etfs.search(summary=theme)
    theme_analysis[theme] = len(theme_etfs)
    print(f"{theme} ETF数量: {len(theme_etfs)}")
print("\n--- 前5大ETF提供商分析 ---")

# 获取所有ETF数据
all_etfs = etfs.select()

# 定义用于数据清洗的函数
# 这个函数智能地从 'family' 或 'name' 列提取供应商名称
def get_provider_name(row):
    family_name = row['family']
    if pd.notna(family_name) and isinstance(family_name, str):
        return family_name
    
    name = row['name']
    if pd.notna(name) and isinstance(name, str):
        return name.split(' ')[0]
    
    return 'Unknown'

# 应用函数，创建干净的 'provider_clean' 列
all_etfs['provider_clean'] = all_etfs.apply(get_provider_name, axis=1)

# 在清洗过的数据上进行统计，并获取前5名
top_5_providers = all_etfs['provider_clean'].value_counts().nlargest(5)

# 打印结果
print("前5大ETF提供商:")
for provider, count in top_5_providers.items():
    print(f"{provider}: {count} 只ETF")
```

得到返回：

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/lvzhuoyang_B785_Fig06.png)

### 4.3 加密货币生态系统

```python
# --- 1. 初始化数据库 ---
# 初始化加密货币数据库。
# 如果是第一次运行，它会自动下载所需的数据。
try:
    cryptos = financedatabase.Cryptos()
    print("数据库加载成功！")
except Exception as e:
    print(f"数据库加载失败，错误信息: {e}")
    # 如果数据库加载失败，则退出程序
    exit()


# --- 2. 获取并分析数据 ---
print("\n开始分析加密货币交易对分布...")

# 从数据库中获取所有加密货币交易对，返回一个DataFrame
all_crypto_pairs = cryptos.select()
print(f"共找到 {len(all_crypto_pairs)} 个加密货币交易对。")

# 检查获取到的DataFrame是否为空，以避免后续操作出错
if not all_crypto_pairs.empty:
    # --- 统计并显示排名前10的加密货币 ---
    # 使用 pandas 的 value_counts() 方法高效地统计每个币种的交易对数量
    # 然后用 nlargest(10) 获取数量最多的前10个
    top_cryptos = all_crypto_pairs['cryptocurrency'].value_counts().nlargest(10)

    print("\n--- 前10大加密货币 (按交易对数量排名) ---")
    for crypto, count in top_cryptos.items():
        print(f"{crypto}: {count} 个交易对")
else:
    print("未能加载加密货币数据，或者数据为空。")
```

得到返回：

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/lvzhuoyang_B785_Fig07.png)

### 4.4 量化投资应用

```python
# 构建多因子选股池
def build_factor_universe():
    # 大盘价值股
    # (这里只是一个筛选示例，实际的价值因子需要更复杂的计算)
    large_value = equities.select(
        country='United States',
        market_cap='Large Cap',
        sector=['Communication Services', 'Consumer Discretionary', 'Consumer Staples']
    )

    # 小盘成长股
    # (这里只是一个筛选示例，实际的成长因子需要更复杂的计算)
    small_growth = equities.select(
        country='United States',
        market_cap='Small Cap',
        sector=['Communication Services', 'Consumer Discretionary', 'Consumer Staples']
    )

    # 使用 .index.tolist() 来获取股票代码列表
    return {
        'large_value': large_value.index.tolist(),
        'small_growth': small_growth.index.tolist()
    }

# 调用函数并查看返回结果
factor_universe = build_factor_universe()

# 打印部分结果以验证
print("大盘价值股 (部分):", factor_universe['large_value'][:5])
print("小盘成长股 (部分):", factor_universe['small_growth'][:5])
```

得到返回：

```
大盘价值股 (部分): ['002795.SZ', '0A40.L', '0A46.L', '0A4A.L', '0A4Z.L']
小盘成长股 (部分): ['06S.F', '0A3U.L', '0A4I.L', '0A4O.L', '0A7T.L']
```



### 4.5 主题投资研究

```python
# ESG投资标的挖掘
esg_etfs = etfs.search(summary='ESG')
clean_energy_etfs = etfs.search(summary='Clean Energy')

print(f"ESG相关ETF: {len(esg_etfs)} 只")
print(f"清洁能源ETF: {len(clean_energy_etfs)} 只")
```

得到返回：

```
ESG相关ETF: 1547 只
清洁能源ETF: 69 只
```

## 5. 总结

基于我们的实际测试，FinanceDatabase是一个定位清晰的金融数据工具：

**核心价值**：作为全球金融工具的"分类目录"，它解决了**标的发现问题** - 帮你找到"有哪些符合条件的股票/ETF/基金"，而不是"这些标的表现如何"。

**主要用途**：

- 量化策略的候选池构建
- 跨国市场结构研究
- 主题投资标的挖掘

**小结**：FinanceDatabase包含30万+金融工具的分类体系需要与其他工具配合才能进行深度分析。FinanceDatabase提供分类筛选，yfinance等工具提供价格数据，FinancialModelingPrep等API提供财务数据。理解其作为"工具发现引擎"而非"数据分析平台"的本质，是用好这个数据库的关键。对于需要系统性筛选金融工具的研究者和投资者而言，为金融分析提供了有价值的起点，但距离完整的分析解决方案还有距离。


------



