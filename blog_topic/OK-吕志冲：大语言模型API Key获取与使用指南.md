> 作者：吕志冲 (西南交通大学)  
> 邮箱：lyuzhichong@my.swjtu.edu.cn

> ⚠️ 免责声明：本文内容仅为学习交流所用，不构成任何商业推广或使用建议。

在使用各类大语言模型时，获取并管理**API Key**是第一步，也是最重要的安全与成本控制环节。本节将系统介绍如何从零开始申请并使用API Key，涵盖两类典型途径：
 *其一*，通过**OpenRouter**这样的第三方集成平台，一次接入即可调用多家厂商的主流模型；
 *其二*，直接在**Deepseek**这样的单一平台上开通官方API，适合对特定模型有深度需求的场景。

接下来的两节将依次说明这两种方式的注册、密钥创建、调用示例以及费用管理，为后续模型接入与开发打下完整基础。

# 1. 第三方集成平台介绍—以OpenRouter为例

OpenRouter是一个集成多家前沿大语言模型的统一访问平台。它将来自OpenAI、Anthropic、Google、DeepSeek等厂商的主流与特色模型集中在同一接口下，为开发者和研究者提供一站式的模型调用与管理服务。

✨ **主要特点与优势**

- 🧩 **模型齐全**：涵盖GPT、Claude、Gemini、DeepSeek、LLaMA等，持续引入新兴模型。
- 🔗 **接口统一**：不论供应商哪家，都可用一致的API规范调用。
- 🟢 **体验门槛低**：提供部分免费模型与在线Chat体验，方便试用。
- 💳 **计费透明**：按调用量付费，用多少算多少。
- 🌐 **国内直连**：无需额外跨境网络工具即可注册和使用。

通过以上优势，OpenRouter既适合初学者快速上手，也满足科研与企业在多模型环境下的深度开发需求。了解OpenRouter的定位与优势后，接下来可以动手创建账号，正式开始使用。

## 1.1 注册OpenRouter账号

🔗 访问官网：https://openrouter.ai/

👉 点击右上角**Sign In**

📨 使用GitHub、Google等账号或个人邮箱注册登录

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913102251287.png)

## 1.2 获取API Key

完成注册并登录后，就可以生成自己的API Key，这是调用模型的前提。有两个路径创建API密钥：

🛠️ **方法一：** 点击右上角的头像，选择`Keys`。然后点击`Create Key`，创建一个新的API密钥。

🧭 **方法二：** 在`Models`中选择具体模型，然后点击`API`，不过也是跳转到方法一的页面。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913104204324.png)

`API Key`创建说明：

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913104111187.png)

## 1.3 API Keys使用说明

有了API Key，就可以选择所需模型并在项目中调用，下面将介绍具体操作方式。

首先，选择`Models`，然后在搜索框输入所需模型名称。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913130252557.png)

以GPT-4o模型为例，调用API介绍如下：

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913132449002.png)

## 1.4 余额查看与充值说明

在正式调用模型前，建议先熟悉余额查询和充值步骤，以便合理规划成本。首先，按照顺序进入余额界面。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913125904245.png)

接着，完善个人信息。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913105545085.png)

信息填写完毕后，进入支付界面。推荐使用支付宝/微信扫码支付，也可采用银行卡等支付方式。若使用支付宝/微信支付需在以下界面的下方勾选`Use one-time payment methods`。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913105720813.png)

最后，进入下方支付界面，填写支付金额后可选择支付方式并扫码完成支付。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913110030089.png)

至此，从OpenRouter平台概览到注册、密钥获取、模型调用，再到账户充值的完整流程已经介绍完毕。

# 2. 单一平台介绍—以Deepseek为例

相较于可一次接入多家模型的第三方平台，有些应用场景更适合**直接使用单一模型提供商的官方API**，以获得更高的稳定性与更细致的控制。Deepseek是国内团队推出的高性能大语言模型平台，具有**⚡ 推理速度快、💰 成本可控、🀄 中文优化** 等优势，为科研和企业应用提供了稳定的模型服务。

在这一小节中，将简要介绍如何注册Deepseek账号、创建并管理官方API Key，并展示基础调用示例，帮助用户在无需中间平台的情况下直接体验和部署 Deepseek 模型。

## 2.1 注册Deepseek 账号

开始之前，需要先注册并登录Deepseek账号：

1. 🔗 访问官网：https://www.deepseek.com/
2. 👉 点击右上角**API 开放平台**
3. 📨 使用手机号、微信或邮箱登录

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913145400246.png)

完成注册后，即可进入平台的开放接口管理界面，为下一步创建API Key做准备。

## 2.2 获取API Key

完成注册并登录后，可立即创建自己的API Key：

1. 🔑 在左侧菜单点击**API keys**
2. ➕ 选择**创建 API key**
3. ✍️ 输入自定义名称并生成密钥

官方入口：https://platform.deepseek.com/

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913145712784.png)

>  ⚠️ **重要提示**：请将此API key保存在安全且易于访问的地方。出于安全原因，你将无法通过API keys管理界面再次查看它。如果你丟失了这个 key，将需要重新创建。

## 2.3 API Keys使用说明

进入**开放平台**界面后，可点击**接口文档**查看各类模型的详细调用方式和参数示例。开发者可直接参考官方示例代码快速完成模型接入。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913150646868.png)

具体代码说明如下：

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913151038370.png)

💡 **费用控制**：建议提前浏览 **模型 & 价格** 页面，了解不同模型的 token 单价及计费规则，以便合理规划预算。
![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913151014416.png)

**扣费规则：** 扣减费用 = token 消耗量 × 模型单价，对应的费用将直接从充值余额或赠送余额中进行扣减。 当充值余额与赠送余额同时存在时，优先扣减赠送余额。

## 2.4 余额查看与充值说明

为确保调用不中断，可随时查看余额并进行充值：

1. 💰 **余额查看**：在账户主页查看剩余额度

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913151615236.png)

2. 💳 **充值操作**：点击充值按钮后选择合适的支付方式（支持微信、支付宝等）

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250913151732881.png)

通过以上流程，可以在不借助任何第三方聚合平台的情况下，直接实现 Deepseek 模型的安全调用与成本管理。

# 3. 总结

从 **第三方集成平台—OpenRouter** 到 **单一平台—Deepseek**，本文介绍了两类常见的API Key使用路径及其适用情境：

- 🔗 **OpenRouter**：提供统一接口，可一次接入多家前沿大语言模型，适合需要跨模型对比、快速切换或统一管理的科研与开发项目。
- ⚡ **Deepseek**：突出单一平台的直接性与针对性，尤其适合注重调用稳定、中文优化或成本可控的企业与个人研究者。

综合来看，读者可依据自身需求灵活选择：若强调多模型集成与扩展性，可首选**OpenRouter**；若侧重性能稳定与本土化优势，可直接采用**Deepseek**官方API，从而在不同的研究或应用场景中高效完成大语言模型的接入与管理。

> ⚠️ 免责声明：文中涉及的平台信息请读者自行判断选择，如有后续变动或使用问题，均与推文作者和连享会平台无关。

# 4. 相关推文

> Note：产生如下推文列表的 Stata 命令为：  
>   `lianxh`  
> 安装最新版 `lianxh` 命令：  
>   `ssc install lianxh, replace`  

- [连享会](https://www.lianxh.cn/search.html?s=连享会), 2024, [AI编程助手大盘点：不止ChatGPT和Copilot](https://www.lianxh.cn/details/1394.html), 连享会 No.1394.
- [连小白](https://www.lianxh.cn/search.html?s=连小白), 2025, [DeepSeek对话可以分享了！](https://www.lianxh.cn/details/1566.html), 连享会 No.1566.
- [连玉君](https://www.lianxh.cn/search.html?s=连玉君), 2024, [VScode：实用 Markdown 插件推荐](https://www.lianxh.cn/details/1390.html), 连享会 No.1390.
- [连玉君](https://www.lianxh.cn/search.html?s=连玉君), 2024, [借助ChatGPT4o学习排序算法：AI代码助手好酸爽！](https://www.lianxh.cn/details/1393.html), 连享会 No.1393.
- [连玉君](https://www.lianxh.cn/search.html?s=连玉君), 2025, [老连买电脑：ChatGPT，DeepSeek，豆包来帮忙](https://www.lianxh.cn/details/1561.html), 连享会 No.1561.
- [连玉君](https://www.lianxh.cn/search.html?s=连玉君), 2023, [连玉君：我与ChatGPT聊了一个月](https://www.lianxh.cn/details/899.html), 连享会 No.899.
- [罗银燕](https://www.lianxh.cn/search.html?s=罗银燕), 2023, [如何在 R 中安装并使用 chatgpt 包？](https://www.lianxh.cn/details/1171.html), 连享会 No.1171.
- [王烨文](https://www.lianxh.cn/search.html?s=王烨文), 2025, [LLM Agent：大语言模型的智能体图解](https://www.lianxh.cn/details/1650.html), 连享会 No.1650.
- [吴小齐](https://www.lianxh.cn/search.html?s=吴小齐), 2024, [强大的Kimi：中国版ChatGPT平替](https://www.lianxh.cn/details/1423.html), 连享会 No.1423.
- [于凡](https://www.lianxh.cn/search.html?s=于凡), 2024, [AI可以编写Stata代码吗？](https://www.lianxh.cn/details/1348.html), 连享会 No.1348.
- [余坚](https://www.lianxh.cn/search.html?s=余坚), 2023, [Stata：ChatGPT你能帮我干点啥？](https://www.lianxh.cn/details/1164.html), 连享会 No.1164.
- [张弛](https://www.lianxh.cn/search.html?s=张弛), 2025, [大语言模型到底是个啥？通俗易懂教程](https://www.lianxh.cn/details/1600.html), 连享会 No.1600.
- [赵文琦](https://www.lianxh.cn/search.html?s=赵文琦), 2025, [LLM系列：ChatGPT提示词精选与实操指南](https://www.lianxh.cn/details/1615.html), 连享会 No.1615.

