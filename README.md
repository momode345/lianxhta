> Update: `2025/6/11 15:51`

## 1. 答疑工作指南

请查看：[TA01：连享会助教答疑手册](https://github.com/arlionn/lianxhta/blob/main/_doc_/TA01-QA.md)

&emsp; 

## 2. 推文选题和发布

> &#x1F34E; 完成后的推文务必提交到坚果云：[点击提交](https://workspace.jianguoyun.com/inbox/collect/57246515e7e142b68971ac22f2bda5d0/submit)  
详见 Section 2.3。  
> 推文格式参见往期推文范本：<https://github.com/arlionn/lianxhta/tree/main/template>

### 2.1 认领选题

1. **注册 Github 账号** (已有可跳过此步骤)。登录 github 后访问 [arlionn/lianxhta](https://github.com/arlionn/lianxhta) 仓库。

2. **Fork [arlionn/lianxhta] 仓库**。[-点击此处-](https://github.com/arlionn/lianxhta/fork) Fork 本仓库 (已 Fork 可跳过)。Fork 后，你的账号下会生成一个同名仓库：`https://github.com/你的账号/lianxhta`。

3. **选题-更改文件名**。
   - 在仓库 (`你的账号/lianxhta`) 中的【&#x1F4C2; **blog\_topic**】文件夹中浏览选题；
   - 选定题目后，按如下规则修改 **页面标题** (&#x1F34E; 注意：是页面标题，而不是正文中的标题)
     - 旧文件名：`B723：推文标题.md` 
     - 新文件名：`OK-B723-姓名：推文标题.md`
   - 注意：(1) 请严格按照上述格式修改，不要随意添加空格和其他符号；(2) `OK` 表示该选题被选中了，正在做。
4. **同步仓库** (初次选题的同学可以跳过此步骤)。如果你在几天前 Fork 了主仓库或准备进行第二次选题，那么请先同步主仓库中的最新代码到你账号下的副本仓库中。具体操作如下 (参见补充说明 1)：
   1. 点击页面右上角的 **Sync fork** 按钮。
   2. 在弹出的对话框中，点击 **Update branch** 按钮。
   3. 等待更新完成后，点击 **View code** 按钮返回代码页面。
5. **本地提交**。点击右上角绿色按钮「**Commit changes**」提交修改。
   - 说明：下方有两个选项，默认选第一个 `Commit directly to the main branch`。
   - 注意：目前，你只是在你的账号下完成了选题修改，但**还没有**提交到主仓库中。请进行下一步。
6. **提交到主仓库中**。在你 Fork 的 `你的账号/lianxhta` 仓库中，点击左上方主菜单中的「**Pull requests** (提交申请)」选项卡，点击「**New pull request**」，选择 `main` 分支，点击「**Create pull request**」，填写标题和描述后提交。
7. **审核**。提交后，等待连老师审核。审核通过后，你会收到邮件通知。
8. **同步本地仓库**。返回你名下的仓库，按照第 4 步的方法，点击 **Sync fork** 按钮，更新你的 Fork 仓库，便可在你名下的仓库中看到最新的选题列表。


#### 注意事项 &#x1F34F;

1. **同步问题**。每次修改选题前，请务必先查看你 Fork 的仓库是否与主仓库保持同步。
   - **需要更新**：如果显示 `This branch is xxx behind arlionn/lianxhta:main`，请依次点击 **Sync Fork** &rarr; **Update xx**，以便拉取主仓库的最新代码。
   ![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250428090708.png)
   完成后你将看到：`This branch is up to date …….` 
   - **不需要更新**：如果显示 `This branch is up to date with arlionn/lianxhta:main`，则说明你 Fork 的仓库已经是最新的了。
2. **自荐选题**。如果你有好的选题 (比如你感兴趣的新方法和模型，或正在精读的文献等)，可以编写一个提纲给我，我会将其添加到选题列表中。提纲为  `.md` 格式，包含选题标题、简介、选题的理由和大致内容框架。 
3. **先到先得**。多人同时认领同一选题时，先认领的优先。
2. **30 天期限**。选题认领后，请在 **30 天** 内完成，否则该选题会被收回。
3. &#x1F34E; 每人 **只能认领一个选题** ，完成并提交至 [坚果云](https://workspace.jianguoyun.com/inbox/collect/57246515e7e142b68971ac22f2bda5d0/submit) 后，才能再认领新的选题。 
4. 如果需要**更换选题**，需要预先跟连老师确认，同意后方可更换。  

### 2.2 推文撰写

- **整体风格和篇幅**：可以在 [lianxh.cn](https://www.lianxh.cn) 中搜索相关主题，查看已有的推文，了解行文风格和格式。可以私信连老师或李晓燕老师获取推文 Markdown 原文。
- **推文格式**：[template](https://github.com/arlionn/lianxhta/blob/main/template/readme.md) 文件夹中放置了一些已经发布的推文，可以参考其行文风格和格式。
- **数学公式**：
  - LaTeX 公式录入方法参见：[Markdown中书写LaTeX数学公式简介](https://www.lianxh.cn/details/250.html)
  - **公式识别**：可以使用 [Mathpix](https://www.lianxh.cn/details/284.html) (收费) 或者 [simpletex](https://simpletex.cn/) (推荐，免费) 来识别数学公式。
- **图片**：
  - **本地图片的命名规则**：请严格按照 `作者汉语拼音_推文编号_Figxx.png` 格式命名图片，例如：`lianyujun_B782_Fig02.png`。由于我们所有人的图片都放置在同一个图床，同名文件会自动「旧的覆盖新的」。如果图片名称过于简单，不但会导致你的图片无法正常显示，也会导致此前的推文图片引用错误。(Update: `2025.8.14`) 
  - 如果截图后使用 [PicGo 软件](https://github.com/arlionn/lianxhta/wiki/lianxh_PicGo) 上传图片，并在 **PicGo 设置** 中选择了「时间戳重命名 (`开`)」，它会自动利用时间戳作为图片名称，通常不会产生冲突。
  - 可以使用 [Snipaste](https://www.lianxh.cn/details/1111.html) 截图工具，直接截图到剪贴板中，然后使用 PicGo 软件上传到连享会图床，将产生的链接 (已自动存入剪切板) 粘贴到 Markdown 文档中。Snipaste 也可以将截图保存为图片文件，方便后续使用。
  - **图片上传到连享会图床**：所有插入连享会推文和答疑文档的图片都必须上传到连享会的图床上，设置方法：[连享会 PicGo 图床配置](https://github.com/arlionn/lianxhta/wiki/lianxh_PicGo)。
- **借助 AI 写推文**：参见连玉君, 2025, **Empirical Research with AI**，[Chapter 7-9](https://lianxhcn.github.io/research_with_AI/body/05_00_read_paper_with_AI.html) 介绍的方法和提示词。 

### 2.3 推文提交

> 完成后的推文务必提交到坚果云：[点击提交](https://workspace.jianguoyun.com/inbox/collect/57246515e7e142b68971ac22f2bda5d0/submit)

- 版本号为：v1, v2 ……
  - 提交完 v1 后，若有修改或更新，可以直接以 v2 再次提交，以此类推。 
- 文件名格式：`v1-B723-姓名：推文标题.md`。
  - 请勿删除推文编号 `B723`，以便于后续的选题统计。
- 不需要提交到码云了 (`2025/5/6`)。

### 2.4 推文发布后

- **查询**。推文发布后，你可以在 [lianxh.cn](https://www.lianxh.cn) 搜索，亦可使用 [lianxh](https://www.lianxh.cn/details/1296.html) 命令查询。
- **跟进**。如果你提交推文 10 天后还没有收到连老师的修改意见，也没有在 [lianxh.cn](https://www.lianxh.cn) 中搜索到你的推文，可以私信连老师询问推文进展。

<br>
<br>

<br>
<br>
<br>
<br>

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/Lianxh_装饰黄线.png)

## 附：操作过程图示

> Note: 如下图片由助教杨云帆同学提供，感谢他的贡献！

### 1. 同步仓库

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefined20250429170014.png)

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefined20250429170036.png)

### 2. 认领选题：修改文档标题
![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefined20250429170051.png)

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250506234155.png)

### 3. 提交到主仓库
![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefined20250429170136.png)

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefined20250429170156.png)

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/undefined20250429170214.png)
