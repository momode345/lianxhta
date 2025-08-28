# B809：Stata 可视化：pheatplot-图示系数的显著性

写一篇推文，介绍这个命令的用法。

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250611232839.png)


## 参考资料

可以翻译 + 意译 + 适当修改和整合如下资料，得到最终的推文文本。建议借助 AI 工具来完成。 

- [How to make heatplot in Stata | Correlation Heat plot](https://www.youtube.com/watch?v=PMo8NovkBFc&pp=0gcJCdgAo7VqN5tD)


### How To Make Heatplot In Stata | Correlation Heat Plot

[How to make heatplot in Stata | Correlation Heat plot](https://thedatahall.com/how-to-make-heatplot-in-stata-correlation-heat-plot/)

## 安装

```stata
net install gr0099.pkg, all replace 
```

执行上述命令后，如下两份文件会自动下载到当前工作路径下：

```raw
online_supplement.pdf
pheatplot-examples.do
```

你可以使用 `shellout online_supplement.pdf` 命令打开 `online_supplement.pdf` 文件，也可以执行 `clickout` 命令，以便在屏幕上呈现上述文件。























---

> Update: `2025/6/26 21:50`
