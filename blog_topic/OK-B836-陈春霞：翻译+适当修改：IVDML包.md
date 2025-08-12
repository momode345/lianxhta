# B836：翻译+适当修改：IVDML包

- <https://github.com/cyrillsch/IVDML>
  - Note: 你可以 Fork 这个仓库，然后就可以在自己的账号下打开编辑模式，看到原始 Markdown 文档并进行翻译和编辑了。 

## 简介

The IVDML package implements an instrumental variable (IV) estimator for potentially heterogeneous treatment effects in the presence of endogeneity as presented in Scheidegger, Guo and Bühlmann (2025). The estimator is based on double/debiased machine learning (DML) (Chernozhukov et al., 2018) and uses efficient machine learning instruments (MLIV) and kernel smoothing.

We assume that we observe $N$ i.i.d. copies of the model $Y_i=\beta\left(A_i\right) D_i+g\left(X_i\right)+\epsilon_i . Y_i$ is the response variable and $D_i$ is the treatment variable. $X_i$ are the observed covariates and $A_i$ is a univariate continuous covariate with respect to which we want to consider heterogeneity (usually $A_i$ is a component of $X_i$ ). $\epsilon_i$ is an error term that satisfies $E\left[\epsilon_i \mid X_i\right]=0$, but we have endogeneity, meaning that $E\left[\epsilon_i \mid D_i, X_i\right] \neq 0$. To deal with the endogeneity, we assume that we have access to an instrumental variable $Z_i$ that satisfies $E\left[\epsilon_i \mid Z_i, X_i\right]=0$. To goal is to conduct inference on the heterogeneous treatment effect $\beta(a)$ for some specific value $a$ of $A_i$.

We quickly describe the main idea of our estimator. Assume for a moment that the treatment effect $\beta(\cdot)=\beta$ is constant (i.e. does not depend on $A_i$ ). Then, $\beta$ is identified via $\beta=\frac{E\left[\left(Y_i-E\left[Y_i \mid X_i\right]\right)\left(E\left[D_i \mid Z_i, X_i\right]-E\left[D_i \mid X_i\right]\right)\right]}{E\left[\left(D_i-E\left[D_i \mid X_i\right]\right)\left(E\left[D_i \mid Z_i, X_i\right]-E\left[D_i \mid X_i\right]\right)\right]}$. Hence, an estimate $\hat{\beta}$ of $\beta$ can be obtained by estimating the so-called nuisance functions $f\left(Z_i, X_i\right)=E\left[D_i \mid Z_i, X_i\right]$, $\phi\left(X_i\right)=E\left[D_i \mid X_i\right]$ and $l\left(X_i\right)=E\left[Y_i \mid X_i\right]$ using arbitrary user-chosen machine-learning and calculating a sample version of the identification given above. In practice, this is done using a cross-fitting scheme (Chernozhukov et al. 2018).

If we allow for the treatment effect $\beta(\cdot)$ to depend on the univariate continuous quantity $A_i$, and $A_i$ is a component of $X_i$, we can make the identification $\beta(a)=\frac{E\left[\left(Y_i-E\left[Y_i \mid X_i\right]\right)\left(E\left[D_i \mid Z_i, X_i\right]-E\left[D_i \mid X_i\right]\right) \mid A_i=a\right]}{E\left[\left(D_i-E\left[D_i \mid X_i\right]\right)\left(E\left[D_i \mid Z_i, X_i\right]-E\left[D_i \mid X_i\right]\right) \mid A_i=a\right]}$. conditional on $A_i=a$. In the sample version, we then use a kernel function $K(\cdot)$ (for example the Gaussian kernel $K(t)=\exp \left(-t^2 / 2\right) / \sqrt{2 \pi}$ ) to estimate the conditional expectations given $A_i=a$.

For a detailed discussion of the method, we refer to Scheidegger, Guo and Bühlmann (2025). We now demonstrate, how the IVDML package is used in practice.

## Tips

论文复现类推文写作指南参见 [连享会·助教答疑和推文指南](https://file.lianxh.cn/KC/lianxh_TA_Guide.pdf)，第 5 节。

可以参考 B813 的初稿生成过程，借助 AI 完成一部分内容。

- [B813-豆包对话过程](https://www.doubao.com/thread/w8e524bfc59a644e2))
- [B813-初稿](https://github.com/arlionn/lianxhta/blob/main/sample/B813-Stragged-DID.md)
