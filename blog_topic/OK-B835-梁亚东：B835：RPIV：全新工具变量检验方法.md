# OK-B835-梁亚东：RPIV：全新工具变量检验方法

Scheidegger, C., Londschien, M., & Bühlmann, P. (2025). A Residual Prediction Test for the Well-Specification of Linear Instrumental Variable Models (Version 1). arXiv. [Link](https://doi.org/10.48550/arXiv.2506.12771) (rep), [PDF](https://arxiv.org/pdf/2506.12771.pdf), [Google](<https://scholar.google.com/scholar?q=A Residual Prediction Test for the Well-Specification of Linear Instrumental Variable Models (Version 1)>). [R codes](https://github.com/cyrillsch/RPIV_Application)

实操代码：

- [RPIV - R codes](https://github.com/cyrillsch/RPIV)
  - [RPIV Application](https://github.com/cyrillsch/RPIV_Application)

摘要：

- The linear instrumental variable (IV) model is widely applied in observational studies. The corresponding assumptions are critical for valid causal inference, and hence, it is important to have tools to assess the model's well-specification. The classical Sargan-Hansen J-test is limited to the overidentified setting, where the number of instruments is larger than the number of endogenous variables. Here, we propose a novel and simple test for the well-specification of the linear IV model under the assumption that the structural error is mean independent of the instruments. Importantly, assuming mean independence allows the construction of such a test even in the just-identified setting. We use the idea of residual prediction tests: if the residuals from two-stage least squares can be predicted from the instruments better than randomly, this signals misspecification. We construct a test statistic based on sample splitting and a user-chosen machine learning method. We show asymptotic type I error control. Furthermore, by relying on machine learning tools, our test has good power for detecting alternatives from a broad class of scenarios. We also address heteroskedasticity- and cluster-robust inference. The test is implemented in the R package `RPIV` and in the `ivmodels` software package for Python.
- Implementations of our test can be found in the R package `RPIV`, available at <https://github.com/cyrillsch/RPIV>, and as part of the `ivmodels` software package for Python (Londschien and Bühlmann, 2024), available at <https://github.com/mlondschien/ivmodels>.

## Tips

论文复现类推文写作指南参见 [连享会·助教答疑和推文指南](https://file.lianxh.cn/KC/lianxh_TA_Guide.pdf)，第 5 节。

可以参考 B813 的初稿生成过程，借助 AI 完成一部分内容。

- [B813-豆包对话过程](https://www.doubao.com/thread/w8e524bfc59a644e2))
- [B813-初稿](https://github.com/arlionn/lianxhta/blob/main/sample/B813-Stragged-DID.md)
