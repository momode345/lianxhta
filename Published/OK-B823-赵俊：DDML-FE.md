# B823：DDML-FE

两篇推文的工作量

- Clarke, P. S., & Polselli, A. (2025). Double machine learning for static panel models with fixed effects. Econometrics Journal. [Link](https://doi.org/10.1093/ectj/utaf011), [PDF](http://sci-hub.ren/10.1093/ectj/utaf011), [Google](<https://scholar.google.com/scholar?q=Double machine learning for static panel models with fixed effects>). [-Appendix-](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/ectj/PAP/10.1093_ectj_utaf011/2/utaf011_online_appendix.pdf?Expires=1752824167&Signature=WKaiVKLA-00aEZIMO9vx3dhZZgTLCbcqvACdqBy02ZNA6Vqkn-jZ4rl68WMd6y91~Mt9CCPUxQG9RwIdOPMzEbmv4~6u3RNzF8ghXmXTSU8yPREyG4f61PV2SPSBmwbRullB7tjgCDJFz3pKyewAVdjrXcupZSlFjDbGXQWeZzSL87jh9ZYpTK7BfpglF9bHX3uTjxtfTM3mttuqL3ID7W6cWkL238ggg~e7oR1LwCnUaA5x7mtOSwqD3cmDWB4m9jN7Fov2G308VLwHTDPdF-SFqU~~uWWX-8Edf04uDFu67FfTj~-2frvalFBDOF~zysYlVw9rYzHMCoNW7ejvqg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA),  [-Replication-](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/ectj/PAP/10.1093_ectj_utaf011/2/utaf011_replication_package_clarke_polselli.zip?Expires=1752824167&Signature=b9JFM59g8TeK1BrfE7-iqsJsuLLNwwwNtyYYxekBfIbm-ywodjDsGdHD8gnt8-Rzbk74zzXbo-9m2cb1iG2IUtkPdZwN1P-d4AWk11m6tQm6D9G529RMfLtf6RWqHpckGeiIJbBpeaGm8fYZ8Q-nbseMr0mRZDzBMPGGVx3Hqxrs-Fnki9JTf7RiO3B7thbh2h36r6CKEw67bIZhMDSbLTo10~Y~-6n5pbv-wss9r8X5IdjFGATQSFgqvBT4H2Xadl9JvidxaD9Jc2gnw67aptFhxNIuXn9JGcVKScn-QL7ZwCq8rMNvr6Wg92fTKl5GEpNp8yFPLKiOagFFKApjFQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA), 

## 3.2. Model and Assumptions

We extend the partially linear model proposed by Robinson (1988) to panel data by introducing a fixed effect $\alpha_i^*$ to give the partially linear panel regression (PLPR) model

$$
Y_{i t}=D_{i t} \theta_0+g_1\left(\boldsymbol{X}_{i t}\right)+\alpha_i^*+U_{i t}
$$

where $g_1$ is a non-linear nuisance function of $\boldsymbol{X}_{i t}$ and $E\left[U_{i t} \mid D_{i t}, \boldsymbol{X}_{i t}, \alpha_i^*\right]=0$, but $E\left[\alpha_i^* \mid D_{i t}, \boldsymbol{X}_{i t}\right] \neq 0$. The target parameter $\theta_0$ is the average partial effect of continuous $D_{i t}$ such that $d \theta_0=E\left[Y_{i t}(d)-Y_{i t}(0)\right]$, or the average treatment effect (ATE) $E\left[Y_{i t}(1)-Y_{i t}(0)\right]$ for binary treatments.

Following Chernozhukov et al. (2018), we focus on the partialled-out PLPR (PO-PLPR) model

$$
Y_{i t}=V_{i t} \theta_0+l_1\left(\boldsymbol{X}_{i t}\right)+\alpha_i+U_{i t}
$$


$$
V_{i t}=D_{i t}-m_1\left(\boldsymbol{X}_{i t}\right)-\gamma_i
$$

where $l_1$ and $m_1$ are nuisance functions, $\alpha_i$ is a fixed effect, $E\left[U_{i t} \mid V_{i t}, \boldsymbol{X}_{i t}, \alpha_i\right]=0$, and $V_{i t}$ is the residual of a non-linear additive noise treatment model, depending on fixed effect $\gamma_i$, and satisfying $E\left[V_{i t} \mid \boldsymbol{X}_{i t}, \gamma_i\right]=0$. The focus will be on the PO-PLPR rather than the PLPR model because, as shown below, $l_1$ is a conditional expectation over $Y_{i t}$ whereas $g_1$ is a conditional expectation over counterfactual $Y_{i t}(0)$ that must generally be learnt iteratively. Moreover, the orthogonalized score on which DML estimation of $\theta_0$ is based involves $V_{i t}$ whether it is derived under PLPR or PO-PLPR (see Section 5).

## 实现

[-Replication-](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/ectj/PAP/10.1093_ectj_utaf011/2/utaf011_replication_package_clarke_polselli.zip?Expires=1752824167&Signature=b9JFM59g8TeK1BrfE7-iqsJsuLLNwwwNtyYYxekBfIbm-ywodjDsGdHD8gnt8-Rzbk74zzXbo-9m2cb1iG2IUtkPdZwN1P-d4AWk11m6tQm6D9G529RMfLtf6RWqHpckGeiIJbBpeaGm8fYZ8Q-nbseMr0mRZDzBMPGGVx3Hqxrs-Fnki9JTf7RiO3B7thbh2h36r6CKEw67bIZhMDSbLTo10~Y~-6n5pbv-wss9r8X5IdjFGATQSFgqvBT4H2Xadl9JvidxaD9Jc2gnw67aptFhxNIuXn9JGcVKScn-QL7ZwCq8rMNvr6Wg92fTKl5GEpNp8yFPLKiOagFFKApjFQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

DML estimation is conducted in r using the XTDML package in the replication package (or accessible in its latest version at <https://github.com/POLSEAN/XTDML> at the time of writing) which is built on DoubleML by Bach et al. ([2024](javascript:;)).

## Tips

论文复现类推文写作指南参见 [连享会·助教答疑和推文指南](https://file.lianxh.cn/KC/lianxh_TA_Guide.pdf)，第 5 节。

可以参考 B813 的初稿生成过程，借助 AI 完成一部分内容。

- [B813-豆包对话过程](https://www.doubao.com/thread/w8e524bfc59a644e2))
- [B813-初稿](https://github.com/arlionn/lianxhta/blob/main/sample/B813-Stragged-DID.md)
