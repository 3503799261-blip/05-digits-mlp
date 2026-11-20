# 数据与第三方来源

## 代码和依赖

本仓库代码、自编教学文本和模拟数据生成器使用根目录 MIT 许可证。NumPy、scikit-learn、Matplotlib 及它们的依赖保留各自许可证；依赖通过 pip 安装，不把其源码或虚拟环境打进交付 ZIP。

## Digits（项目 05 / 06）

- 通过 `sklearn.datasets.load_digits()` 读取 scikit-learn 随包提供的 1,797 个 8×8 手写数字样本，不额外请求网络。
- 来源：Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository. DOI: [10.24432/C50P49](https://doi.org/10.24432/C50P49)。
- [UCI 原始数据页及许可](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)：CC BY 4.0。原始数据集总规模与 scikit-learn 使用的子集不同。
- 变化：分类项目自行划分训练 / 测试集并做标准化；图像检索项目提取特征并划分查询集 / 图库；示例图来自这些样本。

## Wine（项目 07）

- 通过 `sklearn.datasets.load_wine()` 读取 178 个样本、13 个化学特征，目标是三种葡萄酒品种。
- 来源：Aeberhard, S. & Forina, M. (1992). *Wine*. UCI Machine Learning Repository. DOI: [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)。
- [UCI 原始数据页及许可](https://archive.ics.uci.edu/dataset/109/wine)：CC BY 4.0。
- 变化：自行划分数据，训练折内标准化与交叉验证；不把类别解释为葡萄酒质量评分。

CC BY 4.0 许可正文：[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)。

## 其他项目

01–04 的中文文本、FAQ 和 Markdown 文档是随项目自编的教学材料。08–12 的需求、客户、设备、观影交互、日序列由固定随机种子生成。它们不是从真实平台抓取的评论，不是公开业务基准，也不含真实客户信息。详见各项目的数据说明与 `metrics.json`。
