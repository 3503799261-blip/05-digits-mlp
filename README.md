# 手写数字识别：小型 MLP 与线性基线

用一个 64 单元隐藏层的神经网络识别 0–9 的手写数字。项目只有一个训练脚本，CPU 可运行，首次安装依赖后完全离线。

## 运行

Python 3.11 / 3.12，在本目录执行：

```bash
python -m pip install -r requirements.txt
python main.py
python main.py --sample-index 12 --output-dir outputs
```

`--sample-index` 指固定测试集中的索引，不是上传图片；有效范围运行错误提示会显示。每次运行重新训练，默认结果存本目录 `outputs/`。

## 数据与来源

- `sklearn.datasets.load_digits()` 内置的 1,797 张 8×8 灰度图片，64 个像素特征、10 个类别；无需联网下载。
- 原始来源：[UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)，E. Alpaydin、C. Kaynak；数据许可见该页的 CC BY 4.0 声明。代码许可不替代数据许可。
- [scikit-learn 数据说明](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)。这是 digits，不是 MNIST。

## 算法与评估

1. 固定随机种子 42，按类别分层划分 75% 训练、25% 测试。
2. `StandardScaler → MLPClassifier(64, ReLU)`；标准化只在训练集拟合。MLP 从训练集内部保留 15% 做 early stopping，不使用外部测试集选迭代次数。
3. 同一测试集比较多数类 `DummyClassifier`、标准化后的逻辑回归、MLP。
4. 输出 Accuracy、Macro-F1、混淆矩阵和误分类图片。实际分数由本机运行产生，不能把示例当成真实业务效果。

## 结果文件

- `metrics.json`：各模型的测试指标、切分大小、MLP 实际训练轮数。
- `predictions.csv`：每张测试图片的原始索引、真值、两个模型预测。
- `confusion_matrix.png`、`error_examples.png`：错误分布及部分失败案例。
- `example_prediction.json/png`：选中测试样本的预测和图片；类别概率没有做单独校准。

## 可以在面试中讲什么

解释多分类交叉熵、隐藏层作用、为什么要做标准化、early stopping 如何抑制过拟合；用逻辑回归基线说明神经网络是否真的带来提升，分析最容易混淆的数字。

## 局限和可做的小改进

图片非常小，随机按图片切分不等于按书写者切分，无法证明跨书写者泛化；也没有实现手机拍照、旋转矫正或任意分辨率图片识别。可以只改一个因素，如隐藏单元数，保留验证集选参，再在未使用的测试集汇报结果。

## 实际示例结果

见[本次实际运行结果](RESULTS.md)，包含生成的指标和结果图。重新运行 `python main.py` 可以复现流程。
