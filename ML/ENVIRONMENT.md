# 重要實驗環境

讀值日期：2026-11-28。主環境用於表格模型與結果整理；KEDGN及MA-GBT保留獨立環境。以下列重要套件，不宣稱歷史所有實驗都用同一環境。

| 類別 | 重要資訊 |
| --- | --- |
| OS | Microsoft Windows 11 家用版，64位元；Build 26300.10075 |
| Programming Language | Python 3.12.13（主環境）；Python 3.11.2（KEDGN／MA） |
| Tools | git version 2.49.0.windows.1；uv 0.11.8 |
| Libraries | XGBoost 3.4.1、LightGBM 4.7.0、scikit-learn 1.9.1、NumPy 2.5.4、pandas 3.0.6、Matplotlib 3.11.2；KEDGN用PyTorch 2.1.2+cu121 |

## 主要套件在不同環境的版本

| 套件 | 主環境 | KEDGN GPU | MA-GBT |
| --- | --- | --- | --- |
| xgboost | 3.4.1 | — | — |
| lightgbm | 4.7.0 | — | — |
| scikit-learn | 1.9.1 | 1.3.2 | 1.5.2 |
| numpy | 2.5.4 | 1.25.2 | 1.25.2 |
| pandas | 3.0.6 | 2.1.4 | 2.2.3 |
| torch | — | 2.1.2+cu121 | 2.1.2+cpu |

主環境另用SciPy 1.18.1、Joblib 1.6.0、PyArrow 26.0.0。表格中的「—」表示此摘要未列版本，不代表一定未安裝。

KEDGN使用PyTorch隨附的CUDA12.1 runtime（cu121）。GPU為NVIDIA GeForce RTX4060 Laptop GPU，8GB；不將此runtime版本寫成另外安裝的完整CUDA Toolkit版本。Full／R1本輪colsample0.9保持原CPU hist演算法，以隔離特徵抽樣變更。

[機器讀取摘要](reports/P65_environment.json)。完整歷史套件與裝置紀錄以各階段本機environment／protocol檔為準；KEDGN作者requirements與本機環境的差異已在[P64稽核](reports/P64_kedgn_implementation_audit.md)揭露。
