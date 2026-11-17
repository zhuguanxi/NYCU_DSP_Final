# ATENet

作者版本 dc3149ca60f02dafd8e073b336479f280efa51e7，使用 P12-M 的住院死亡、Raindrop 第一折。論文報告 AUROC 85.54%、AUPRC 53.31%。 [GitHub](https://github.com/shlee-labs/ATENet) · [原論文與實作附錄](https://openreview.net/pdf?id=l0kR6m9NDT)

CPU 訓練、驗證、存檔及測試已通過。使用論文的兩個 attention heads、128 維時間嵌入、32 維 GRU、學習率 0.0001、α 0.01、β 0.1。流程測試是一折一輪、16／8／8 筆真實病例，batch 8。測試 AUROC 37.50%、AP 52.38%。完整排程需要五折各 20 輪及 batch 128。

作者程式多處直接呼叫 CUDA。工作副本把這些張量及模型轉至 CPU，並建立原程式用到的圖表輸出目錄。原始作者程式仍保留。 [本機測試日誌](../../artifacts/related_work/ATENet/cpu_smoke_paper_params/train.log)
