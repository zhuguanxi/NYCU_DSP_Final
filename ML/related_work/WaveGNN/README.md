# WaveGNN

作者版本 2bd2c3c899db650c28658764a86224318cd7ec1c，P12 死亡，Raindrop 第一折。論文表 II 報告 AUROC 83.9%、AUPRC 49.4%。 [GitHub](https://github.com/USC-InfoLab/WaveGNN) · [原論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13068137/)

CPU 訓練、驗證、checkpoint 載入及測試已通過。完整作者切分讀到 9,590／1,199／1,199 筆。原尺寸、batch 4 的前六批耗時 122 秒，當時的累計估速約 18 秒／批。每輪 2,398 批，約 12 小時／輪的估算只反映初段，本機已停止原尺寸試跑。論文設定為 20 輪，報告三次執行。

流程測試從作者各切分抽取 8／4／4 筆，兩類各半，模型跑一輪並每批更新一次。測試 AUROC 75.0%、AP 83.33%。原依賴固定 CUDA 版 Torch，本機使用 CPU Torch 與同版 PyG。首次啟動缺少 six，隔離環境補齊後通過。

本機日誌在 [CPU 試跑](../../artifacts/related_work/WaveGNN/cpu_smoke_01/train.log) 和 [原尺寸初段](../../artifacts/related_work/WaveGNN/paper_baseline_retry.log)。
