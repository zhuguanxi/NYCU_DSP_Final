# KEDGN

作者版本 c82bc004cd1d1305fc58ca7150369d78682fccad，使用 physionet 的住院死亡設定及作者第 5 折。論文 ChatGPT 變體在該欄報告 AUROC 87.9%、AUPRC 57.1%。 [GitHub](https://github.com/qianlima-lab/KEDGN) · [論文表 1](https://proceedings.neurips.cc/paper_files/paper/2024/file/7c04aea54c2a60a632a47bd451cd2849-Paper-Conference.pdf)

用官方 Set A 經作者腳本重建 3,997 筆序列，按作者已提交的結果陣列順序核對每筆 ID 和死亡標籤。採已附的 BERT 與 GPT 來源變數表示。CPU 工作副本修改寫死的 CUDA 呼叫、模型讀取位置與作者前處理成功後的非零結束碼。

首輪流程測試為 16／8／8 筆、batch 8、一輪，測試 AUROC 56.25%、AP 56.67%。作者原排程每輪 batch 256、10 輪和五個種子。 [本機小樣本測試日誌](../../artifacts/related_work/KEDGN/cpu_smoke_01/train.log)

另用作者第 5 折的完整 3,198／400／399 筆、原 batch 256、種子 0 跑完一輪。測試有 51 次住院死亡，AUROC 為 73.91%、AP 為 34.95%。實測一輪訓練與驗證約 3.09 分鐘。五種子的 50 輪若維持此速度，約需 2.6 小時，另計前處理及啟動。完整排程尚未完成。可用 CPU 入口的 first_epoch 重現這一輪。 [完整切分一輪日誌](../../artifacts/related_work/KEDGN/cpu_native_epoch1/train.log)
