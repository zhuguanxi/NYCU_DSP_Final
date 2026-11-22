# P60：固定參數的五種子穩定性

2026-11-22，開始前固定。原 A/B 三折已決定全部參數，本輪不重新選參、輪次、欄位、混合權重或最佳種子。固定種子 42、43、44、45、46，全 A+B 8,000 次住院／1,122 次死亡，各重訓原 Full／Elastic、GCS-only Full／Elastic、P26 的 temporal／blind_lgb 及 P33 的 plain 8 輪 CNN。共 35 次元件擬合，組裝七個可執行模型：Full、Elastic、GCS-only Full／Elastic、P26、P28、P33。

欄位、前處理規則及選欄集合固定，補值與編碼由相同全 A/B 重新擬合。GCS 60% 增強位置及 blind_lgb 的跨庫模擬位置沿原訓練種子固定，僅改模型初始化／採樣種子，明確隔離此來源的變異。P26/P28 路由與權重沿原 A/B 選擇；P33 固定 75% P28 +25% CNN，8 輪、批次128、AdamW lr=.0015、weight_decay=.005，架構不變。seed42 的表格及分流元件須在 A/B 逐值重現原模型。

樹模型使用原 CPU 演算法；CNN 開始前檢查 GPU 程序，有其他運算者即用 CPU。訓練每輪再檢查，若別人開始使用則釋放 GPU、在 CPU 以同種子重跑該元件，記錄實際裝置。CPU 與 GPU 的訓練路徑可能產生數值差異，裝置亦為限制，不把這種差異當成純種子效果。

先封存五套權重與統計雜湊，才評估同一 Set C 的 clean、GCS 全遮、六項全遮及尿量對齊 Demo 整點／原始間隔。逐種子列 AP、AUROC、Brier、@0.5 敏感度、相對 Full／P28 差值及遮蔽退化；報五次平均、樣本標準差及範圍，不以最好種子作主結果。五種子機率平均若列出，必須另標集成，不與五次指標平均混淆。原作者 KEDGN A+B 的既有 seed0–4也逐一列出，不重訓、不以其結果調參；其種子索引與本組不可視為相同隨機實驗。

程式 `related_work/p60_seed_stability.py` 及獨立 Torch worker。主環境搭配 P21 CPU 套件，worker 使用 P54 GPU Python（閒置才使用 CUDA）。本機 `artifacts/p60_five_seed_stability/` 保存權重、資料、逐筆分數與日誌，公開只含彙總。程式／規格先 commit，結果另 commit並推 GitHub。Set C 與 Demo 先前已反覆查看，本輪固定配置補強穩定性，不冒充全新獨立確認。
