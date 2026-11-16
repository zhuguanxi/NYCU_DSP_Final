# P44 以真實 eICU Demo 病例增強訓練

依課程規格，把外部資料實際納入訓練。沿用原 P12 A/B 資料與 eICU Demo 的 48 小時資料、住院死亡標籤、欄位與尿量 mL 對齊。每次訓練加入指定的 Demo 整點 stay，每筆權重 1。P12 的總樣本權重和既有 P26 缺失增強設定照原規則。外部 stay 不另外增強。所有強度、模型參數與輸入對齊按既有開發結果與此固定規格，無 Demo 超參數選擇。全程 CPU。

- 重擬 P26 的時間窗 XGBoost 與 GCS 全遮 LightGBM。使用原 A/B 的 650 棵、種子、特徵順序及 GCS 分流。保留原 P28 的 GCS-only Elastic 成分和混合權重。
- Demo 用固定種子的 5 折分層群組切分，以共享患者的 `hospitalid` 相連區塊為分組，核對患者和區塊都不跨同一折的訓練與測試。官方範例頁文字提到 20 家，但發布的兩份原表有 186 個相符的 `hospitalid`，因此本研究使用發布檔案的院鍵分組。整點、原始間隔及原／尿量對齊均用同一套折外模型。合併各折的預測比較凍結 P28、Elastic，並用 1,000 次患者配對重抽。 [eICU Demo 發布頁](https://physionet.org/content/eicu-crd-demo/2.0.1/) · [院鍵資料字典](https://eicu.mit.edu/eicutables/hospital/)
- 另以全 A/B 和所有合格 Demo 病例擬合一次，沿用未接觸 Set C 訓練的全模型評估原 13 種遮蔽，對照凍結 P28 與原 XGBoost。Demo 曾用於多輪映射和分析，這是同一來源的跨折探索比較。完整版獨立驗證沿用 [P40 規格](P40_eicu_sequence_confirmation_protocol.md)。

在 `ML` 使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md)，執行 `& $py src/evaluate_real_eicu_augmentation.py`。完整彙總與模型留在 Git 排除的 `artifacts/p44_real_eicu_augmentation/`。
