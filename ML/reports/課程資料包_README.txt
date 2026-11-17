ICU 住院死亡分類課程資料包

範圍

本資料包保留研究主線的全數前處理模型輸入。P12 使用完整原 A/B/C 切分。eICU 使用公開 Demo v2.0.1，分別用前 48 小時與前 24 小時的合格隊列。每次 ICU 住院是一列，住院死亡為 1、存活為 0。資料表、臨床比較值與遮蔽位置分開保存。完整 eICU、原始資料表、受限資料、模型權重和模型預測均不在壓縮檔。

來源及授權

P12 授權為 Open Data Commons Attribution License v1.0。來源為 PhysioNet/Computing in Cardiology Challenge 2012，Silva、Moody、Scott、Celi、Mark 等，v1.0.0。
https://physionet.org/content/challenge-2012/1.0.0/
https://opendatacommons.org/licenses/by/1-0/

eICU Demo 與本壓縮檔的 eicu_demo/ 衍生資料依 Open Data Commons Open Database License v1.0 提供。資料來源為 Johnson、Pollard、Badawi、Raffa（2021）的 eICU Collaborative Research Database Demo v2.0.1。
https://physionet.org/content/eicu-crd-demo/2.0.1/
https://doi.org/10.13026/4mxk-na84
https://opendatacommons.org/licenses/odbl/1-0/

前處理包含移除無效哨兵與值域外數值、依入 ICU 時間裁切觀察、統一欄位及單位、彙總成模型欄位或逐小時陣列、定義隊列與分折，以及以包內替代鍵重編識別欄位。

P12

- ab_outcomes.parquet 有 A/B 的 8,000 次住院及 1,122 次死亡，含 A/B 來源與原 0–2 三折。c_outcomes.parquet 有 Set C 的 4,000 次住院及 585 次死亡。C 僅用於評估。
- ab_48h_features.parquet 和 c_48h_features.parquet 包含 435 個表格欄位和 140 個時間摘要。ab_24h_features.parquet 和 c_24h_features.parquet 用相同欄位結構，以前 24 小時重建。24 小時隊列分別為 7,817 和 3,906 次住院，標籤及分折透過 stay_key 連回 outcomes。整天時間欄位只能近似風險集。
- *_clinical_scores.parquet 是官方 SAPS-I／SOFA 比較分數。負哨兵轉成缺值。c_time_sensitivity.parquet 記錄 P41 的敏感度分析納入旗標和最後一筆序列觀測的相對分鐘。這些欄位不作模型輸入。
- *_48h_sequence.npz 的 cube 排列為住院次 × 20 通道 × 48 小時。NaN 表示未觀測。static 依 manifest 的三個連續欄位和兩組類別編碼排列，A/B 的 val_hidden 與 val_static 首軸為六個驗證情境。aug_hidden 和 aug_static 首軸 0–2 對應三個訓練折，3 對應全 A/B 重擬。布林遮蔽為真時須遮掉該通道。各個原訓練折的保留範圍由 ab_fold 識別。
- c_48h_masks.npz 的 full_* 是各欄位全遮的位置，matched_* 是 32 次遮蔽的位置。皆為 c_48h_features 的零起算列序。matched_all_six 同時套用三種對齊欄位的 matched_* 與原未對齊三項（Mg／Urine／MechVent）的 full_*。情境定義見 manifest。c_24h_masks.npz 保存 P49 的三情境布林矩陣及 ICU 類型遮蔽。

eICU Demo

- outcomes_48h.parquet 有 782 次住院及 91 次死亡。outcomes_24h.parquet 有 1,398 次住院及 121 次死亡，包含 48 小時隊列。兩個時長共用 stay_key、patient_key、hospital_key。hospital_block 與 demo_fold 按各隊列獨立編號。demo_fold 為各住院的折外測試折（0–4），同一折內訓練和測試患者與院鍵區塊互斥。48 小時折用於 P44／P45，24 小時折用於 P50。
- 48h_*_features.parquet 保存原對齊或尿量毫升對齊各自的整點（hourly）與原始間隔（native）模型欄位。命名相同的 *_sequence.npz 含序列；排列和 P12 相同。24h_urine_mapped_*_features.parquet 是 24 小時隊列的兩種取樣。
- apache_iva_hospital_risk 為發布的 24 小時 APACHE IVa 住院死亡比較值，有效範圍 0–1。缺列或無效值為空白。48 小時與 24 小時分別有 660 和 1,166 次住院具有效分數。APACHE 機率僅用來評估比較。

讀取及核對

以 pandas/pyarrow 讀取 Parquet。依 stay_key 連結特徵與標籤；以 NumPy np.load(..., allow_pickle=False) 讀取 NPZ，先核對 stay_key 順序。435 欄的定義見 feature_dictionary_435.parquet；時間摘要的欄名和統計量見 manifest.json。所有來源及封裝檔案的 SHA-256 均列在 manifest 或壓縮檔旁的 SHA256SUMS.txt。

本版到 P50。P12 已用於多輪比較，eICU Demo 也曾用於流程開發與增強資料。正式外部確認需要獨立患者資料。
