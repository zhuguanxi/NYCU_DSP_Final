# ICU 機器學習探索：階段報告索引與總結

[2026 年 11 月 21 日完整研究進度](reports/研究進度總覽.md) 涵蓋 P0–P54 和論文重現。原期末 PDF 整理截至 P51，P52 及後續文獻日誌核對另見最新進度。

[P54：A+B 重訓 KEDGN 的 Set C／eICU Demo 結果](reports/P54_kedgn_ab_transfer_results.md) 使用原 A/B 三折選共同輪次，全 A+B 訓練五個種子，再與 [P53 Set A 凍結版](reports/P53_kedgn_frozen_transfer_results.md) 及原模型做同病例比較。訓練使用 GPU，資料與權重不提交 Git。R1 已取消，其他論文背景重跑已停止。

## 目前可核對的主要結果

| 已評估情境 | 模型與 AP | 相對對照的配對 AP 差（95% 區間） |
| --- | --- | --- |
| Set C 未遮蔽 | P33，0.60447 | 對凍結 Full +0.02462 [+0.01109, +0.03775] |
| Set C GCS 全遮 | P26，0.53733 | 對 GCS-only Full +0.03241 [+0.01363, +0.05132] |
| eICU Demo 尿量對齊整點，患者／院鍵區塊五折 | P45，0.51115 | 對凍結 P28 +0.00971 [−0.01335, +0.03271] |

P33、P26 評估沿用完全未用於擬合的 Set C。Set C 曾多次用於比較。P45 使用真實外部病例增強，Demo 的同折訓練與測試患者互斥。外部 AP 改善仍不確定。取得有授權的完整版 eICU v2.0 後，可採 P46 固定入口排除所有 Demo 患者並比較。 [Set C 時間敏感度](reports/P41_p12_outcome_timing_results.md) · [真實外部病例增強](reports/P44_real_eicu_augmentation_results.md) · [外部增強分支](reports/P45_augmentation_branch_ablation_results.md) · [完整版確認入口](reports/P46_eicu_adaptation_confirmation_protocol.md)

## 目前研究進度

- [組員研究報告第 10 節](實驗報告_組員分享版.md#10-缺失韌性與跨資料集評估) 整理原版 MA-GBT、缺失增強、時間窗分流與混合模型的共同設定。
- [Set C 固定全遮蔽的配對比較](reports/P30_full_mask_paired_results.md) 報告相對凍結對照的 AP 與 AUROC 差及區間。
- [eICU Demo 混合模型](reports/P28_robust_ensemble_results.md) 與 [GCS、通氣欄位對齊](reports/P32_eicu_ventilation_results.md) 保留兩種取樣與各對照的結果。
- [P33 逐小時序列模型](reports/P33_hourly_sequence_results.md) 提高 Set C 未遮蔽 AP，eICU Demo 的 AP 點估計較 P28 低。
- [P34 四小時序列與量測分布](reports/P34_sequence_cadence_results.md) 比較等量隨機與末段遮蔽，檢查逐小時模型的跨庫差異。
- [P35 權重門檻](reports/P35_sequence_gate_results.md) 與 [P36 通道別增強](reports/P36_profile_sequence_results.md) 用 A/B 比較 eICU 觀測小時分布模擬，兩輪均維持固定四小時模型。
- [P37 序列分數修正](reports/P37_sequence_blend_results.md) 用 A/B 選擇對數勝算修正幅度，保留原模型後重跑 Set C 和 eICU Demo。
- [P38 序列通道消融](reports/P38_sequence_channel_ablation_results.md) 比較從四小時卷積移除尿量、呼吸率及兩者，表格模型沿用原輸入。
- [P39 序列種子與平均](reports/P39_sequence_seed_ensemble_results.md) 用 A/B 三折比較通道消融的種子敏感度及三網路平均。
- [P29 完整 eICU 確認規格](reports/P29_eicu_ensemble_confirmation_protocol.md) 有 CPU 執行入口。正式結果等已授權的 eICU v2.0 本機資料，先排除所有 Demo 患者再計分。
- [P40 完整 eICU 次要比較](reports/P40_eicu_sequence_confirmation_results.md) 可在 P29 同一隊列上核對 P38。Demo 試跑檢查 CPU 流程和凍結分數。
- [P41 P12 時間欄位稽核](reports/P41_p12_outcome_timing_results.md) 核對官方結果表與時間敏感度隊列的凍結 Set C 分數。
- [P42 機率校準與分類門檻](reports/P42_frozen_calibration_results.md) 用 A/B 折外預測擬合，檢查 Set C 與 eICU Demo 的機率誤差和分類表現。
- [P43 eICU 醫院層級敏感度](reports/P43_eicu_hospital_variation_results.md) 用原凍結分數核對醫院重抽、逐區塊省略及院內排序。
- [P44 真實外部病例增強](reports/P44_real_eicu_augmentation_results.md) 將 eICU Demo 合格病例納入 P12 A/B 訓練，評估獨立 Set C 與患者分折的 Demo。
- [P45 外部增強分支消融](reports/P45_augmentation_branch_ablation_results.md) 在同一隊列拆開有 GCS 與無 GCS 分支的外部增強。
- [P46 完整 eICU 的 P45 比較](reports/P46_eicu_adaptation_confirmation_results.md) 提供含 P45 的 CPU 驗證入口及患者排除測試結果。
- [P47 P12 拆分稽核](reports/P47_p12_split_overlap_results.md) 核對原始紀錄、435 維凍結特徵與高重疊動態觀測。
- [P48 臨床評分配對比較](reports/P48_clinical_score_baselines_results.md) 在共同隊列比較 SAPS-I、SOFA、APACHE IVa 和凍結模型，附近年文獻的切分及標籤差異。
- [P49 24 小時觀察窗](reports/P49_24h_landmark_results.md) 用 A/B 選定 CPU 模型，在 Set C 測試遮蔽，並於 eICU 同一隊列比較前 24 小時的模型和 APACHE IVa。
- [P50 24 小時真實外部增強](reports/P50_24h_eicu_augmentation_results.md) 使用互斥的患者與院鍵區塊五折，檢查增強後 AP 與 Set C 的 GCS 全遮結果。
- [P51 課程資料包](reports/P51_course_dataset_package_results.md) 保存截至 P50 的模型輸入、分折與本機 ZIP 驗證紀錄。
- [期末研究報告](reports/期末研究報告.md) 整理主要效能、原前處理、文獻與重現入口。PDF 位於本機 Git 排除的 `artifacts/course_submission/icu_mortality_report/`。

組員分享版包含完整前處理、模型參數、切分、指標與文獻比較限制。第 10 節整理缺失韌性、外部增強及確認入口。 [組員研究報告](實驗報告_組員分享版.md)

早期表格基準與序列資料的輸入定義見下列規格。 [P7 基準評估](reports/P7_C資料集_凍結模型評估.md) · [深度模型資料銜接](reports/深度學習銜接備忘錄.md) · [P33 實際序列模型規格](reports/P33_hourly_sequence_protocol.md)

<!-- C_EVALUATION_START -->
## 後續已完成 C 評估（2026-10-13）

[P7：C資料集凍結模型評估](reports/P7_C資料集_凍結模型評估.md)：4,000筆、死亡585筆。Full AP 0.5798／AUROC 0.8749；Elastic AP 0.5713／AUROC 0.8743。使用原凍結模型與前處理，新增訓練0次，總訓練量維持158次。此後C已看過模型表現；後續方法仍應在A/B開發。

下方保留原P0–P6開發成果；「未評估C」的描述是該階段歷史狀態。
<!-- C_EVALUATION_END -->

以下是 2026-10-13 的 P0–P6 歷史結果。當時的模型分數來自 A/B 開發交叉驗證，後續序列與外部增補見上方的階段報告。

## 核心結果

- 開發資料8,000筆，死亡1,122筆（14.025%）。三組原始特徵數150／292／435。
- 選定 **XGBoost** 作後續表格基準。完整F2三折AP **0.5450 ± 0.0321**、AUROC **0.8656**。
- 三折精簡策略 **Elastic**：AP **0.5385 ± 0.0406**；相對完整平均差 **-0.0065**。此策略按開發結果選取，分數仍可能偏樂觀。
- 最後以全部A/B重新fit的完整／精簡模型，輸入分別 **435／322** 欄。這和三折平均欄數可能不同；P0–P6尚未評估C；後續結果見頂端P7連結。
- 對增補時間摘要與觀測模式的收益，LR與boosting不完全一致，支持繼續保留共同設定下的消融。

## 各階段報告

| 階段 | 報告 | 主要產出 |
|---|---|---|
| P0 | [資料與特徵重建](reports/P0_資料與特徵重建.md) | 原始序列、規則、字典、對齊與質量稽核 |
| P1 | [基準模型](reports/P1_基準模型.md) | Dummy與四種方法的三折性能／成本 |
| P2 | [有限調參](reports/P2_有限調參.md) | 18組候選配置，最高分與完整配置影響 |
| P3 | [特徵家族](reports/P3_特徵家族.md) | 基本摘要、變化、觀測模式的消融 |
| P4 | [特徵篩選](reports/P4_特徵篩選.md) | 品質／Elastic Net／Top10／Top20比較 |
| P5 | [變數關係與解釋](reports/P5_變數關係與解釋.md) | 相關、有效n、分組置亂、LR與SHAP |
| P6 | [穩定性與敏感性](reports/P6_穩定性補充.md) | 同一開發集五折、成人限定三折 |

## 五折及成人敏感性

| strategy | nfold | adult | AP | AP_SD | AUROC | n |
|---|---|---|---|---|---|---|
| Elastic | 3 | True | 0.5369 | 0.0304 | 0.8613 | 316.3333 |
| Elastic | 5 | False | 0.5551 | 0.0392 | 0.8692 | 319.0000 |
| Full | 5 | False | 0.5515 | 0.0383 | 0.8678 | 435.0000 |

五折／成人限定屬於開發集穩定性補充。成人分析重新切分，不能與原三折做嚴格成對推論；沒有作患者身分層級隔離。不能用平均±SD宣稱95%信賴區間或臨床有效性。

## 哪些變數值得後續研究

下面是完整boosting模型在驗證折的group permutation依賴排名；數值是AP下降，並非死亡風險增加幅度，也不是因果重要性。

| variable | clinical_name_zh | mean_AP_drop | fold_SD |
|---|---|---|---|
| GCS | 昏迷指數 | 0.1516 | 0.0200 |
| static_Age | 年齡 | 0.0369 | 0.0106 |
| Urine | 每次記錄尿量 | 0.0275 | 0.0060 |
| BUN | 血中尿素氮 | 0.0213 | 0.0119 |
| Lactate | 乳酸 | 0.0138 | 0.0053 |
| Bilirubin | 膽紅素 | 0.0134 | 0.0078 |
| Temp | 體溫 | 0.0115 | 0.0030 |
| HR | 心率 | 0.0099 | 0.0022 |
| pH | 動脈pH | 0.0096 | 0.0029 |
| Weight | 體重時間序列 | 0.0093 | 0.0062 |

高相關且可替代的變數，單項置亂可能低估依賴；已保存相關簇聯合置亂結果。另有數值摘要與missing/count的不同意義，需一起對照LR係數與原始分布。

![重要性](figures/group_importance.png)

## 銜接深度學習

- [候選通道](artifacts/candidate_variables_for_dl.csv) 與 [候選欄位](artifacts/selected_features.csv) 是訓練內selector的跨折穩定性記錄，**不能直接視為深度模型的最佳輸入**。
- 原始序列 [observations_AB.parquet](artifacts/observations_AB.parquet) 仍包含所有原始變數、值、時間與清理旗標。Weight admission與動態序列有重疊，資料字典已揭露。
- 後續應比較深度模型完整／精簡通道，保留原始時間、mask與可得靜態資訊，不把手工summary等同於時間序列。
- 已保存共同RecordID切分與完整／精簡表格pipelines。深度模型也只在A/B選方案，最後凍結後再共同評估C。
- 時間／缺失／變數關係的探索只能提供設計假說，不證明attention／graph一定優於本輪表格模型。

## 稽核與實際成本

- 總計 **158** 次fit，其中正式修正版 **120** 次，初版欄名除錯 **38** 次不納入結果。
- 正式版本fit耗時合計 **319.0秒**；這不包含特徵提取、相關性、SHAP／置亂等成本。實際牆鐘時間不可用此合計代替。
- 最高採樣RSS **458.3MB**；所有正式版本fit收斂警告 **0** 個。
- 初版官方縮寫TropI/TropT未對到原始TroponinI/TroponinT，已修正重建並重跑；[除錯紀錄](artifacts/superseded_debug_fits.json) 保留，沒有混入最終分數。
- [驗證結果](artifacts/study_validation.json)：唯一鍵、標籤對齊、OOF全覆蓋及AP重算、肌鈣蛋白欄位、模型保存讀回與C原壓縮檔雜湊均已檢查。

## 限制與下一步

未做外部驗證、統計顯著或臨床因果宣稱。保留非負極端值與部分生理零值，這些仍可能包含紀錄問題；資料清理與先前EDA不同。C先前已被EDA探索，後續須揭露此限制。

建議先審查候選清單及資料質量，再決定深度方法與外部增補來源。本輪已提供實際表格基準，可用於後續共同切分比較。


## 重現與審閱

[深度學習銜接備忘錄](reports/深度學習銜接備忘錄.md)說明序列、mask、時間間隔與共同切分。精簡模型最終322欄，**仍包含全部37個動態變數**；候選表已補上選擇依據、跨折排名範圍、缺失率與保留理由。

專案專屬環境位於`.venv`，套件固定於[requirements-lock.txt](requirements-lock.txt)。以下命令從專題根目錄執行，已完成配置會使用現有快取，不必重新訓練。重建輸入可能使資料雜湊改變；若要完整重跑應先另存目前成果，保留訓練預算與來源紀錄。

```powershell
ML/.venv/Scripts/python.exe ML/src/prepare.py
ML/.venv/Scripts/python.exe ML/src/study.py baseline
ML/.venv/Scripts/python.exe ML/src/study.py tune
ML/.venv/Scripts/python.exe ML/src/study.py families
ML/.venv/Scripts/python.exe ML/src/study.py selection
ML/.venv/Scripts/python.exe ML/src/explain.py
ML/.venv/Scripts/python.exe ML/src/study.py confirm
ML/.venv/Scripts/python.exe ML/src/finalize.py
ML/.venv/Scripts/python.exe ML/src/complete_reports.py
```

訓練量已按本次追加核准設為最多170次；目前158次。調參、selector與最終模型保存皆計入。最高RSS數字僅涵蓋訓練採樣，不代表SHAP／序列讀取的整個流程峰值。


## Windows 本機設定

本機以 `DSP` 為專題根目錄，`ML` 和 `data` 位於同一層。建立新環境時，從 `DSP` 的 PowerShell 執行：

```powershell
uv venv .\ML\.venv --python 3.12
uv pip sync --python .\ML\.venv\Scripts\python.exe .\ML\requirements-lock.txt
uv pip check --python .\ML\.venv\Scripts\python.exe
```

PhysioNet 2012 的本機資料路徑如下。A/B 的個案文字檔直接放在各自的目錄內。

```text
data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/
├── set-a/*.txt
├── set-b/*.txt
├── Outcomes-a.txt
├── Outcomes-b.txt
├── set-c.tar.gz
└── Outcomes-c.txt
```

`prepare.py` 以 A/B 重建開發特徵，並讀取 C 壓縮檔以記錄筆數與 SHA-256。匯入的 `ML/artifacts/` 已含 A/B 特徵與執行快取，可先從 `study.py baseline` 驗證既有流程。從原始資料重建前，請在獨立副本保留原本的 `ML/artifacts/`、報告和圖表，因為程式會寫回這些位置。

重跑已凍結模型的 C 評估時，需備妥上述所有檔案及 `ML/artifacts/final_models/`。C 壓縮檔需符合 `preparation_manifest.json` 中的原始 SHA-256，程式也會先驗證 A/B 特徵。

```powershell
.\ML\.venv\Scripts\python.exe .\ML\src\evaluate_c.py
```

原始資料與逐筆預測、快取、模型等 `ML/artifacts/` 產物保留在本機。從 Git 取回程式後，如需查閱已保存的實驗產物，可從原 Drive 專案補入 `ML/artifacts/`。
