# 完整研究報告的重建

這裡只編排已公開的整體結果，不會訓練模型、重抽統計或接觸原始患者資料。依目前發布安排，完整報告 Markdown／PDF、P51 歷史快照、`final_report_content.md` 及來源／編排 JSON 暫只保留本機，已由 `.gitignore` 排除。GitHub 保留編排程式與圖表；單獨 clone 不含報告草稿範本。

從 `ML` 執行。圖表只需公開彙總；`compose`／`pdf`／`check` 需本機已有內容範本與報告產物：

```powershell
# 圖：使用既有專案環境的 Matplotlib 3.11.2
& .venv/Scripts/python.exe reporting/plot_final_report.py

# 文字、表格、PDF：使用含 reportlab / pypdf / pillow 的 Python
python reporting/build_final_report.py compose
python reporting/build_final_report.py pdf
python reporting/build_final_report.py check
```

本機已使用 Codex 隨附的 ReportLab 4.4.9、pypdf 6.21.4、Pillow 13.0.0 完成 PDF。一般 Python 可依 [報告套件](../requirements-report.txt) 安裝；圖表使用主研究環境已有的 Matplotlib，不必把它混入舊模型環境。

PDF 預設嵌入 Windows 的 Microsoft JhengHei 字型 `msjh.ttc`／`msjhbd.ttc`。其他系統可用 `--font` 與 `--bold-font` 指定合法可用、涵蓋繁體中文的 TrueType 字型。字型不提交 Git。若手動修改最終 Markdown，只執行 `pdf` 即可重新排版；執行 `compose` 會依內容範本與公開彙總重建 Markdown，因此應把持續保留的文字修改寫回 `final_report_content.md`。

`compose` 首次會保留原 P51 歷史版本，後續不覆寫它。`check` 核對來源雜湊、PDF／Markdown 身分、重要數值與中文文字、相對連結、公開結果列數，以及資料目錄未被 Git 追蹤。視覺檢查另以 Poppler 將 PDF 轉 PNG，再檢查所有頁的字型、表格、換頁與圖表。來源與輸出紀錄在本機 `reports/final_report_sources.json`、`reports/final_report_build.json`，目前暫不公開。

此流程不改 P60/P61 的模型、種子、測試配置或既有 JSON。報告中的五次指標平均、五種子機率集成與原單一凍結模型必須維持分開，並保留部分缺失退化、外部 AP 與模型取捨的限制。
