"""Build the experimental registry, scientific plots and evidence-led research story."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ML=Path(__file__).resolve().parents[1]
REPORT=ML/'reports'
OUT=ML/'artifacts'
FIG=ML/'figures'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def ci(x): return f'[{x[0]:+.5f}, {x[1]:+.5f}]'


def tables(p55,p54,p57,p59):
    rows=[]
    def add(cohort,hours,scenario,model,setting,n,deaths,metrics,source,alignment='urine_mapped'):
        rows.append(dict(cohort=cohort,hours=hours,scenario=scenario,model=model,setting=setting,
                         alignment=alignment,stays=n,deaths=deaths,
                         **{key:metrics[key] for key in ('ap','auroc','brier','sensitivity')},source=source))
    for r in p55['metrics']:
        add('P12 Set C',48,r['scenario'],r['model'],'P12-only frozen',4000,585,r,'P55',alignment='P12')
    for mode in ('hourly','native'):
        r=p54['results'][f'eicu_{mode}']
        for model,m in [('kedgn_ab',r['ensemble']),*r['controls'].items()]:
            if model=='set_a_kedgn': model='kedgn_set_a'
            add('eICU Demo',48,mode,model,'P12-only frozen',r['stays'],r['deaths'],m,'P54')
        r=p57['results'][mode]
        for model,m in r['metrics'].items():
            add('eICU Demo',24,mode,model,'P12 + Demo grouped OOF' if model=='p50' else 'P12-only frozen',1398,121,m,'P57')
        for model,m in r['matched']['metrics'].items():
            add('eICU Demo APACHE shared',24,mode,model,'published clinical score' if model=='apache_iva' else
                'P12 + Demo grouped OOF' if model=='p50' else 'P12-only frozen',1166,100,m,'P57')
    p52=read(OUT/'p52_24h_catboost_blend/evaluation.json')['set_c']
    for case,r in p52['views'].items():
        for model,m in r['metrics'].items():
            add('P12 Set C proxy',24,case,model,'P12-only frozen',p52['stays'],p52['deaths'],m,'P52',alignment='P12')
    for r in p59['set_c']:
        add('P12 Set C',48,r['scenario'],f'ma_{r["variant"]}_{r["model"]}','P12-only frozen',4000,585,r['metrics'],'P59',alignment='P12')
    for r in p59['eicu']:
        for key,alignment in [('before','original_mapping'),('after','urine_mapped')]:
            add('eICU Demo',48,r['mode'],f'ma_{r["variant"]}_{r["model"]}','P12-only frozen',782,91,r[key],'P59',alignment=alignment)
    target=REPORT/'研究結果總表.csv'
    with target.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return rows


def plots(p55,p58,p57):
    FIG.mkdir(exist_ok=True)
    font=Path('C:/Windows/Fonts/msjh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    cases=['clean','full_GCS','full_all_six']
    models=['frozen_full','gcs_only_full','p26','p33','kedgn_ab']
    names=['Full','GCS-only','P26','P33','KEDGN A+B']
    colors=['#77818b','#3b76af','#159589','#e67f32','#8665ad']
    points={(r['scenario'],r['model']):r['ap'] for r in p55['metrics']}
    fig,ax=plt.subplots(figsize=(10.2,5.6))
    x=np.arange(3)
    for i,(model,name,color) in enumerate(zip(models,names,colors)):
        values=[points[c,model] for c in cases]
        bars=ax.bar(x+(i-2)*.155,values,.148,label=name,color=color)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=8,padding=3)
    ax.axhline(585/4000,color='#a6abb0',ls=':',lw=1,label='死亡率參考')
    ax.set_xticks(x,['未遮蔽','GCS 全遮','六項全遮']);ax.set_ylabel('AP')
    ax.set_ylim(0,.7);ax.set_title('同一 Set C：完整資料與固定缺失情境')
    ax.legend(ncol=3,loc='upper center',frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.1,.025,'4,000 次住院，585 次死亡；KEDGN 為五種子機率平均。其餘單欄結果見 P55。',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1))
    fig.savefig(FIG/'研究總結_AP情境.png',dpi=180)
    fig.savefig(FIG/'研究總結_AP情境.svg')
    plt.close(fig)
    labels=[];values=[];intervals=[];palette=[]
    for case,label in zip(cases,['Set C 完整：P33−Full','Set C GCS 全遮：P33−Full','Set C 六項全遮：P33−Full']):
        r=p58['set_c']['scenarios'][case]
        labels.append(label);values.append(r['delta']['ap']);intervals.append(r['paired_ci']['ap']);palette.append('#e67f32')
    for mode,cn in [('hourly','整點'),('native','原始間隔')]:
        r=p58['eicu_demo']['modes'][mode]
        labels.append(f'Demo 48h {cn}：P33−Full');values.append(r['delta']['frozen_full']['ap'])
        intervals.append(r['paired_ci']['frozen_full']['ap']);palette.append('#3b76af')
    for mode,cn in [('hourly','整點'),('native','原始間隔')]:
        r=p57['results'][mode]
        labels.append(f'Demo 24h {cn}：P52−P49');values.append(r['metrics']['p52']['ap']-r['metrics']['p49']['ap'])
        intervals.append(r['patient_ci']['p52_vs_p49']['ap']);palette.append('#159589')
    fig,ax=plt.subplots(figsize=(10.2,5.6))
    for i,(value,bounds,color) in enumerate(zip(values,intervals,palette)):
        ax.plot(bounds,[i,i],color=color,lw=2)
        ax.plot(value,i,'o',color=color,ms=7)
    ax.axvline(0,color='#606870',ls='--',lw=1)
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
    ax.set_xlim(-.085,.105);ax.set_xlabel('AP 差（候選減對照）；線段為 95% 配對區間')
    ax.set_title('Set C 改善與 Demo 不確定性分開判讀')
    ax.spines[['top','right']].set_visible(False)
    fig.text(.06,.02,'Set C：2,000 次住院重抽；Demo：1,000 次患者重抽。24h 與 48h 為不同隊列。',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1))
    fig.savefig(FIG/'研究總結_配對差.png',dpi=180)
    fig.savefig(FIG/'研究總結_配對差.svg')
    plt.close(fig)


def story(p55,p57,p58,p59):
    metrics={(r['scenario'],r['model']):r for r in p55['metrics']}
    a=p58['set_c']['scenarios']
    lines=['# 研究主線與結果敘事','',
           '建議題目：**觀測資訊與缺失韌性的 ICU 住院死亡預測：從時間序列混合到跨資料庫驗證**。','',
           '## 一句話主張','',
           '我們研究的不是單純刪減特徵，而是：當 ICU 量測不完整、量測流程改變或資料庫不同時，如何兼顧死亡風險排序與缺失韌性。時間摘要、缺失專家與輕量序列混合在固定 P12 留出集有效；外部 AP 的增益仍不確定，因此將機率誤差、補值依賴與跨庫表現分開評估。','',
           '## 主角與研究問題','',
           '主模型以 **P33** 為主：435 欄表格及 140 個時間摘要 → 有／無 GCS 的分流模型 P26 → 與 GCS-only Elastic 作 log odds 混合 P28 → 與 20 通道逐小時 CNN 作 75%／25% 機率混合。原 Full 是基準，GCS-only 是缺失增強對照，P26 是專家消融，MA-GBT 是補值依賴方法對照，KEDGN 是實際重現並在共同資料上重訓的競爭基準。','',
           '這樣主線保持一個明確可執行模型；P26 的某些遮蔽點估計更高，留作取捨分析，不按測試情境事後替換主模型。P56 額外檢查了條件混合，A/B 沒有勝過原 P33，所以保留既有規則。','',
            '### 問題一：時間與缺失表示的組合能否改善完整資料排序？','',
           '原 A/B 8,000 次住院／1,122 次死亡使用固定三折開發，縮放、補值和參數只由訓練部分估計；Set C 4,000／585 用凍結模型計分。P33 未遮蔽 AP 0.60447，高於 Full 0.57985，ΔAP +0.02462，95% 區間 [+0.01109, +0.03775]。KEDGN 的原作者五種子測試已核對，再用 A+B 三折選 9 輪，全 A+B 五種子重訓，Set C AP 0.57502。','',
           '這支持「在本研究共同留出設定中，這套時間與缺失表示有效」。KEDGN 與本組仍有表示、前處理、平衡採樣和模型容量差異，不能將結果包裝成打敗所有 SOTA。','',
           '### 問題二：完整資料較高，缺失時是否也較穩？','',
           '| Set C 情境 | Full AP | P33 AP | ΔAP［95% 配對區間］ |','| --- | ---: | ---: | --- |']
    for case,label in [('clean','未遮蔽'),('full_GCS','GCS 全遮'),('full_all_six','六項全遮')]:
        r=a[case]
        lines.append(f'| {label} | {r["metrics"]["frozen_full"]["ap"]:.5f} | {r["metrics"]["p33"]["ap"]:.5f} | {r["delta"]["ap"]:+.5f} {ci(r["paired_ci"]["ap"])} |')
    lines+=['',
            '同一 P33 的三種情境都有正向配對區間。更重要的是，扣除各自未遮蔽分數後，GCS 全遮的 AP 損失較 Full 少 0.05368［+0.03663, +0.07163］，六項全遮少 0.03203［+0.01214, +0.05118］。因此韌性結論同時有遮蔽後分數及較小退化的支持。','',
            f'KEDGN A+B 的 GCS 全遮 AP 為 {metrics["full_GCS","kedgn_ab"]["ap"]:.5f}、六項全遮為 {metrics["full_all_six","kedgn_ab"]["ap"]:.5f}；完整八種情境與兩版 KEDGN 見 P55。這些是共同病例上的敏感度結果，不能解讀成 GCS 或其他變數的臨床因果效果。','',
            '在這兩種遮蔽中，P33 對 A+B KEDGN 的 ΔAP 分別為 +0.08728［+0.06193, +0.11394］及 +0.05199［+0.02505, +0.07889］。KEDGN 使用更多 A/B 病例後未遮蔽 AP 提高，但整欄缺失的 AP 退化沒有隨之消失；不規則序列建模能力與缺失韌性仍需要分別測試。','',
            '### 問題三：缺失韌性可以直接變成跨資料庫增益嗎？','',
            'eICU Demo 48 小時沿同一 782 次住院／91 次死亡，核對欄位、單位與住院死亡標籤。P33 對 Full 的整點 ΔAP +0.00498［−0.06114, +0.05663］，原始間隔 −0.00466［−0.06581, +0.04827］；兩者均未確認改善。P45 的真實 Demo 增強使用患者／院鍵區塊折外預測，對 P28 亦是點估計上升但區間跨零。','',
            '24 小時是另一個觀察窗與風險集，主要模型為 P52。P57 補齊它對 P50 外部增強的比較：P52 的 AP 點估計較高，患者與院鍵區塊區間都跨零。P52 對 P49 的 Brier 誤差下降則在兩種取樣及兩種重抽下都有支持。應說「本 Demo 的機率誤差改善」，不能替換成「外部 AP 已證實提升」。','',
            'MA-GBT 提供機制對照：尿量對齊後，一些原版模型的補值依賴大幅下降，但 AP 變化很小且區間跨零；五欄增強版的尿量對齊前後分數相同。這支持補值依賴與風險排序是不同研究目標，降低前者不保證提高後者。','',
            '## 主要實驗與補充實驗如何安排','',
            '1. 資料與前處理：兩個資料庫、住院死亡、觀測窗、單位、缺失旗標、訓練內估計統計與原始資料雜湊。',
            '2. 主要模型：Full → GCS-only → 時間與分流 → P33，讓讀者理解每個成分解決哪一個問題。',
            '3. 主結果：同一 P33 的 clean／GCS／六項全遮與損失差，使用 P58；以 P55 的競爭基準補完整性。',
            '4. 外部結果：P12-only 轉移與 Demo 折外增強分開；用 P57 展示 AP 與 Brier 的結論差異。',
            '5. 補充分析：MA-GBT α／增強／補值依賴、單欄遮蔽、P56 無增益、24h APACHE 同窗比較、時間與重複紀錄稽核。','',
            '![完整與缺失情境 AP](../figures/研究總結_AP情境.png)','',
            '![配對 AP 差及區間](../figures/研究總結_配對差.png)','',
            '## 可以主張的貢獻','',
            '- 建立可重跑、同病例、同遮蔽與來源可核對的 ICU 死亡分類比較，保留資料與選模邊界。',
            '- 同一固定模型在 P12 未遮蔽與指定缺失情境同時改善，且缺失退化較小。',
            '- 實際重現 MA-GBT 與 KEDGN，並把補值依賴、排序及機率誤差分開量化。',
            '- 透過跨庫、映射修正與分組外部增強，指出域內韌性不能直接當作外部效能提升的證據。','',
            '## 必須主動交代的限制','',
            'Set C 與 Demo 已反覆用於比較；雖未將 Set C 放入擬合，仍不能稱為研究全程未查看的新測試。配對 bootstrap 只反映固定模型的抽樣差異，未涵蓋全部重新訓練與歷史模型搜尋變異。Demo 樣本與死亡事件有限；未取得完整版，故不主張新版獨立 eICU 驗證。','',
            'P12 的觀測時鐘與結果整天編碼有既有不一致；P41 保守時間子集仍支持主要改善，但 24 小時隊列仍屬風險集代理。48h 與 24h 的病例、死亡率和輸入不同，不能只比較 AP 大小。發布說明的醫院數與實際院鍵數不同；檔案已符合官方 SHA-256，但實際醫院數不由院鍵數推斷。','',
            '缺失是人工整欄介入，未完整重現臨床決策與介面缺失。eICU 的未標籤觀測分布與資料字典曾用於對齊／缺失設計，故稱 P12-only 訓練轉移，不稱完全未接觸目標資料的盲測。P45/P50 使用部分 Demo 標籤訓練，必須稱分組折外適應估計。','',
            '## 結論用語','',
            '「時間與缺失資訊的組合，在固定 P12 留出隊列同時改善死亡排序及指定缺失韌性；但跨資料庫 AP 增益仍未確定。模型對補值的依賴、排序品質與機率誤差需要分別驗證，域內改善不能取代外部確認。」','',
            '不把負向消融、未知外部效果或未完成的其他論文重跑藏起來。Warpformer、ViTST、WaveGNN 等可作方法背景，完整同隊列復現者只按實際完成狀態列出。','',
            '## 對照資料','',
            '[完成驗收](研究完成驗收.json) · [結果總表](研究結果總表.csv) · [P55](P55_kedgn_missingness_results.md) · [P56](P56_observation_mixture_results.md) · [P57](P57_demo_paired_results.md) · [P58](P58_joint_objectives_results.md) · [P59](P59_ma_alignment_results.md) · [文獻定位](文獻定位與可比性.md)','']
    (REPORT/'研究結論與故事.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    verified=read(REPORT/'研究完成驗收.json')
    if not verified['experiments_complete']: raise ValueError('Experimental completion not verified')
    p55=read(REPORT/'P55_kedgn_missingness_summary.json')
    p54=read(REPORT/'P54_kedgn_ab_transfer_summary.json')
    p57=read(REPORT/'P57_demo_paired_summary.json')
    p58=read(REPORT/'P58_joint_objectives_summary.json')
    p59=read(REPORT/'P59_ma_alignment_summary.json')
    rows=tables(p55,p54,p57,p59)
    plots(p55,p58,p57)
    story(p55,p57,p58,p59)
    registry=dict(scope=verified['scope'],experiments=verified['evidence'],result_rows=len(rows),
                  primary_model='P33',robustness_specialist='P26',
                  external_ap_improvement_confirmed=False,
                  evidence='研究完成驗收.json',results='研究結果總表.csv',story='研究結論與故事.md')
    (REPORT/'研究實驗登錄.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Published {len(rows)} aggregate metric rows, scientific figures, registry and research story.',flush=True)


if __name__=='__main__': main()
