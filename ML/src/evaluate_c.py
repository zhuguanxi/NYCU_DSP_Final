"""Evaluate frozen AB pipelines on C. No estimator fit or threshold selection."""
import os
from pathlib import Path
import json, time, tarfile, collections
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.mplcache'))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, auc, confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_curve)
from sklearn.calibration import calibration_curve
from prepare import DATA, ROOT, OUT, REPORT, extract_record, sha, UNITS

DEST = OUT / 'C_evaluation'
FIG = ROOT / 'ML/figures'
BOOTSTRAP_N = 2000
SEED = 20261013


def md(frame):
    z = frame.copy()
    for name in z:
        if pd.api.types.is_float_dtype(z[name]):
            z[name] = z[name].map(lambda v: f'{v:.4f}')
    return '| ' + ' | '.join(z.columns) + ' |\n|' + '|'.join(['---'] * len(z.columns)) + '|\n' + '\n'.join(
        '| ' + ' | '.join(map(str, row)) + ' |' for row in z.itertuples(index=False, name=None))


def ranking_metrics(y, score):
    precision, recall, _ = precision_recall_curve(y, score)
    return dict(AP=float(average_precision_score(y, score)),
        AUROC=float(roc_auc_score(y, score)), AUPRC_trapezoid=float(auc(recall, precision)),
        Brier=float(brier_score_loss(y, score)))


def all_metrics(y, score):
    p = score >= .5
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(**ranking_metrics(y, score), threshold=.5,
        Precision=float(precision_score(y, p, zero_division=0)),
        Recall=float(recall_score(y, p, zero_division=0)),
        Specificity=float(tn / (tn + fp)), F1=float(f1_score(y, p, zero_division=0)),
        Accuracy=float(accuracy_score(y, p)), Balanced_accuracy=float(balanced_accuracy_score(y, p)),
        MCC=float(matthews_corrcoef(y, p)), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def verify_extractor(ab):
    # All 8,000 rows must match existing development inputs before exposing C scores.
    a = pd.read_csv(OUT / 'record_alignment.csv')
    for k, row in enumerate(a.itertuples(index=False), 1):
        path = DATA / ('set-' + row.source_set.lower()) / f'{row.RecordID}.txt'
        feat, _, _ = extract_record(path.read_bytes(), row.RecordID)
        rebuilt = np.asarray([feat[c] for c in ab.columns], dtype=float)
        assert np.allclose(rebuilt, ab.loc[row.RecordID].to_numpy(dtype=float),
                           rtol=1e-12, atol=1e-12, equal_nan=True), row.RecordID
        if k % 2000 == 0:
            print(f'Extractor equivalence verified: {k}/8000 AB records', flush=True)


def bootstrap(y, predictions):
    rng = np.random.default_rng(SEED)
    rows = []
    for repeat in range(BOOTSTRAP_N):
        ids = rng.integers(0, len(y), size=len(y))
        yy = y[ids]
        assert np.unique(yy).size == 2
        for name, score in predictions.items():
            rows.append(dict(repeat=repeat, model=name, **ranking_metrics(yy, score[ids])))
        if (repeat + 1) % 500 == 0:
            print(f'Fixed-model paired bootstrap: {repeat + 1}/{BOOTSTRAP_N}', flush=True)
    samples = pd.DataFrame(rows)
    samples.to_csv(DEST / 'bootstrap_samples.csv', index=False)
    summary = []
    for name, score in predictions.items():
        point = ranking_metrics(y, score)
        for metric, value in point.items():
            v = samples.loc[samples.model == name, metric]
            summary.append(dict(model=name, metric=metric, estimate=value,
                CI95_low=float(v.quantile(.025)), CI95_high=float(v.quantile(.975))))
    summary = pd.DataFrame(summary)
    summary.to_csv(DEST / 'bootstrap_CI.csv', index=False)
    paired = []
    for metric in ranking_metrics(y, predictions['Full']):
        p = samples.pivot(index='repeat', columns='model', values=metric)
        delta = p.Elastic - p.Full
        paired.append(dict(metric=metric,
            Elastic_minus_Full=ranking_metrics(y, predictions['Elastic'])[metric] - ranking_metrics(y, predictions['Full'])[metric],
            CI95_low=float(delta.quantile(.025)), CI95_high=float(delta.quantile(.975))))
    paired = pd.DataFrame(paired)
    paired.to_csv(DEST / 'paired_difference_CI.csv', index=False)
    return summary, paired


def plot(y, predictions):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    calibration = []
    for name, score in predictions.items():
        fpr, tpr, _ = roc_curve(y, score)
        pr, re, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, label=f'{name}: AUROC={roc_auc_score(y, score):.4f}')
        axes[1].step(re, pr, where='post', label=f'{name}: AP={average_precision_score(y, score):.4f}')
        observed, predicted = calibration_curve(y, score, n_bins=10, strategy='quantile')
        axes[2].plot(predicted, observed, 'o-', label=name)
        calibration.extend(dict(model=name, mean_predicted=float(a), observed_fraction=float(b)) for a, b in zip(predicted, observed))
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].axhline(y.mean(), linestyle='--', color='gray', label=f'Prevalence={y.mean():.4f}')
    axes[2].plot([0, 1], [0, 1], '--', color='gray')
    for ax, title, xlabel, ylabel in zip(axes,
        ['C evaluation: ROC', 'C evaluation: PR', 'C evaluation: calibration'],
        ['False positive rate', 'Recall', 'Mean predicted probability'],
        ['Recall', 'Precision', 'Observed mortality fraction']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(FIG / 'C_evaluation_curves.png', dpi=170); plt.close(fig)
    pd.DataFrame(calibration).to_csv(DEST / 'calibration_bins.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (name, score) in zip(axes, predictions.items()):
        cm = confusion_matrix(y, score >= .5, labels=[0, 1])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Alive', 'Death'],
            yticklabels=['Alive', 'Death'], xlabel='Predicted', ylabel='Observed',
            title=f'{name}: threshold 0.5')
    fig.tight_layout(); fig.savefig(FIG / 'C_confusion_matrices.png', dpi=170); plt.close(fig)


def main():
    DEST.mkdir(exist_ok=True)
    archive = DATA / 'set-c.tar.gz'
    ab = pd.read_parquet(OUT / 'features_AB.parquet')
    original_sha = sha(archive)
    assert original_sha == json.loads((OUT / 'preparation_manifest.json').read_text())['C_sha256']
    ledger_sha = sha(OUT / 'fit_ledger.jsonl')
    metadata = json.loads((OUT / 'final_models/metadata.json').read_text())
    frozen = {name: sha(OUT / 'final_models' / f'{name}.joblib') for name in ['Full', 'Elastic']}
    locked = dict(date='2026-10-13', training_set='A+B, 8000 encounters', threshold=.5,
        evaluated_models=['Full', 'Elastic'], decision_metric='AP', preprocessing='unchanged AB rules and fitted pipelines',
        model_sha256=frozen, AB_feature_sha256=sha(OUT / 'features_AB.parquet'),
        C_archive_sha256=original_sha, bootstrap_repeats=BOOTSTRAP_N, bootstrap_seed=SEED)
    # Written before reading any C label or producing a C prediction.
    (DEST / 'evaluation_protocol.json').write_text(json.dumps(locked, indent=2))
    verify_extractor(ab)
    features = []; alignment = []; flags = []; quality = collections.Counter()
    with tarfile.open(archive, 'r:gz') as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.endswith('.txt')), key=lambda m: m.name)
        assert len(members) == 4000
        for k, member in enumerate(members, 1):
            rid = int(Path(member.name).stem)
            content = tar.extractfile(member).read()
            feat, records, record_flags = extract_record(content, rid)
            features.append(dict(RecordID=rid, **feat)); flags.append(dict(RecordID=rid, **record_flags))
            from hashlib import sha256
            alignment.append(dict(RecordID=rid, archive_member=member.name, sha256=sha256(content).hexdigest()))
            quality.update((r[2], r[5]) for r in records)
            if k % 1000 == 0: print(f'C features reconstructed: {k}/4000', flush=True)
    x = pd.DataFrame(features).set_index('RecordID').loc[:, ab.columns]
    assert x.index.is_unique and set(x.index).isdisjoint(ab.index)
    labels = pd.read_csv(DATA / 'Outcomes-c.txt').set_index('RecordID')
    assert labels.index.is_unique and set(labels.index) == set(x.index)
    y = labels.loc[x.index, 'In-hospital_death'].astype(int).to_numpy()
    assert set(y) == {0, 1}
    x.to_parquet(DEST / 'features_C.parquet')
    pd.DataFrame(dict(RecordID=x.index, In_hospital_death=y)).to_csv(DEST / 'labels_C.csv', index=False)
    pd.DataFrame(alignment).to_csv(DEST / 'record_alignment_C.csv', index=False)
    flags = pd.DataFrame(flags); flags.to_csv(DEST / 'cohort_quality_flags_C.csv', index=False)
    pd.DataFrame([dict(variable=a, quality_flag=b, n=n) for (a, b), n in quality.items()]).to_csv(DEST / 'measurement_quality_C.csv', index=False)
    predictions = {}; results = []
    for name in ['Full', 'Elastic']:
        pipe, info, cols = joblib.load(OUT / 'final_models' / f'{name}.joblib')
        assert info['feature_data_sha256'] == locked['AB_feature_sha256']
        assert info['train_n'] == 8000 and cols == metadata[name]['columns']
        assert np.array_equal(pipe['model'].classes_, [0, 1])
        start = time.perf_counter(); score = pipe.predict_proba(x[cols])[:, 1]; elapsed = time.perf_counter() - start
        assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
        predictions[name] = score
        results.append(dict(model=name, input_n=len(cols), encoded_n=info['encoded_n'], predict_s=elapsed, **all_metrics(y, score)))
        print(f'{name} C: {ranking_metrics(y, score)}', flush=True)
    result = pd.DataFrame(results)
    # Constant mortality probability learned from A+B, never from C labels.
    ab_y = pd.read_csv(OUT / 'labels_AB.csv').In_hospital_death
    constant = np.full(len(y), ab_y.mean())
    pd.DataFrame([dict(model='AB_prior', probability=ab_y.mean(), **all_metrics(y, constant))]).to_csv(DEST / 'constant_baseline_C.csv', index=False)
    result.to_csv(DEST / 'metrics_C.csv', index=False)
    pred = pd.DataFrame(dict(RecordID=x.index, y=y, score_Full=predictions['Full'], score_Elastic=predictions['Elastic']))
    pred.to_csv(DEST / 'predictions_C.csv', index=False)
    ci, paired = bootstrap(y, predictions)
    plot(y, predictions)
    # Adult subset is descriptive only; do not refit or pick a new model.
    adult = x.static_Age.ge(18).to_numpy()
    adult_result = pd.DataFrame([dict(model=name, n=int(adult.sum()), deaths=int(y[adult].sum()),
        **all_metrics(y[adult], score[adult])) for name, score in predictions.items()])
    adult_result.to_csv(DEST / 'adult_sensitivity_C.csv', index=False)
    comparison = []
    dev = pd.read_csv(OUT / 'selection_folds.csv')
    dev5 = pd.read_csv(OUT / 'confirmation_folds.csv')
    for name in predictions:
        three = dev[(dev.model == 'XGBoost') & (dev.strategy == name)]
        five = dev5[(dev5.strategy == name) & (dev5.nfold == 5) & (~dev5.adult)]
        point = result[result.model == name].iloc[0]
        comparison.append(dict(model=name, AB_3fold_AP=three.AP.mean(), AB_5fold_AP=five.AP.mean(),
            C_AP=point.AP, C_minus_AB3_AP=point.AP-three.AP.mean(),
            AB_3fold_AUROC=three.AUROC.mean(), C_AUROC=point.AUROC))
    comparison = pd.DataFrame(comparison); comparison.to_csv(DEST / 'development_vs_C.csv', index=False)
    assert sha(archive) == original_sha and sha(OUT / 'fit_ledger.jsonl') == ledger_sha
    assert all(sha(OUT / 'final_models' / f'{name}.joblib') == value for name, value in frozen.items())
    loaded = pd.read_csv(DEST / 'predictions_C.csv')
    for name in predictions:
        assert np.isclose(average_precision_score(loaded.y, loaded[f'score_{name}']), result.loc[result.model == name, 'AP'].iloc[0])
    manifest = dict(**locked, n=len(x), deaths=int(y.sum()), mortality_rate=float(y.mean()),
        age_under18=int(flags.Age_under18.sum()), no_dynamic=int(flags.no_dynamic_observations.sum()),
        labels_sha256=sha(DATA / 'Outcomes-c.txt'), C_feature_sha256=sha(DEST / 'features_C.parquet'),
        model_fit_count_added=0, original_models_unchanged=True, original_C_archive_unchanged=True,
        all_8000_AB_extraction_equivalence_passed=True, C_AB_RecordID_overlap=0,
        prediction_roundtrip_metrics_passed=True, bootstrap='ordinary paired encounter-level percentile bootstrap; fixed models')
    (DEST / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2))
    write_report(result, ci, paired, comparison, adult_result, manifest, float(ab_y.mean()))
    print('C evaluation completed; no additional training', flush=True)


def write_report(result, ci, paired, comparison, adult_result, manifest, prior):
    full = result.set_index('model').loc['Full']; reduced = result.set_index('model').loc['Elastic']
    delta = reduced.AP - full.AP
    ap_ci = paired[paired.metric == 'AP'].iloc[0]
    interpretation = ('差值的95%區間包含0，這次樣本不足以支持兩者有明確的AP優劣。' if ap_ci.CI95_low <= 0 <= ap_ci.CI95_high
        else '差值的95%區間未包含0；這是固定兩個模型在此C樣本上的差異，不代表所有重新訓練或外部資料都會如此。')
    report = f'''# C 資料集：凍結模型的表現評估報告

日期：2026-10-13。使用A+B全部8,000筆訓練後保存的兩個XGBoost模型，直接預測C。**新增模型訓練0次，沒有用C調參、重選特徵、重新補值／標準化或挑閾值。**

## 評估範圍與資料

- C共 **{manifest['n']:,}筆**，死亡 **{manifest['deaths']}筆（{100*manifest['mortality_rate']:.3f}%）**，存活{manifest['n']-manifest['deaths']}筆。
- 使用0–48小時（含端點）觀測預測本次住院死亡；不是入ICU當下的預測。
- 年齡低於18有{manifest['age_under18']}筆，沒有有效動態觀測有{manifest['no_dynamic']}筆；主評估照原先規則保留。
- 完整Full輸入435欄，精簡Elastic輸入322欄；編碼後分別438／325欄。兩者仍包含全部37個動態臨床變數。
- C與A/B的RecordID零重疊；缺少跨住院病人識別，不聲稱病人身分層級隔離。
- 從原壓縮檔直接讀取，原始資料與凍結模型檔保持不變。共用的特徵提取已在全部8,000筆A/B與原保存矩陣逐欄核對一致，避免C採不同處理。

## 主要指標

{md(result[['model','input_n','AP','AUROC','AUPRC_trapezoid','Brier','predict_s']])}

AP是主要指標，使用average_precision_score；AUPRC_trapezoid是PR曲線梯形積分，兩者數字不同且不得混稱。AP／AUROC越高越好，Brier越低越好。predict_s是一次4,000筆pipeline推論的實測，不含特徵建構與圖表，非穩定硬體benchmark。

固定A/B死亡率{prior:.5f}作常數預測的基準，其C AP等於C死亡率{manifest['mortality_rate']:.5f}，AUROC=0.5；機率值沒有利用C死亡率擬合。詳見[常數基準](../artifacts/C_evaluation/constant_baseline_C.csv)。

### 固定模型的95% bootstrap區間

{md(ci)}

以C的4,000次encounter為單位，有放回抽樣{BOOTSTRAP_N}次，seed={SEED}，採2.5／97.5百分位。兩個模型共用每次抽樣，未重新訓練。區間反映此測試樣本的抽樣不確定性，不涵蓋訓練隨機性、跨住院相依或換醫院的變異；AP亦受死亡比例影響。

## 完整與精簡的成對比較

{md(paired)}

差值方向都是Elastic−Full；AP／AUROC／AUPRC正值較好，Brier負值較好。精簡AP差 **{delta:+.4f}**。{interpretation}

原訂0.01 AP損失是工程容忍值，不是顯著性門檻。即使點估計符合容忍值，也不能據此保證精簡在所有資料上不劣；以差值區間一起判斷。本次C結果只用來評估已凍結方案，不據此修改原A/B選定的模型。

本次AP差的95%區間為 **[{ap_ci.CI95_low:+.4f}, {ap_ci.CI95_high:+.4f}]**：點估計損失小於0.01，但區間下界低於−0.01，尚不能確立「損失不超過0.01」。

## 固定閾值0.5的分類行為

{md(result[['model','Precision','Recall','Specificity','F1','Accuracy','Balanced_accuracy','MCC']])}

{md(result[['model','TN','FP','FN','TP']])}

Full在{manifest['deaths']}個死亡病例中的漏判數為{int(full.FN)}，Elastic為{int(reduced.FN)}。固定閾值結果描述目前決策行為，不代表0.5是適合任務的最佳門檻。Accuracy易受多數存活者影響，需連同死亡Recall與Precision看。若後續要挑閾值，回到A/B訓練內驗證，不能在C挑完再將同一份C當新測試。

完整模型Precision為 **{100*full.Precision:.2f}%**、死亡Recall為 **{100*full.Recall:.2f}%**。排序分數有辨識能力，不表示目前的固定門檻已能找出大多數死亡病例。Accuracy **{100*full.Accuracy:.2f}%**亦不能作為唯一成功判準。

![混淆矩陣](../figures/C_confusion_matrices.png)

## 曲線與機率校準

![ROC、PR與校準](../figures/C_evaluation_curves.png)

校準圖以10個等樣本數分箱描述平均預測機率與實際死亡比例，未重新校準。PR虛線是C死亡比例。單一校準曲線沒有提供分箱信賴區間，不能宣稱完成臨床校準驗證。

## 與先前A/B開發成績對照

{md(comparison)}

A/B三／五折是較小訓練子集與多折平均，C是用全部A/B重新訓練後的單一模型測試。兩者訓練量、分布與統計方式不同，差值不是嚴格成對泛化效應，也不能單靠C較高或较低判定資料洩漏／過擬合。若要與Warpformer等文獻比較，仍需核對相同資料、切分、清理與AP/AUPRC定義；本報告不是SOTA排名。

## 成人限定的描述性補充

{md(adult_result[['model','n','deaths','AP','AUROC','Brier','Recall']])}

這是同一個凍結A/B模型在C的Age≥18子集評估，沒有訓練成人專用模型，不與先前成人三折當成同一protocol。

## 結論與後續使用

1. 取得可追溯的C結果：Full AP **{full.AP:.4f}**／AUROC **{full.AUROC:.4f}**；Elastic AP **{reduced.AP:.4f}**／AUROC **{reduced.AUROC:.4f}**。
2. 精簡摘要的效能與成本須一起看，仍未證明可以減少原始序列通道；兩個模型保持全部37個動態變數。
3. C已正式產生模型分數。未來深度模型仍只在A/B調方法，凍結後可用同一C共同對照；但本次分數及先前EDA已被看過，C不能再稱為完全未接觸的測試集。若依C結果改方案，該改動屬新開發，需另設獨立驗證或揭露反覆測試限制。

## 可重現產出

- [指標表](../artifacts/C_evaluation/metrics_C.csv)、[逐筆預測](../artifacts/C_evaluation/predictions_C.csv)。
- [評估規格](../artifacts/C_evaluation/evaluation_protocol.json)、[來源／凍結驗證](../artifacts/C_evaluation/evaluation_manifest.json)。
- [成對區間](../artifacts/C_evaluation/paired_difference_CI.csv)、[bootstrap明細](../artifacts/C_evaluation/bootstrap_samples.csv)。
- [C特徵](../artifacts/C_evaluation/features_C.parquet)、[品質旗標](../artifacts/C_evaluation/cohort_quality_flags_C.csv)。

從專題根目錄執行：`ML/.venv/Scripts/python.exe ML/src/evaluate_c.py`。此程式只提取、預測與評估，不调用fit；再次執行不增加模型訓練量。
'''
    (REPORT / 'P7_C資料集_凍結模型評估.md').write_text(report.replace('较低', '較低').replace('调用', '呼叫'), encoding='utf-8')
    index = ROOT / 'ML/README.md'
    text = index.read_text(encoding='utf-8')
    # Preserve development results as history, while making C status unambiguous.
    title, body = text.split('\n', 1)
    if '\n<!-- C_EVALUATION_START -->' in body:
        body = body.split('\n<!-- C_EVALUATION_START -->')[0] + body.split('<!-- C_EVALUATION_END -->\n', 1)[1]
    body = body.replace('全部模型分數來自A/B開發交叉驗證，C未作模型評估',
        'P0–P6的模型分數來自A/B開發交叉驗證，該階段未評估C')
    body = body.replace('最終C表現尚未知', 'P0–P6尚未評估C；後續結果見頂端P7連結')
    notice = f'''\n<!-- C_EVALUATION_START -->
## 後續已完成 C 評估（2026-10-13）

[P7：C資料集凍結模型評估](reports/P7_C資料集_凍結模型評估.md)：{manifest['n']:,}筆、死亡{manifest['deaths']}筆。Full AP {full.AP:.4f}／AUROC {full.AUROC:.4f}；Elastic AP {reduced.AP:.4f}／AUROC {reduced.AUROC:.4f}。使用原凍結模型與前處理，新增訓練0次，總訓練量維持158次。此後C已看過模型表現；後續方法仍應在A/B開發。

下方保留原P0–P6開發成果；「未評估C」的描述是該階段歷史狀態。
<!-- C_EVALUATION_END -->
'''
    index.write_text(title + '\n' + notice + body, encoding='utf-8')


if __name__ == '__main__':
    main()
