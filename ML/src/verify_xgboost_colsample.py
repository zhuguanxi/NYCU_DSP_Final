"""Independent integrity and metric audit of the fixed P65 feature-sampling study."""
import csv,json,statistics,subprocess
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,recall_score
from train_xgboost_colsample import ML,ROOT,P63,SEEDS,MODELS,R1_PARAMS,read,sha,write,canonical,verify_sources,validate_frozen

METRICS=('ap','auroc','brier','sensitivity')
def score(y,p):return dict(ap=average_precision_score(y,p),auroc=roc_auc_score(y,p),brier=brier_score_loss(y,p),sensitivity=recall_score(y,p>=.5))
def privacy(value):
    if isinstance(value,dict):
        assert not set(value)&{'ids','y','probabilities','predictions','RecordID','patientunitstayid','uniquepid'}
        for v in value.values():privacy(v)
    elif isinstance(value,list):
        for v in value:privacy(v)
def main():
    frozen=validate_frozen();protocol=read(ROOT/'protocol.json');evaluation=read(ROOT/'evaluation.json')
    public=read(ML/'reports/P65_colsample_summary.json');baseline=read(ML/'reports/P63_report_five_seed_summary.json')
    assert public['complete'] and public['new_final_fits']==10 and public['fixed_colsample_bytree']==.9
    assert not public['selected_on_tests'] and not public['parameter_search'] and not public['cross_seed_probability_averaging']
    assert public['sample_sd_ddof']==1 and public['sensitivity_threshold']==.5
    assert public['frozen_sha256']==sha(ROOT/'frozen.json')==evaluation['frozen_sha256']
    assert public['protocol_sha256']==sha(ROOT/'protocol.json')
    assert public['evaluator_source_sha256']==sha(ML/'src/evaluate_xgboost_colsample.py')
    verify_sources(evaluation['source_sha256'])
    actual={(r['cohort'],r['model'],r['seed']):r for r in public['individual_metrics']}
    means={(r['cohort'],r['model']):r for r in public['mean_sd']}
    assert len(actual)==50 and len(means)==10
    for case,meta in evaluation['cases'].items():
        file=ML/meta['file'];assert sha(file)==meta['sha256'];verify_sources(meta['input_sha256'])
        with np.load(file) as a:
            p=a['probabilities'];y=a['y'];ids=a['ids'];assert p.shape==(2,5,len(y))
            assert list(a['models'])==list(MODELS) and list(a['seeds'])==list(SEEDS)
            assert (len(y),int(y.sum()))==((782,91) if case.startswith('eicu') else (4000,585))
            assert len(set(ids))==len(ids)
            for j,name in enumerate(MODELS):
                values={m:[] for m in METRICS}
                for k,seed in enumerate(SEEDS):
                    row=actual[case,name,seed];computed=score(y,p[j,k])
                    for metric,value in computed.items():
                        assert abs(row[metric]-value)<1e-12;values[metric].append(value)
                for metric,items in values.items():
                    assert abs(means[case,name][metric+'_mean']-statistics.mean(items))<1e-12
                    assert abs(means[case,name][metric+'_sd']-statistics.stdev(items))<1e-12
                checked=next(r for r in public['seed_probability_checks'] if r['model']==name and r['cohort']==case)
                assert checked['unique_probability_vectors']==len({v.tobytes() for v in p[j]})
    for record in frozen['records']:
        loaded=joblib.load(ML/record['file']);model=loaded[0]['model'] if record['model']=='full_col90' else loaded
        assert model.get_params()['random_state']==record['seed']==record['actual_training_seed']
        if record['model']=='full_col90':
            expected={**protocol['Full_template_parameters'],'random_state':record['seed'],'colsample_bytree':.9}
            assert canonical(model.get_params())==expected
        else:
            for key,value in {**R1_PARAMS,'random_state':record['seed'],'colsample_bytree':.9}.items():assert model.get_params()[key]==value
        config=json.loads(model.get_booster().save_config())['learner']['gradient_booster']['tree_train_param']
        assert abs(float(config['colsample_bytree'])-.9)<1e-6
        assert all(float(config[k])==1 for k in ('subsample','colsample_bylevel','colsample_bynode'))
        assert subprocess.run(['git','check-ignore','--quiet','--','ML/'+record['file']],cwd=ML.parent).returncode==0
    old={(r['cohort'],r['model']):r for r in baseline['mean_sd']}
    for row in public['original_comparisons']:
        new=means[row['cohort'],row['model']];prior=old[row['cohort'],row['reference']]
        for metric in METRICS:
            for suffix in ('mean','sd'):
                assert row[metric+'_new_'+suffix]==new[metric+'_'+suffix]
                assert row[metric+'_reference_'+suffix]==prior[metric+'_'+suffix]
            assert row[metric+'_delta_mean']==new[metric+'_mean']-prior[metric+'_mean']
    assert len(public['main_six_metrics'])==30
    for row in public['main_six_metrics']:
        source=means[row['cohort'],row['model']] if row['model'] in MODELS else old[row['cohort'],row['model']]
        for metric in METRICS:
            for suffix in ('_mean','_sd'):assert row[metric+suffix]==source[metric+suffix]
    for filename,n in [('P65_all_seed_results.csv',50),('P65_mean_sd_results.csv',10),('P65_original_comparisons.csv',10),('P65_main_six_metrics.csv',30)]:
        with (ML/'reports'/filename).open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f));assert len(rows)==n
            for row in rows:privacy(row)
    privacy(public)
    for filename in ('P63_all_seed_results.csv','P63_mean_sd_results.csv','P63_main_model_metrics.csv'):
        assert (ML/'reports'/filename).read_bytes()==subprocess.check_output(['git','show','23e2789:ML/reports/'+filename],cwd=ML.parent)
    protected=read(P63/'protected_report_hashes.json');verify_sources(protected)
    tracked=set(filter(None,subprocess.check_output(['git','ls-files','-z'],cwd=ML.parent).decode().split('\0')))
    assert all('ML/'+p not in tracked for p in protected)
    environment=read(ML/'reports/P65_environment.json')
    assert all(environment['main_environment']['libraries'][k]==v for k,v in protocol['versions'].items())
    value=dict(complete=True,new_fits=10,actual_seeds=list(SEEDS),new_models=2,cohorts=5,individual_metric_rows=50,
        mean_sd_rows=10,comparison_rows=10,six_model_four_metric_rows=30,actual_colsample=.9,row_subsample=1.,
        other_parameters_fixed=True,all_weight_hashes_verified=True,all_primary_metrics_recomputed=True,sample_sd_ddof=1,
        original_P63_numerical_CSVs_unchanged=True,protected_report_files_preserved=len(protected),
        distinct_tree_structures=public['distinct_tree_structures'],all_new_AP_sds_nonzero=all(r['ap_sd']>0 for r in means.values()),
        no_cross_seed_probability_averaging=True,test_selection=False,environment_matches_training_protocol=True,
        summary_sha256=sha(ML/'reports/P65_colsample_summary.json'))
    write(ML/'reports/P65_verification.json',value);print(json.dumps(value,indent=2),flush=True)

if __name__=='__main__':main()
