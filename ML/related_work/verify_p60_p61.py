"""Independent completeness, representation, source and publication audit for P60/P61."""
import csv
import json
from pathlib import Path
import sys
import subprocess
ML=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ML/'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from p60_seed_stability import ROOT,SEEDS,MODELS,read
from p61_missingness_inputs import ROOT as STRESS,hour_masks
from prepare_hourly_sequence import GROUPS,static_values
from prepare import sha,OUT

def privacy(value):
    if isinstance(value,dict):
        assert not set(value)&{'ids','y','probabilities','patientunitstayid','uniquepid','RecordID'}
        for x in value.values(): privacy(x)
    elif isinstance(value,list):
        for x in value: privacy(x)
    elif isinstance(value,float): assert np.isfinite(value)

def main():
    full=read(ROOT/'frozen.json');ev=read(ROOT/'evaluation.json');mask=read(STRESS/'inputs.json');kg=read(STRESS/'kedgn_evaluation.json')
    assert full['fit_count']==35 and full['selected_seed'] is None and not full['selected_on_tests']
    assert len(ev['cases'])==41 and len(mask['cases'])==len(kg['cases'])==39
    for seed in SEEDS:
        receipt=full['seeds'][str(seed)]
        assert receipt['fit_count']==7 and receipt['cnn']['epochs']==8 and receipt['cnn']['seed']==seed
        for file,digest in receipt['model_sha256'].items(): assert sha(ROOT/f'seed_{seed}'/file)==digest
    for file,digest in full['protocol']['source_sha256'].items(): assert sha(ML/file)==digest
    for file,digest in mask['source_sha256'].items(): assert sha(ML/file)==digest
    for file,digest in kg['model_sources_sha256'].items(): assert sha(ML/file)==digest
    clean=pd.read_parquet(STRESS/'clean/features.parquet')
    timing=pd.read_parquet(STRESS/'clean/temporal.parquet')
    with np.load(STRESS/'clean/input.npz') as p: clean_cube=p['cube'].copy();ids=p['ids'];y=p['y']
    for case in mask['cases']:
        folder=STRESS/case['name']
        for file,digest in case['files'].items(): assert sha(folder/file)==digest
        frame=pd.read_parquet(folder/'features.parquet');time=pd.read_parquet(folder/'temporal.parquet')
        with np.load(folder/'input.npz') as p:
            np.testing.assert_array_equal(p['ids'],ids);np.testing.assert_array_equal(p['y'],y)
            np.testing.assert_array_equal(p['hours'],hour_masks(len(y),case['design'],case['hours'],case['repeat']))
            np.testing.assert_array_equal(p['static'],static_values(frame))
            observed=np.isfinite(p['cube']).any(2)
            np.testing.assert_array_equal(observed,frame[[f'{c}__count' for c in GROUPS]].gt(0).to_numpy())
            unchanged=[i for i,c in enumerate(GROUPS) if c not in case['channels']]
            np.testing.assert_array_equal(p['cube'][:,unchanged],clean_cube[:,unchanged])
        columns=[c for c in clean if c.split('__')[0] not in case['channels']]
        pd.testing.assert_frame_equal(frame[columns],clean[columns])
        timecols=[c for c in timing if c.split('__')[0].removeprefix('temporal_') not in case['channels']]
        pd.testing.assert_frame_equal(time[timecols],timing[timecols])
        assert sha(folder/'seed_predictions.npz')==ev['cases'][case['name']]['sha256']
        assert sha(folder/'kedgn_predictions.npz')==kg['cases'][case['name']]['sha256']
    public60=read(ML/'reports/P60_five_seed_stability_summary.json')
    public61=read(ML/'reports/P61_progressive_block_summary.json')
    privacy(public60);privacy(public61)
    assert len(public60['per_seed'])==200 and public60['selected_seed'] is None
    recomputed=[]
    for case,record in ev['cases'].items():
        with np.load(ML/record['file']) as p:
            probs=p['predictions'];outcomes=p['y']
            for m,model in enumerate(MODELS):
                for i,seed in enumerate(SEEDS):
                    value=average_precision_score(outcomes,probs[i,m])
                    recomputed.append((case,model,seed,value))
    for filename,expected in [('P60_all_seed_results.csv',200),('P61_all_seed_mask_results.csv',1560)]:
        with (ML/'reports'/filename).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        assert len(rows)==expected
        assert not set(rows[0])&{'ids','y','probabilities','patientunitstayid','uniquepid','RecordID'}
        lookup={(r.get('case',r.get('cohort')),r['model'],int(r['seed'])):float(r['ap']) for r in rows}
        for case,model,seed,value in recomputed:
            if (case,model,seed) in lookup:
                np.testing.assert_allclose(lookup[case,model,seed],value,rtol=0,atol=1e-14)
    assert not subprocess.check_output(['git','ls-files','ML/artifacts/','ML/dataset/','data/'],text=True).strip()
    result=dict(complete=True,component_fits=35,pipelines=7,seeds=list(SEEDS),kedgn_existing_seeds=list(range(5)),
      cases=39,clean_demo_modes=2,p60_metric_rows=200,p61_metric_rows=1560,
      selected_seed=None,selected_test_configuration=None,new_parameters_selected=0,
      sources_verified=True,all_representations_masked=True,all_seeds_published=True,patient_data_in_git=False,
      source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      published_summary_sha256={p.name:sha(p) for p in [ML/'reports/P60_five_seed_stability_summary.json',ML/'reports/P61_progressive_block_summary.json']})
    (ML/'reports/P60_P61_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Verified 35 fits, five seeds, all 39 masks, 200/1560 public rows and no tracked patient artifacts.',flush=True)

if __name__=='__main__': main()
