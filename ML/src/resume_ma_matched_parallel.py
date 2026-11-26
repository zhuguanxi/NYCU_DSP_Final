"""Resume identical P62 jobs with checked parallel sorting and split evaluation."""
import os
os.environ['NUMBA_NUM_THREADS']='2'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from functools import partial
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import train_ma_matched as training
from ma_matched_parallel import parallel_splits

def initialize_worker():
    training.accelerated_splits=parallel_splits
    original=training.write
    def receipt(path,value):
        if Path(path).parent.name=='jobs':
            value={**value,'computation_backend':'exact CPU parallel features/sorting',
                'backend_source_sha256':training.sha(Path(__file__).with_name('ma_matched_parallel.py'))}
        return original(path,value)
    training.write=receipt

def main():
    root=training.ROOT;check=training.read(root/'parallel_numeric_check.json')
    assert check['all_candidate_scores']=='exact' and check['full_fold_tree_and_probability_check']=='exact'
    assert check['parallel_source_sha256']==training.sha(Path(__file__).with_name('ma_matched_parallel.py'))
    assert check['ten_tree_seconds']<check['reference_ten_tree_seconds'],'No measured speedup; keep the original runner'
    source={name:training.sha(Path(__file__).with_name(name)) for name in ('resume_ma_matched_parallel.py','ma_matched_parallel.py')}
    record=dict(cpu_only=True,workers=6,threads_per_worker=2,source_sha256=source,
        numeric_check_sha256=training.sha(root/'parallel_numeric_check.json'),
        scientific_protocol_unchanged=True,completed_receipts_reused=True)
    path=root/'parallel_resume_execution.json'
    if path.exists():assert training.read(path)==record
    else:training.write(path,record)
    training.ProcessPoolExecutor=partial(ProcessPoolExecutor,initializer=initialize_worker)
    training.train(6)
    for name,digest in source.items():assert training.sha(Path(__file__).with_name(name))==digest
    print('Identical P62 study completed with checked CPU parallel backend',flush=True)
if __name__=='__main__':main()
