"""Execute the disjoint final CV block with separate receipts, then atomically cache it."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import json
import shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
import train_ma_matched as training
BASE=training.ROOT
ASSIST=BASE/'assist_elastic_gcs'

def execute(job):
    # The helper never writes a main-worker .tmp file or a final model.
    training.ROOT=ASSIST
    return training.fit_job(job)

def main():
    protocol=training.read(BASE/'protocol.json');training.verify_sources(protocol)
    ASSIST.mkdir(exist_ok=True)
    source=BASE/'protocol.json';copy=ASSIST/'protocol.json'
    if copy.exists():assert training.sha(copy)==training.sha(source)
    else:shutil.copyfile(source,copy)
    jobs=[('Elastic',True,a,str(f)) for a in (None,*training.ALPHAS) for f in range(3)]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(execute,job) for job in jobs]):
            receipt=future.result();key=receipt['key'];rows.append(receipt)
            path=BASE/'jobs'/f'{key}.json';path.parent.mkdir(exist_ok=True)
            # A complete, closed helper receipt is published with atomic create-only semantics.
            # If the main queue ever overlaps, compare scientific output, never overwrite it.
            try:os.link(ASSIST/'jobs'/f'{key}.json',path)
            except FileExistsError:
                old=training.read(path)
                for field in ('key','features','augmentation','alpha','fold','validation','protocol_sha256'):
                    assert old[field]==receipt[field],(key,field)
    training.verify_sources(protocol)
    training.write(BASE/'assist_execution.json',dict(cpu_only=True,main_workers=6,helper_workers=4,
        block='Elastic GCS60 CV only',unique_jobs=len(rows),helper_receipts_separate=True,
        protocol_sha256=training.sha(source),helper_source_sha256=training.sha(Path(__file__))))
    print('ASSIST COMPLETE: 24 cached Elastic/GCS60 CV jobs; same frozen scientific protocol',flush=True)
if __name__=='__main__':main()
