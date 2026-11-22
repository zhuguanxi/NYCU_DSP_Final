"""Verify the raw Demo tables against the publisher's SHA256SUMS, publish no IDs."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ML=Path(__file__).resolve().parents[1]
SOURCE='https://physionet.org/files/eicu-crd-demo/2.0.1/SHA256SUMS.txt'
EXPECTED={
 'patient.csv.gz':'15e1eb52169828e3cfff3b67132a087298de4d0365bbb31bb294f6043bffd474',
 'hospital.csv.gz':'4ed356f9c4e7c8ce80e89241697e94b77f21c2f325cd6ad126794b1e92805ef4',
 'lab.csv.gz':'e06852850abeb65211bc22ac43f17121ba41b14656fac862396449af84d8273a',
 'vitalPeriodic.csv.gz':'04dc093a2e77cd4c52ac7f2a47a8356fbb7456c7f15deb20d5abb1ca4303231d',
 'vitalAperiodic.csv.gz':'c294f46647990cb648b3ebfa589e83f330bdc0c2c922b5e8f23aa0ef99954635',
 'nurseCharting.csv.gz':'475000d63d4ef883b2e891d7197997201300bedc13fcf23e67add1abee11dafe',
 'respiratoryCharting.csv.gz':'21bfe647fea0a229d09ff04c775ec4270d20ae084131a987943da836df534235',
 'intakeOutput.csv.gz':'45db8ee0e64a4e3745284bde5828f740175770c45d00ea0c27daff789d18106e',
 'apachePatientResult.csv.gz':'ce799692737f7a712c59aa8fbb9a302cee7c719887f0638caed3ae144b166821',
 'apacheApsVar.csv.gz':'2dc6d63f9807c7e663ffa253998691605d527b702ac1db8cc5dda92308061e78',
 'apachePredVar.csv.gz':'99d5a4ab2c5cd07b6ee0a540c86a576e5975ed3b93df8525e3d287d5babf3eae',
 'respiratoryCare.csv.gz':'9e260512866498995b49d0f0676795b62115fc1045de3f7daf2e356769259de7',
}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    folder=ML/'dataset/EICU/eicu-crd-demo-2.0.1'
    checks=[]
    for name,expected in EXPECTED.items():
        path=folder/name; actual=sha(path)
        if actual!=expected: raise ValueError(f'Publisher checksum mismatch: {name}')
        checks.append(dict(file=name,expected_sha256=expected,actual_sha256=actual,verified=True))
    frame=pd.read_csv(folder/'patient.csv.gz',usecols=['patientunitstayid','uniquepid','hospitalid'])
    raw=dict(stays=len(frame),patients=int(frame.uniquepid.nunique()),hospital_keys=int(frame.hospitalid.nunique()))
    result=dict(release='eICU Demo 2.0.1',checksum_source=SOURCE,source_verified_on='2026-11-22',
                verified_files=checks,raw_counts=raw,
                hospital_note='Publisher description says 20 hospitals, while matching published files contain different hospital-key counts. Physical hospital count is not inferred from keys.',
                raw_data_published=False,patient_ids_published=False)
    path=ML/'reports/資料來源核對.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified_tables=len(checks),raw_counts=raw),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
