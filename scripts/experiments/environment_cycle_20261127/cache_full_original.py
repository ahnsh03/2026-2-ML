"""Prepare shared original-train cache and compare every target to the original version."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,time
import cv2,numpy as np
from quality_estimation.adaptation_cache import prepare_one
from common.bridge import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','data-root','cache','output','report']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();cv2.setNumThreads(1)
    rows=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    if len(rows)!=26001 or len({r['sample_id'] for r in rows})!=26001:raise ValueError('original train must remain complete')
    if a.output.exists():raise ValueError('cache manifest exists; preserve previous output')
    start=time.monotonic();a.output.parent.mkdir(parents=True,exist_ok=True)
    def one(row):
        out=prepare_one(a.data_root,row,a.cache)
        target=cv2.imread(str(a.cache/out['cache_relative']/'targets.png'),cv2.IMREAD_UNCHANGED)
        for j,name in enumerate(['drivable','road_marking']):
            old=cv2.imread(str(a.data_root/row['original_targets'][name]),cv2.IMREAD_UNCHANGED)
            if old is None or not np.array_equal(old,target[:,:,j]):raise ValueError('original target differs: '+row['sample_id']+':'+name)
        valid=cv2.imread(str(a.data_root/row['original_valid_mask']),cv2.IMREAD_UNCHANGED)
        if valid is None or not np.array_equal(valid.astype(bool),target[:,:,2].astype(bool)):raise ValueError('original padding differs')
        return out
    temporary=a.output.with_suffix('.partial.jsonl');pixels=set()
    with temporary.open('x') as stream,ThreadPoolExecutor(a.workers) as pool:
        for j,out in enumerate(pool.map(one,rows),1):
            if out['pixel_sha256'] in pixels:raise ValueError('original duplicate pixel')
            pixels.add(out['pixel_sha256']);stream.write(json.dumps(out)+'\n')
            if j%250==0:stream.flush();print('original cache',j,'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    temporary.replace(a.output)
    report=dict(rows=len(rows),unique_decoded_pixels=len(pixels),all_original_two_head_targets_bitwise_equal=True,all_original_padding_bitwise_equal=True,raw_files_copied=False,
                original_manifest_sha256=sha256_file(a.manifest),cached_manifest_sha256=sha256_file(a.output),seconds=time.monotonic()-start,source_sha256=sha256_file(Path(__file__)))
    a.report.write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',report,flush=True)

if __name__=='__main__':main()
