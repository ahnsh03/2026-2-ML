"""Extract only the frozen RGB environment columns from a registered cohort."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json
import cv2,numpy as np
from common.bridge import sha256_file
from quality_estimation.scene_environment import SceneEnvironment
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.region_candidates import native_histograms


def main():
    p=argparse.ArgumentParser();
    for n in ['index','data-root','environment','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('environment output exists')
    env=SceneEnvironment(a.environment);profile=SpatialProfile.load();rows=json.loads(a.index.read_text())['confirmation']['rows'];cv2.setNumThreads(1)
    def one(r):
        path=a.data_root/r['image']
        if sha256_file(path)!=r['expected_sha256']['rgb']:raise ValueError('RGB changed')
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        return native_histograms(rgb,profile,env.hist_columns)
    with ThreadPoolExecutor(4)as pool:x=np.stack(list(pool.map(one,rows)))
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,X=x,ids=[r['sample_id']for r in rows],columns=env.hist_columns)
    a.output.with_suffix('.json').write_text(json.dumps(dict(rows=len(rows),sha256=sha256_file(a.output),index_sha256=sha256_file(a.index),columns=env.hist_columns,source_sha256=sha256_file(Path(__file__)),uses_ground_truth=False),indent=2)+'\n');print(x.shape)
if __name__=='__main__':main()
