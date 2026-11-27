"""RGB-only weather/illumination histogram candidates on the frozen nine-region profile."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,time,hashlib
import cv2,numpy as np
from quality_estimation.region_candidates import candidate_schema,native_histograms
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.image_io import read_image


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.output.exists():raise ValueError('feature output exists')
    rows=[json.loads(s) for s in a.index.read_text().splitlines() if s.strip()]
    profile=SpatialProfile.load();columns=[n for n in candidate_schema(profile)['histograms'] if '.sharpness_' not in n];assert len(columns)==400
    cv2.setNumThreads(1);start=time.monotonic()
    def one(r):
        bgr,digest=read_image(Path('/mnt'),r['image'],cv2.IMREAD_UNCHANGED)
        if digest!=r['expected_sha256']['rgb']:raise ValueError('RGB bytes changed')
        rgb=cv2.cvtColor(bgr[:,:,:3],cv2.COLOR_BGR2RGB)
        if rgb.shape!=(720,1280,3):raise ValueError('fixed front geometry required')
        return native_histograms(rgb,profile,columns)
    values=[]
    with ThreadPoolExecutor(a.workers) as pool:
        for j,x in enumerate(pool.map(one,rows),1):
            values.append(x)
            if j%250==0:print('RGB environment',j,'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    x=np.stack(values).astype(np.float32);assert np.isfinite(x).all()
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,X=x,ids=[r['sample_id'] for r in rows],columns=columns)
    a.output.with_suffix('.json').write_text(json.dumps(dict(rows=len(rows),columns=columns,source_index_sha256=hashlib.sha256(a.index.read_bytes()).hexdigest(),feature_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),regions_sha256=profile.regions_sha256,uses_GT=False,uses_inference=False,seconds=time.monotonic()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('ENVIRONMENT FEATURES COMPLETE',len(rows),x.shape,flush=True)

if __name__=='__main__':main()
