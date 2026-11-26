"""Prepare RGB-only prospective replay rows, plus deterministic day/night figures."""
from pathlib import Path
import argparse,json
import numpy as np

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();n=a.study;a.output.mkdir(parents=True,exist_ok=False)
 registry=read(n/'registry_test_final/index_thinned.json');rows=[r for v in registry.values() for r in v['rows']];manifest=[]
 for r in rows:
  manifest.append(dict(sample_id=r['sample_id'],image=r['image'],expected_sha256={'rgb':r['expected_sha256']['rgb']},expected_pixel_sha256=r['expected_pixel_sha256'],session_id=f"{r['site']}_w{r['weather']}_h{r['hour']}",view='front'))
 (a.output/'rgb_only_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
 pairs=[]
 for site in sorted({r['site'] for r in rows}):
  day=[r for r in rows if r['site']==site and r['weather']==0 and r['hour']==13];night=[r for r in rows if r['site']==site and r['weather']==0 and r['hour']==20]
  reference=day[len(day)//2];other=min(night,key=lambda r:np.linalg.norm(np.asarray(r['xy'])-reference['xy']))
  pairs.append(dict(site=site,day=reference['sample_id'],night=other['sample_id'],position_difference_m=float(np.linalg.norm(np.asarray(reference['xy'])-other['xy'])),scope='Same new segment and nearby recorded camera position; not an exact-pose experiment. Each frame uses its own bound GT.'))
 (a.output/'day_night_pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
 print('RGB-only prospective replay rows',len(manifest),'pairs',len(pairs),flush=True)
if __name__=='__main__':main()
