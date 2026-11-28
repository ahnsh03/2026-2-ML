"""Register every planned static capture after completion; no model-based filtering."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse,json,hashlib
import cv2,numpy as np
from common.bridge import sha256_file
from quality_estimation.image_io import decoded_pixel_sha256


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=a.study
    plan=s/'protocol/new_confirmation_frozen.json';frozen=read(plan)
    cohorts_path=s/'confirmation_cohorts.json';captures=[];source_receipts=[]
    if cohorts_path.exists():
        cohorts=read(cohorts_path)
        complete=read(s/'confirmation_collection_complete.json')
        if not cohorts['complete']or cohorts['parent_plan_sha256']!=sha256_file(plan)or complete['cohorts_sha256']!=sha256_file(cohorts_path):raise ValueError('resumed collection incomplete/changed')
        directories=[s/c['directory']for c in cohorts['collections']]
        for c in cohorts['collections']:
            root=s/c['directory']
            for file,key in [('after.json','after_sha256'),('collection_context.json','context_sha256')]:
                if sha256_file(root/file)!=c[key]:raise ValueError('collection receipt changed')
            if c['rows']and sha256_file(root/'manifest.jsonl')!=c['manifest_sha256']:raise ValueError('source manifest changed')
    else:directories=[s/'new_confirmation']
    for source in directories:
        after=read(source/'after.json')
        if not all(after[k]is True for k in ['pose','resume','weather','time','control']):raise ValueError('restoration incomplete')
        context=read(source/'collection_context.json');expected_plan=plan if source.name=='new_confirmation'else s/'protocol/confirmation_subsets'/(source.name+'.json')
        if context['plan_sha256']!=sha256_file(expected_plan):raise ValueError('collection plan changed')
        if expected_plan!=plan and read(expected_plan)['parent_frozen_plan_sha256']!=sha256_file(plan):raise ValueError('resumed population differs')
        if not cohorts_path.exists()and not(source/'COMPLETE').exists():raise ValueError('incomplete collection')
        source_rows=[json.loads(line)for line in(source/'manifest.jsonl').read_text().splitlines()]if(source/'manifest.jsonl').exists()else[]
        if len(source_rows)!=after['completed_frames']:raise ValueError('accepted count differs')
        captures.extend((source,r)for r in source_rows)
        source_receipts.append(dict(directory=str(source.relative_to(s)),manifest_sha256=sha256_file(source/'manifest.jsonl')if source_rows else None,rows=len(source_rows),context_sha256=sha256_file(source/'collection_context.json')))
    if sha256_file(s/'protocol/contrast_candidate_frozen.json')!=frozen['gp_candidate_sha256']:raise ValueError('candidate changed')
    poses={p['id']:p for p in frozen['poses']};expected={(p['id'],w,h)for p in poses.values()for w,h in p['conditions']};observed=set();rows=[];pixels=[]
    for source,r in captures:
        site=int(r['site'].split('site')[-1]);slot=(site,r['weather'],r['hour'])
        if slot not in expected or slot in observed:raise ValueError('unexpected/duplicate planned slot')
        if datetime.fromisoformat(r['started_utc'])<datetime.fromisoformat(frozen['created_utc']):raise ValueError('capture before freeze')
        observed.add(slot)
        for kind,file in r['files'].items():
            path=source/file['path']
            if sha256_file(path)!=file['sha256']:raise ValueError('raw '+kind+' changed')
        rgb=cv2.cvtColor(cv2.imread(str(source/r['files']['Intensity']['path'])),cv2.COLOR_BGR2RGB)
        if rgb.shape!=(720,1280,3):raise ValueError('native camera shape differs')
        pixel=decoded_pixel_sha256(rgb);pixels.append(pixel);pose=poses[site]
        xy=[r['before']['ego']['transform']['location'][axis]for axis in ['x','y']]
        rows.append(dict(sample_id='confirmation29_'+r['sample_id'],origin_sample_id=r['sample_id'],view='front',image=str(source.relative_to(s))+'/'+r['files']['Intensity']['path'],semantic=str(source.relative_to(s))+'/'+r['files']['Semantic']['path'],depth=str(source.relative_to(s))+'/'+r['files']['Depth']['path'],instance=str(source.relative_to(s))+'/'+r['files']['Instance']['path'],expected_sha256=dict(rgb=r['files']['Intensity']['sha256'],semantic=r['files']['Semantic']['sha256']),pixel_sha256=pixel,expected_pixel_sha256=pixel,group=pose['road'],site=pose['road'],cohort='confirmation',weather=r['weather'],hour=r['hour'],xy=xy,original_hw=[720,1280],simulator_release='26.R1',collection_plan_sha256=sha256_file(plan),pose_id=site,static_completion_basis=r['completion_basis']))
    if observed!=expected:raise ValueError('planned slots missing')
    if len(set(r['sample_id']for r in rows))!=len(rows):raise ValueError('duplicate IDs')
    a.output.mkdir(parents=True,exist_ok=False);index=a.output/'index.json';index.write_text(json.dumps(dict(confirmation=dict(simulator_release='26.R1',rows=rows)),indent=2)+'\n')
    (a.output/'rgb.jsonl').write_text(''.join(json.dumps(dict(sample_id=r['sample_id'],image=r['image'],view=r['view'],expected_sha256=dict(rgb=r['expected_sha256']['rgb'])))+'\n'for r in rows))
    report=dict(rows=len(rows),unique_rgb_pixels=len(set(pixels)),exact_duplicate_pixel_groups=[n for n in Counter(pixels).values()if n>1],roads=len(set(r['site']for r in rows)),plan_sha256=sha256_file(plan),source_collections=source_receipts,index_sha256=sha256_file(index),source_sha256=sha256_file(Path(__file__)),selection_uses_perception_predictions=False,scope='All frozen static slots. Near-identical viewpoints are grouped by six physical segments; rows are not independent scene units.')
    (a.output/'registration.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
