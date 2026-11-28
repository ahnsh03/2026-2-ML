"""Preserve four-channel sources and freeze geometry-thinned cohort membership."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse
import hashlib
import json
import cv2
import numpy as np
from scipy.spatial import cKDTree
from quality_estimation.image_io import decoded_pixel_sha256,color_channels


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--cohort',choices=['development','calibration','confirmation','adaptation'],required=True)
    parser.add_argument('--fit-pack',type=Path,required=True)
    parser.add_argument('--original-ego',type=Path)
    args=parser.parse_args();study=args.study.resolve();root=study/args.cohort;registry=root/'registry'
    if registry.exists():raise ValueError('cohort registry already exists; preserve the frozen index')
    complete=read(root/'COLLECTION_COMPLETE.json');plan=read(root/'plan_frozen.json');digest=sha(root/'plan_frozen.json')
    if complete['plan_sha256']!=digest:raise ValueError('collection plan changed')
    routes={r['route_id']:r for r in plan['routes']};all_rows=[];file_checks=[];summaries=[]
    for success in sorted((root/'runs').glob('*/_SUCCESS')):
        directory=success.parent;summary=read(directory/'summary.json')
        if summary['plan_sha256']!=digest:continue
        prefix=summary['run_id'].rsplit('_'+summary['route_id']+'_w',1)[0]
        if not read(root/(prefix+'_completion.json'))['restoration'].get('verified'):
            raise ValueError('cohort has an unverified restoration')
        accepted=[json.loads(s)for s in (directory/'manifest.jsonl').read_text().splitlines()if json.loads(s).get('accepted')]
        declared=next(iter(read(directory/'index.json').values()))['rows']
        if len(accepted)!=len(declared):raise ValueError('accepted capture/index count differs')
        lookup={r['sample_id']:r for r in accepted}
        for row in declared:
            raw=lookup[row['sample_id']]
            for channel,source in raw['files'].items():file_checks.append((root/source['path'],source['sha256']))
            if set(raw['files'])!={'Intensity','Semantic','Depth','Instance'}:raise ValueError('four source channels required')
            rgb_file=study/row['image'];bgr=cv2.imread(str(rgb_file),cv2.IMREAD_UNCHANGED)
            if bgr is None:raise ValueError('native RGB missing')
            bgr=color_channels(bgr,'RGB')
            if bgr.shape!=(720,1280,3):raise ValueError('native RGB geometry differs')
            pixels=decoded_pixel_sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            role='candidate_pool'if args.cohort=='adaptation'else args.cohort
            all_rows.append(dict(row,image='e/'+study.name+'/'+row['image'],semantic='e/'+study.name+'/'+row['semantic'],
                expected_pixel_sha256=pixels,pixel_sha256=pixels,adaptation_role=role,source_role='after_r1_training',
                source_key='scene28_'+args.cohort+'_'+row['site'],parent_session_id=summary['run_id'],
                simulator_release='26.R1',original_hw=[720,1280],collection_plan_sha256=digest,
                expected_support_stratum=routes[summary['route_id']]['expected_support_stratum'],
                role_origin='Frozen before model evaluation; scout-presence support only.'))
        summaries.append(summary)
    if len(summaries)!=complete['conditions']:raise ValueError('cohort condition inventory differs')
    def verify(item):
        if sha(item[0])!=item[1]:raise ValueError('source bytes changed: '+str(item[0]))
    with ThreadPoolExecutor(8)as pool:list(pool.map(verify,file_checks))
    thinned=[]
    for summary in summaries:
        last=None
        for row in [r for r in all_rows if r['parent_session_id']==summary['run_id']]:
            xy=np.array(row['xy'])
            if last is None or np.linalg.norm(xy-last)>=3:thinned.append(row);last=xy
    fit=np.load(args.fit_pack);distance=cKDTree(fit['xy']).query(np.array([r['xy']for r in all_rows]))[0]
    if distance.min()<25:raise ValueError('cohort overlaps quality-fit spatial buffer')
    original=None
    if args.original_ego:original=float(cKDTree(np.load(args.original_ego)['ego_xy']).query(np.array([r['xy']for r in all_rows]))[0].min())
    if len(set(r['sample_id']for r in all_rows))!=len(all_rows):raise ValueError('duplicate capture IDs')
    if len(set(r['pixel_sha256']for r in all_rows))!=len(all_rows):raise ValueError('duplicate decoded RGB')
    if set(fit['ids'])&{r['sample_id']for r in all_rows}:raise ValueError('quality training ID reused')
    registry.mkdir()
    for name,rows in [('index_all.json',all_rows),('index_frozen.json',thinned)]:
        (registry/name).write_text(json.dumps({args.cohort:dict(simulator_release='26.R1',rows=rows)},indent=2)+'\n')
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),cohort=args.cohort,conditions=len(summaries),raw_frames=len(all_rows),
                thinned_frames=len(thinned),four_channel_files_verified=len(file_checks),minimum_quality_fit_distance_m=float(distance.min()),
                minimum_original_perception_ego_distance_m=original,maximum_cte_m=max(s['maximum_cte_m']for s in summaries),
                maximum_speed_mps=max(s['maximum_speed_mps']for s in summaries),all_successful_epochs_restored=True,
                index_sha256=sha(registry/'index_frozen.json'),plan_sha256=digest,source_sha256=sha(__file__),
                semantic_labels_inspected_for_membership=False,model_predictions_inspected_for_membership=False,
                scope='3m XY thinning only. Same-map qualified segments; original perception-training proximity is reported without claiming complete geographic/FOV independence. Scout frames are excluded.')
    (registry/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
