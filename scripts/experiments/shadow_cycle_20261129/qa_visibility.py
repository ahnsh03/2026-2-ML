"""Offline marking visibility cues and missed-review QA, never an online input."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from common.bridge import sha256_file
from quality_estimation.build_dataset import native_marking
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.review_targets import support_aware_targets

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','data-root','pack','assessment','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=read(a.index)['confirmation']['rows'];profile=SpatialProfile.load();roi=profile.native[::2,::2]
    with np.load(a.pack)as z:data={k:z[k]for k in z.files}
    if [r['sample_id']for r in rows]!=data['ids'].tolist():raise ValueError('QA membership differs')
    target=support_aware_targets(data['counts']);assessment=read(a.assessment/'summary.json');names=['frozen_RF','frozen_GP','RF_rank_plus_z_0.25'];selected={n:set(assessment['selection']['global'][n]['selected_ids'])for n in names};records=[]
    for i,r in enumerate(rows):
        paths=[a.data_root/r[k]for k in ['image','semantic']]
        if sha256_file(paths[0])!=r['expected_sha256']['rgb']or sha256_file(paths[1])!=r['expected_sha256']['semantic']:raise ValueError('QA source bytes changed')
        rgb=cv2.cvtColor(cv2.imread(str(paths[0])),cv2.COLOR_BGR2RGB);semantic=cv2.imread(str(paths[1]));gt=cv2.resize(np.isin(native_marking(semantic),[1,2]).astype(np.uint8),(640,360),interpolation=cv2.INTER_NEAREST).astype(bool)
        actual=np.bincount(roi[gt],minlength=9)
        if not np.array_equal(actual,data['counts'][i,:,0]+data['counts'][i,:,2]):raise ValueError('Displayed Semantic GT does not match scored GT')
        thumb=cv2.resize(rgb,(640,360),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(thumb,cv2.COLOR_RGB2GRAY).astype(float)/255.;near=np.isin(roi,[6,7]);lane=gt&near;ring=(cv2.dilate(gt.astype(np.uint8),np.ones((7,7),np.uint8))>0)&~gt&near
        def med(mask):return float(np.median(gray[mask]))if mask.any()else None
        lane_luma=med(lane);background=med(ring);delta=None if lane_luma is None or background is None else lane_luma-background
        rec=dict(sample_id=r['sample_id'],site=r['site'],pose_id=r['pose_id'],sdk_hour=int(r['hour']),sdk_weather=int(r['weather']),whole_iou=float(target['whole_frame_iou'][i])if np.isfinite(target['whole_frame_iou'][i])else None,near_gt_pixels=int(target['near']['gt_pixels'][i]),near_prediction_pixels=int(target['near']['predicted_pixels'][i]),near_fn=bool(target['near']['false_negative'][i]),near_fp=bool(target['near']['false_positive'][i]),near_error=bool(target['near']['error'][i]),far_error=bool(target['far']['error'][i]),near_luma=med(near),near_dark_fraction=float(np.mean(gray[near]<40/255.)),marking_luma=lane_luma,local_background_luma=background,signed_marking_luma_difference=delta,very_low_contrast_cue=abs(delta)<=.02 if delta is not None else None,selected={n:r['sample_id']in selected[n]for n in names},gt_count_match=True)
        records.append(rec)
    groups={}
    for hour in sorted(set(r['sdk_hour']for r in records)):
        rr=[r for r in records if r['sdk_hour']==hour];errors=[r for r in rr if r['near_error']];low=[r for r in rr if r['very_low_contrast_cue']is True]
        groups[str(hour)]=dict(rows=len(rr),near_errors=len(errors),very_low_contrast_cue=len(low),median_near_luma=float(np.median([r['near_luma']for r in rr])),median_abs_marking_luma_difference=float(np.median([abs(r['signed_marking_luma_difference'])for r in rr if r['signed_marking_luma_difference']is not None])),selected_errors={n:sum(r['selected'][n]for r in errors)for n in names},missed_errors={n:sum(not r['selected'][n]for r in errors)for n in names})
    cases=[]
    for hour in [5,6,7]:
        for event in ['near_fn','near_fp']:
            options=sorted([r for r in records if r['sdk_hour']==hour and r[event]and not r['selected']['frozen_RF']],key=lambda r:r['sample_id'])
            if options:cases.append(dict(sample_id=options[0]['sample_id'],reason='First lexicographic unselected '+event+' at SDKhour'+str(hour),hour=hour,event=event))
    options=sorted([r for r in records if r['near_gt_pixels']>=50 and r['near_prediction_pixels']>=50 and not r['near_error']and r['far_error']],key=lambda r:r['sample_id'])
    if options:cases.append(dict(sample_id=options[0]['sample_id'],reason='First lexicographic near-retained/far-error example',event='near_retained_far_error'))
    report=dict(schema='offline-marking-visibility-cue-qa-1',rows=len(records),all_displayed_gt_counts_match=True,by_hour=groups,case_choices=cases,source_sha256=sha256_file(Path(__file__)),input_sha256=dict(index=sha256_file(a.index),pack=sha256_file(a.pack),assessment=sha256_file(a.assessment/'summary.json')),scope='GT-conditioned paint/background luminance is an OFFLINE descriptive cue. Absolute difference<=0.02 is a fixed diagnostic flag, not physical visibility, normality/safety truth or a new failure threshold. No QA feature enters the online model; no selection policy is tuned.')
    (a.output/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n'for r in records));(a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
