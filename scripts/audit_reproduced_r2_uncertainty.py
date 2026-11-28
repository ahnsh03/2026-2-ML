#!/usr/bin/env python3
"""Independently audit frozen r2 predictions, intervals and GP information slots."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from compare_official_review_replay import read_completed
from quality_estimation.support_aware_review import SupportAwareReview
from quality_estimation.review_targets import gaussian_utility, support_aware_targets


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def quality_metrics(pred, truth, groups, mask):
    valid=mask & np.isfinite(truth)
    error=abs(pred['estimated_iou']-truth)
    covered=(truth>=pred['interval90_lower']) & (truth<=pred['interval90_upper'])
    width=pred['interval90_upper']-pred['interval90_lower']
    def macro(values):
        return float(np.mean([values[valid & (groups==g)].mean()
                              for g in np.unique(groups[valid])]))
    return dict(rows=int(valid.sum()),mae=float(error[valid].mean()),macro_mae=macro(error),
                coverage=float(covered[valid].mean()),macro_coverage=macro(covered),
                mean_width=float(width[valid].mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gp-bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=a.root;out=a.output
    out.mkdir(parents=True,exist_ok=False)
    ref=read(r/'nas_metadata/confirmation/review_assessment/summary.json')['results']
    _,primary_rows,_=read_completed(r/'replay590_primary/gpu')
    _,rf_rows,rf_candidates=read_completed(r/'replay590_rf/gpu')
    ids=np.array([v['sample_id']for v in primary_rows])
    assert ids.tolist()==[v['sample_id']for v in rf_rows]
    predictions={name:{k:np.asarray([v[k]for v in rows])for k in
                 ['estimated_iou','quality_predictive_scale','interval90_lower','interval90_upper',
                  'review_score','utility_latent_sd','review_group','near_minus_far_error_score']}
                 for name,rows in [('primary',primary_rows),('RF_only',rf_rows)]}
    model=SupportAwareReview(a.gp_bundle)
    assert sha(a.gp_bundle/'bundle_manifest.json')==ref['GP_only']['bundle_manifest_sha256']
    assert model.policy==ref['GP_only']['policy']
    with np.load(r/'replay590_primary/gpu/features.npz')as z:
        assert z['ids'].tolist()==ids.tolist()
        lookup={n:i for i,n in enumerate(z['ordered_features'].tolist())}
        x=z['X'][:,[lookup[n]for n in model.columns]].copy()
    # Re-evaluate GP Q and error posteriors from our independently extracted RGB features.
    raw,cov=model.model.predict(x[:,model.quality_take])
    q=np.clip(raw[:,0],0,1);scale=np.sqrt(np.maximum(cov[:,0,0],1e-6))
    errors,error_cov=model.errors.predict(x[:,model.error_take],observation=False)
    utility,usd=gaussian_utility(errors,error_cov,model.weights)
    contrast,csd=gaussian_utility(errors,error_cov,np.array([1.,-1.,0.]))
    gp=dict(estimated_iou=q,quality_predictive_scale=scale,
            interval90_lower=np.clip(q+model.interval[0]*scale,0,1),
            interval90_upper=np.clip(q+model.interval[1]*scale,0,1),
            review_score=utility,utility_latent_sd=usd,review_group=np.zeros(len(ids),int),
            near_minus_far_error_score=contrast,near_minus_far_error_latent_sd=csd)
    assert model.policy['grouping']=='global' and model.policy['mode']=='supported_utility'
    predictions['GP_only']=gp
    selection={}
    selection['GP_only']=model.select_with_features(ids,gp,59,x)[0]
    info_policy=ref['GP_information_control']['policy']
    assert info_policy['model_sha256']==model.policy['model_sha256']
    assert info_policy['perception_identity']==model.identity
    # Use the previously frozen information policy, never search policies on these labels.
    model.policy=copy.deepcopy(info_policy)
    assert model.policy['grouping']=='global'
    selection['GP_information_control']=model.select_with_features(ids,gp,59,x)[0]
    model.verify_unchanged()
    for name,rows in [('primary',read_completed(r/'replay590_primary/gpu')[2]),('RF_only',rf_candidates)]:
        index={sid:i for i,sid in enumerate(ids)}
        selection[name]=np.array([index[v['sample_id']]for v in rows])
    comparisons={}
    for name,chosen in selection.items():
        assert ids[chosen].tolist()==ref[name]['selected_ids'],name
        comparisons[name]={'selected_ids_order_equal':True,'selected':len(chosen)}
    # Record all selections before opening GT in this audit (the cohort was previously inspected).
    save(out/'selections_frozen.json',dict(ids={n:ids[s].tolist()for n,s in selection.items()},
         policies={n:ref[n]['policy']for n in selection},not_a_new_blind_test=True))
    with np.load(r/'gt590_recomputed/results/recomputed_counts.npz')as z:
        assert z['ids'].tolist()==ids.tolist()
        counts=z['counts'].copy()
    with np.load(r/'nas_metadata/confirmation/packs_selected/confirmation.npz')as z:
        assert z['ids'].tolist()==ids.tolist()
        assert np.array_equal(counts,z['counts'])
        groups=z['site'].copy();hours=z['hour'].copy()
    targets=support_aware_targets(counts);truth=targets['whole_frame_iou']
    all_rows=np.ones(len(ids),bool);supported=targets['whole_gt_pixels']>=50
    quality={};table=[]
    for name,pred in predictions.items():
        quality[name]={}
        populations={'all':all_rows,'gt50':supported}
        populations.update({'hour_'+str(int(h)):hours==h for h in np.unique(hours)})
        populations.update({'segment_'+str(g):groups==g for g in np.unique(groups)})
        for population,mask in populations.items():
            m=quality_metrics(pred,truth,groups,mask);quality[name][population]=m
            table.append(dict(model=name,population=population,**m))
        for local,key in [('mae','quality_mae'),('macro_mae','quality_segment_macro_mae'),
                          ('coverage','interval90_coverage'),('macro_coverage','interval90_segment_macro_coverage'),
                          ('mean_width','interval90_mean_width')]:
            np.testing.assert_allclose(quality[name]['all'][local],ref[name][key],atol=1e-4,rtol=0)
        for local,key in [('macro_mae','gt_supported_quality_segment_macro_mae'),
                          ('macro_coverage','gt_supported_interval90_segment_macro_coverage')]:
            np.testing.assert_allclose(quality[name]['gt50'][local],ref[name][key],atol=1e-4,rtol=0)
    for name,chosen in selection.items():
        found=int(targets['near']['error'][chosen].sum());total=int(targets['near']['error'].sum())
        comparisons[name].update(near_errors_found=found,near_errors_total=total,recall=found/total)
        np.testing.assert_allclose(found/total,ref[name]['review']['near_error_recall'],atol=1e-12,rtol=0)
    near=targets['near'];far=targets['far']
    both=near['error']&far['error']
    retained=(near['gt_pixels']>=50)&(near['predicted_pixels']>=50)&~near['error']
    far_only=retained&far['error'];pair=both|far_only
    score_sets=dict(quality_predictive_sd=scale,gp_utility_latent_sd=usd,
                    gp_near_minus_far_mean=contrast,
                    rf_near_minus_far_mean=predictions['primary']['near_minus_far_error_score'],
                    gp_near_minus_far_z=np.divide(contrast,csd,out=np.zeros_like(contrast),where=csd>0))
    uncertainty={}
    for name,score in score_sets.items():
        assert np.isfinite(score).all()
        uncertainty[name]=dict(near_error_auc=float(roc_auc_score(near['error'],score)),
            near_and_far_vs_near_retained_far_auc=float(roc_auc_score(both[pair],score[pair])),
            spearman_vs_absolute_q_error=float(spearmanr(score,abs(q-truth)).statistic))
    reference_uncertainty=read(Path(__file__).resolve().parents[1]/'docs/baselines/scene_cycle_20261128/confirmation/reports/uncertainty_semantics.json')['evidence']
    for name,value in uncertainty.items():
        for key in value:
            np.testing.assert_allclose(value[key],reference_uncertainty['comparisons'][name][key],atol=1e-4,rtol=0)
    for name,pred in predictions.items():
        np.savez_compressed(out/(name+'_predictions.npz'),ids=ids,**pred)
    with (out/'quality_by_population.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    result=dict(status='matched',rows=len(ids),quality=quality,selection=comparisons,uncertainty=uncertainty,
        error_groups=dict(both=int(both.sum()),near_retained_far_error=int(far_only.sum())),
        fitted_or_retuned=False,ros_executed=False,
        scope='Independent posterior/selection/GT metric recalculation. GP information policy from frozen official assessment. No fresh holdout or new adoption decision.',
        inputs={str(p):sha(p)for p in [a.gp_bundle/'bundle_manifest.json',
          r/'replay590_primary/gpu/features.npz',r/'gt590_recomputed/results/recomputed_counts.npz',
          r/'nas_metadata/confirmation/review_assessment/summary.json']})
    save(out/'summary.json',result)
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    save(out/'source_sha256.json',{Path(__file__).name:sha(Path(__file__))})
    (out/'_SUCCESS').write_text(sha(out/'summary.json')+'\n')
    print(json.dumps(dict(status=result['status'],selection=comparisons,uncertainty=uncertainty),indent=2))


if __name__=='__main__':
    main()
