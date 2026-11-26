"""Dev-only risk calibration check/BO and portable model-version review bundle."""
from pathlib import Path
import argparse,json,itertools,shutil,hashlib,sys
import numpy as np
from scipy.special import ndtr
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.region_environment import RegionEnvironment
from quality_estimation.versioned_calibration import apply_road_calibration
from quality_estimation.diagnostic_targets import quality_or_near_defect_probability,auxiliary_targets
from quality_estimation.expected_deficit import bounded_expected_deficit
from quality_estimation.cluster_review import select_review,mixture_priority
from quality_estimation.bounded_bo import next_expected_improvement
from prepare import dump
sys.path.insert(0,str(Path(__file__).parents[1]/'roi9_gp_20261125'))
from study import load

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--refit',type=Path,required=True);p.add_argument('--packs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();S=a.study;out=a.output
    if out.exists():raise ValueError('policy output exists')
    out.mkdir(parents=True);selected=json.loads((a.refit/'quality_selection_frozen.json').read_text());rec=selected['selected'];model=SpatialQualityGP.load(a.refit/rec['model'])
    data,schema=load(a.packs,['development']);primary=np.array([str(site).startswith('drive26_') for site in data['site']]);assert primary.sum()==361
    x=feature_matrix(data,schema,rec['columns']);raw,cov=model.predict(x);cal=json.loads((a.refit/'calibration.json').read_text());cm,cc=apply_road_calibration(raw,cov,cal)
    supported=(data['counts'][:,:,0]+data['counts'][:,:,2]).sum(1)>=100;valid=primary&supported&np.isfinite(data['y']);ix=np.flatnonzero(valid);truth=data['y'][ix]<.7;roads=data['site'][ix]
    def macro(values,groups):return float(np.mean([np.mean(values[groups==g]) for g in np.unique(groups)]))
    p_raw=ndtr((.7-raw[:,0])/np.sqrt(np.maximum(cov[:,0,0],1e-12)));p_cal=ndtr((.7-cm[:,0])/np.sqrt(np.maximum(cc[:,0,0],1e-12)))
    raw_brier=macro((p_raw[ix]-truth)**2,roads);cal_brier=macro((p_cal[ix]-truth)**2,roads);use_cal=cal_brier<=raw_brier+.002
    mean,c=(cm,cc) if use_cal else (raw,cov);probability=p_cal if use_cal else p_raw
    noise=float(model.noise_variance)*float(model.y_scale_[0])**2;epistemic=np.sqrt(np.maximum(cov[:,0,0]-noise,0))
    environment=RegionEnvironment.load(a.refit/'environment.json');env=environment.assign(feature_matrix(data,schema,environment.columns));labels=env['environment_cluster']
    scores={'probability':probability,'expected_deficit':bounded_expected_deficit(mean[:,0],c[:,0,0],.7)/.7,'mean':1-np.clip(raw[:,0],0,1)}
    if len(model.y_mean_)==4:scores['union_near']=quality_or_near_defect_probability(mean,c)
    # Keep independent audit support when deployment labels would otherwise be selected-only.
    families=list(scores);parameters=[dict(score_family=f,uncertainty_weight=u,quota_exponent=q,audit_fraction=.1) for f,u,q in itertools.product(families,[0.,.1,.25],[0.,.5,1.])]
    coords=np.array([[families.index(v['score_family'])/max(len(families)-1,1),v['uncertainty_weight']/.25,v['quota_exponent'],v['audit_fraction']/.1] for v in parameters]);trials={}
    def objective(i):
        if i in trials:return trials[i]['objective']
        v=parameters[i];priority=mixture_priority(scores[v['score_family']][ix],epistemic[ix],v['uncertainty_weight']);details=[]
        for fraction in [.1,3072/7442]:
            budget=int(np.ceil(len(ix)*fraction));local=select_review(data['ids'][ix],priority,labels[ix],budget,quota_exponent=v['quota_exponent'],audit_fraction=v['audit_fraction']);picked=np.zeros(len(ix),bool);picked[local]=True
            def recall(groups):
                values=[picked[(groups==g)&truth].mean() for g in np.unique(groups) if ((groups==g)&truth).sum()>=3]
                return float(np.mean(values)) if values else None
            road=recall(roads);cluster=recall(labels[ix]);normal_cost=float((~truth[local]).mean());loss=.6*(1-(road if road is not None else 0))+.2*(1-(cluster if cluster is not None else 0))+.2*normal_cost
            details.append(dict(fraction=fraction,budget=budget,road_recall=road,cluster_recall=cluster,normal_fraction=normal_cost,loss=loss))
        value=float(np.mean([d['loss'] for d in details]));trials[i]=dict(parameters=v,objective=value,budgets=details);return value
    enough=truth.sum()>=20 and len(set(roads[truth]))>=2
    initial=[0,len(parameters)//3,2*len(parameters)//3,len(parameters)-1];initial=list(dict.fromkeys(initial));history=[];losses=[]
    if enough:
        for _ in range(16):
            i=next_expected_improvement(coords,history,losses,initial);history.append(i);losses.append(objective(i))
        random=initial+np.random.default_rng(261126).permutation([i for i in range(len(parameters)) if i not in initial]).tolist()[:16-len(initial)];rloss=[objective(int(i)) for i in random]
        best=min(set(history+random),key=lambda i:(trials[i]['objective'],parameters[i]['uncertainty_weight'],parameters[i]['audit_fraction'],parameters[i]['score_family']!='probability'))
        chosen=parameters[best];reason='Development-only two-budget BO/random comparison; no heldout labels'
    else:
        random=[];rloss=[];chosen=dict(score_family='probability',uncertainty_weight=0.,quota_exponent=1.,audit_fraction=.1)
        reason='Fewer than20 low-IoU development rows or2positive roads; skip BO and preserve a simple quality-ranked queue with10% random audit.'
    dump(out/'policy_search.json',dict(bo_indices=history,bo_losses=losses,random_indices=random,random_losses=rloss,shared_initial=initial,trials={str(k):v for k,v in trials.items()},
         enough_positive_support=bool(enough),positive_rows=int(truth.sum()),positive_roads=len(set(roads[truth])),selected=chosen,reason=reason,raw_brier=raw_brier,calibrated_brier=cal_brier,calibration_enabled=bool(use_cal),
         audit_fraction_floor=.1,objective='Mean dev low-IoU recall across roads/environment groups at10% and41.3% budgets, with review-normal cost. Not measured retraining gain.'))
    # Selection is frozen before any heldout data are used.
    shutil.copy2(a.refit/rec['model'],out/'quality_gp.npz');shutil.copy2(a.refit/'calibration.json',out/'calibration.json');shutil.copy2(a.refit/'environment.json',out/'environment.json')
    identity=selected['perception_identity'];policy=dict(version='versioned-review-policy-1',**chosen,model_sha256=sha(out/'quality_gp.npz'),perception_identity=identity,
         environment_sha256=sha(out/'environment.json'),calibration_sha256=sha(out/'calibration.json'),calibration_enabled=bool(use_cal),absolute_quality_threshold=.7,relative_thresholds_enabled=False,
         selection_population='Whole supplied archive buffer; uncertainty ranks are computed after inference batches are concatenated.',scope='Candidate review priority only; GT verification and admission are separate.')
    dump(out/'policy.json',policy)
    manifest=dict(version='roi9-versioned-review-bundle-1',profile=selected['profile'],regions_sha256=selected['regions_sha256'],perception_identity=identity,
          perception_weights=selected['perception_weights'],checkpoint_sha256=selected['checkpoint_sha256'],ordered_features=rec['columns'],target_names=rec['targets'],
          quality_development_acceptable_to_rf=selected['gp_development_acceptable_to_rf'],quality_selection_sha256=sha(a.refit/'quality_selection_frozen.json'),
          artifact_sha256={name:sha(out/name) for name in ['quality_gp.npz','calibration.json','environment.json','policy.json','policy_search.json']},
          source_sha256=sha(Path(__file__)),scope='Offline prototype; no live ROS, automatic training admission or operating-normality certification.')
    dump(out/'bundle_manifest.json',manifest);print('PACKAGED',chosen,'calibration',use_cal,flush=True)
if __name__=='__main__':main()
