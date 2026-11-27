"""Audit RGB environment discovery against simulator tags and geographic transfer."""
from pathlib import Path
import argparse,json,time,itertools
import numpy as np,joblib,faiss
from faiss.contrib.e_means import NumpyEmeans
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score,adjusted_rand_score,adjusted_mutual_info_score,confusion_matrix
from quality_estimation.portable_forest import PortableForest
from study_data import read,dump,sha,load_sources,memberships,lighting,folds


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();S=a.study;out=S/'environment';out.mkdir(exist_ok=True);faiss.omp_set_num_threads(2)
 data,schema,rows=load_sources(a.parent);member=memberships(S,data['ids']);look={s:i for i,s in enumerate(data['ids'])}
 hz=[np.load(S/name) for name in ['environment_features.npz','environment_promoted_features.npz']];hids=np.concatenate([z['ids'] for z in hz]);hx=np.concatenate([z['X'] for z in hz]);hcols=hz[0]['columns'].tolist();hlook={s:i for i,s in enumerate(hids)};assert len(hlook)==len(hids)
 pool=np.flatnonzero(data['role']=='candidate_pool');fit=np.unique(np.r_[pool,member['quality_fit']]);dev=member['development'];primary=np.array([rows[i]['source_key'].startswith('drive_development_') for i in dev]);y=data['weather'].astype(int);light=lighting(data['hour']);assert ((y[fit]>=0)&(y[fit]<6)).all()
 hist=lambda ix: hx[[hlook[data['ids'][i]] for i in ix]]
 recipe=read('config/region_feature_recipe_v2.json');weathercols=recipe['tasks']['weather']['ordered_features'];hourcols=recipe['tasks']['hour']['ordered_features'];selected_hist=list(dict.fromkeys(weathercols+hourcols));top_hist=[n for n in hcols if n.startswith(('upper_top_','hood.'))];views={'paper_selected_hist':selected_hist,'all_hist400':hcols,'top_hood_hist':top_hist}
 all_ids=np.unique(np.r_[fit,dev,member['calibration']]);hist_all=hist(all_ids);offset={i:j for j,i in enumerate(all_ids)};f=np.array([offset[i] for i in fit]);d=np.array([offset[i] for i in dev]);c=np.array([offset[i] for i in pool]);splits=folds(data['xy'][fit]);classifier_records=[];models={};embedding={}
 for view,columns in views.items():
  x=hist_all[:,[hcols.index(n) for n in columns]];errors=[]
  for tr,va in splits:
   score=[]
   for target in [y,light]:
    m=RandomForestClassifier(n_estimators=160,max_depth=16,min_samples_leaf=2,max_features=.5,class_weight='balanced_subsample',random_state=271127,n_jobs=2).fit(x[f[tr]],target[fit[tr]]);score.append(float(balanced_accuracy_score(target[fit[va]],m.predict(x[f[va]]))))
   errors.append(score)
  prob=[];oob=[];model_pair=[]
  for name,target in [('weather',y),('lighting',light)]:
   m=RandomForestClassifier(n_estimators=200,max_depth=16,min_samples_leaf=2,max_features=.5,class_weight='balanced_subsample',oob_score=True,random_state=271127,n_jobs=2).fit(x[f],target[fit]);model_pair.append(m);prob.append(m.predict_proba(x));oob.append(m.oob_decision_function_)
   portable=PortableForest.from_sklearn(m,columns,dict(profile=schema['profile'],regions_sha256=schema['regions_sha256'],role='RGB-only environment, independent of perception checkpoint',target=name));portable.save(out/f'{view}_{name}.npz');assert np.max(abs(portable.predict_proba(x[d])-m.predict_proba(x[d])))<1e-10
  record=dict(view=view,columns=columns,geographic_cv_scores=errors,mean_cv_weather=float(np.mean(errors,0)[0]),mean_cv_lighting=float(np.mean(errors,0)[1]),dev_weather=float(balanced_accuracy_score(y[dev[primary]],model_pair[0].predict(x[d[primary]]))),dev_lighting=float(balanced_accuracy_score(light[dev[primary]],model_pair[1].predict(x[d[primary]]))),historical_weather=float(balanced_accuracy_score(y[dev[~primary]],model_pair[0].predict(x[d[~primary]]))))
  classifier_records.append(record);models[view]=model_pair
  # OOB probabilities for centroid fitting, full-forest probabilities for new
  # images. This is explicitly supervised RGB representation, not label-free.
  train_lookup={int(i):j for j,i in enumerate(fit)};probs=np.concatenate(prob,axis=1);trainprobs=np.concatenate(oob,axis=1);candidate_oob=np.stack([trainprobs[train_lookup[int(i)]] for i in pool]);embedding[view]=(np.sqrt(np.maximum(probs,0)),np.sqrt(np.maximum(candidate_oob,0)))
  dump(out/'classifiers_partial.json',classifier_records);print('ENV CLASSIFIER',view,record['mean_cv_weather'],record['dev_weather'],record['dev_lighting'],flush=True)
 best=min(classifier_records,key=lambda r:(-(r['mean_cv_weather']+r['mean_cv_lighting']),len(r['columns'])));chosen=best['view'];learned,train_learned=embedding[chosen]
 # Raw RGB27/54 and paper histograms remain explicit unsupervised controls.
 rgb=data['rgb'][all_ids];rgbcols=schema['rgb'];photo=['intensity','intensity_std','dark_fraction','glare_fraction','intensity_p90_minus_p10','saturation_median','blue_minus_red','hue_cos','hue_sin'];raw27=[n for n in rgbcols if n.startswith(('upper_top_','hood.')) and n.split('.')[1] in photo];raw54=[n for n in rgbcols if n.startswith(('upper_top_','hood.'))]
 representations={'rgb27':(rgb[:,[rgbcols.index(n) for n in raw27]],None,raw27),'rgb54':(rgb[:,[rgbcols.index(n) for n in raw54]],None,raw54),'paper_hist_pca':(hist_all[:,[hcols.index(n) for n in selected_hist]],None,selected_hist),'supervised_prob9':(learned,train_learned,['sqrt_weather_probability_'+str(i) for i in range(6)]+['sqrt_lighting_probability_'+str(i) for i in range(3)])}
 records=[];archives={}
 for name,(raw,train_raw,columns) in representations.items():
  scaler=StandardScaler().fit(raw[c]) if name!='supervised_prob9' else None
  x=scaler.transform(raw) if scaler else raw;train_x=scaler.transform(train_raw) if scaler and train_raw is not None else (train_raw if train_raw is not None else x[c]);pca=None
  if name=='paper_hist_pca':pca=PCA(n_components=min(24,len(columns)),whiten=True,random_state=271127).fit(x[c]);x=pca.transform(x);train_x=pca.transform(train_x)
  active=np.flatnonzero(train_x.var(0)>1e-12);x=x[:,active];train_x=train_x[:,active]
  for k in [8,16,24]:
   assignments=[]
   for seed in [271127,271128,271129]:
    centers,_=NumpyEmeans(train_x.shape[1],k,seed=seed,nepochs=5).train(np.ascontiguousarray(train_x,np.float32));distance=cdist(x,centers,'sqeuclidean');assigned=distance.argmin(1);train_labels=cdist(train_x,centers,'sqeuclidean').argmin(1);assignments.append(train_labels);mapping=[]
    for target in [y,light]:
     classes=np.unique(target[pool]);freq={int(t):max(int((target[pool]==t).sum()),1) for t in classes};mapping.append({g:int(max(classes,key=lambda t:np.sum((train_labels==g)&(target[pool]==t))/freq[int(t)])) for g in range(k)})
    predicted=[np.array([mapping[j][int(g)] for g in assigned[d[primary]]]) for j in range(2)];record=dict(representation=name,method='official_e_means',k=k,seed=seed,fit_rows=len(pool),active_dimensions=len(active),dev_weather=float(balanced_accuracy_score(y[dev[primary]],predicted[0])),dev_lighting=float(balanced_accuracy_score(light[dev[primary]],predicted[1])),weather_ami=float(adjusted_mutual_info_score(y[pool],train_labels)),lighting_ami=float(adjusted_mutual_info_score(light[pool],train_labels)),counts=np.bincount(train_labels,minlength=k).tolist(),supervised_representation=name=='supervised_prob9')
    records.append(record)
    if seed==271127:
     path=out/f'{name}_k{k}.npz';np.savez_compressed(path,centers=centers,active=active,assigned=assigned,train_assigned=train_labels,all_source_indices=all_ids);joblib.dump(dict(scaler=scaler,pca=pca,columns=columns,active=active),out/f'{name}_transform.joblib');archives[(name,k)]=str(path.relative_to(S))
   stability=float(np.mean([adjusted_rand_score(assignments[i],assignments[j]) for i,j in [(0,1),(0,2),(1,2)]]))
   for r in records[-3:]:r['seed_ari']=stability
   print('ENV CLUSTERS',name,k,records[-3]['dev_weather'],records[-3]['dev_lighting'],stability,flush=True)
 # Matched classical controls on RGB27; same representation and first seed.
 raw=representations['rgb27'][0];sx=StandardScaler().fit(raw[c]);x=sx.transform(raw);active=np.flatnonzero(x[c].var(0)>1e-12);x=x[:,active]
 for method in ['lloyd','diagonal_gmm']:
  model=KMeans(16,n_init=3,max_iter=100,random_state=271127) if method=='lloyd' else GaussianMixture(16,covariance_type='diag',max_iter=100,reg_covar=1e-4,random_state=271127)
  z=model.fit_predict(x[c]);dd=model.predict(x[d[primary]]);scores=[]
  for target in [y,light]:
   classes=np.unique(target[pool]);mapping={g:int(max(classes,key=lambda t:np.sum((z==g)&(target[pool]==t))/max(int((target[pool]==t).sum()),1))) for g in range(16)};pred=np.array([mapping[int(g)] for g in dd]);scores.append(float(balanced_accuracy_score(target[dev[primary]],pred)))
  records.append(dict(representation='rgb27',method=method,k=16,dev_weather=scores[0],dev_lighting=scores[1]));print('ENV CONTROL',method,scores,flush=True)
 first=[r for r in records if r['method']=='official_e_means' and r['seed']==271127];passing=[r for r in first if r['dev_weather']>=.8 and r['dev_lighting']>=.9 and r['seed_ari']>=.7]
 eligible=passing or first;winner=max(eligible,key=lambda r:(r['dev_weather']+r['dev_lighting']+.1*r['seed_ari']-.001*r['k']))
 result=dict(classifiers=classifier_records,selected_classifier=best,clusters=records,selected_cluster=winner,selected_cluster_artifact=archives[(winner['representation'],winner['k'])],semantic_cluster_gate_passes=bool(passing),direct_rgb_tag_gate_passes=bool(best['dev_weather']>=.8 and best['dev_lighting']>=.9),threshold_grouping='Choose supported RGB groups if gates pass; otherwise oracle simulator tags remain the offline reference. No fabricated weather labels.',simulator_weather_classes_in_primary_dev=sorted(set(y[dev[primary]].tolist())),source_sha256=sha(__file__),scope='Classifier representation selection uses geographic CV; cluster recipe/semantic support uses development only. Weather labels are simulator settings, not visibility-normality truth. No fresh confirmation opened.')
 dump(out/'environment_selection_frozen.json',result);(out/'_SUCCESS').write_text(sha(out/'environment_selection_frozen.json')+'\n');print('ENVIRONMENT COMPLETE',winner['representation'],winner['k'],'gate',bool(passing),flush=True)

if __name__=='__main__':main()
