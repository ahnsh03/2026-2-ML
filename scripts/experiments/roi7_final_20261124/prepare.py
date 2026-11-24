"""Register inputs using geometry and provenance only, before GT and predictions are opened."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE/'protocol.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def clearance(vertices, tree):
    """Exact 2-D minimum to prior points. Any closer point lies in midpoint ball r+length/2."""
    best=float(tree.query(vertices)[0].min())
    for a,b in zip(vertices[:-1],vertices[1:]):
        delta=b-a
        length=float(np.linalg.norm(delta))
        if best==0 or length==0:
            continue
        candidates=tree.query_ball_point((a+b)/2,best+length/2+1e-8)
        if candidates:
            offset=tree.data[candidates]-a
            t=np.clip(offset.dot(delta)/(length*length),0,1)
            best=min(best,float(np.linalg.norm(offset-t[:,None]*delta,axis=1).min()))
    return best


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','models','whole','map','out'):
        ap.add_argument(name,type=Path)
    args=ap.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'lock.json').exists()
    proto=args.data/'ml_roi10_20261123/afternoon/protocols'
    meta=args.data/'ml_quality_study_20261123/protocols'
    packs=[]
    prior_count=0
    for dr in (args.models,args.models/'reserve'):
        for name,record in read(dr/'receipt.json')['sources'].items():
            file=dr/(name+'.npz')
            assert sha(file)==record['pack_sha256']
            packs.append(file)
            prior_count+=record['rows']
    assert prior_count==14331
    for name in ('development','held'):
        file=args.whole/(name+'.npz')
        assert sha(file)==read(args.whole/(name+'_receipt.json'))['pack_sha256']
        packs.append(file)
    points,used_ids,used_pixels=[],set(),set()
    for path in packs:
        with np.load(path) as p:
            points.extend(p['xy'])
            used_ids.update(p['sample_id'])
            used_pixels.update(p['pixel_sha256'])
    tree=cKDTree(np.unique(points,axis=0))
    link_map={r['idx']:np.asarray(r['points'])[:,:2] for r in read(args.map)}
    link_cache={}
    keys=['extra_cal','newroads','shapes']+[r['key'] for r in read(proto/'archive_roster.json')['sets']
                                            if r['role']=='archive_rpc']
    registered=[]; sources={}
    for key in keys:
        path=proto/(key+'.jsonl')
        source=rows(path)
        sources[key]=sha(path)
        static=key in keys[:3]
        if static:
            plan=read(meta/('road_shape_transfer.json' if key=='shapes' else 'real_g1_extension.json'))
            assert sha(args.map)==plan['source_link_sha256']
            locations={p['id']:p for p in plan['poses']}
        for r in source:
            assert r['pixel_sha256'] not in used_pixels and r['sample_id'] not in used_ids
            used_ids.add(r['sample_id'])
            used_pixels.add(r['pixel_sha256'])
            r.update(session=key,static=static,position_distance=float(tree.query(r['xy'])[0]))
            if static:
                p=locations[int(r['pose'].rsplit('_',1)[1])]
                link=p['requested_link']
                requested=np.array([p['pose']['location'][axis] for axis in 'xy'])
                assert np.linalg.norm(np.array(r['xy'])-requested)<.1
                assert np.linalg.norm(link_map[link][p['waypoint_index']]-requested)<.1
                if link not in link_cache:
                    link_cache[link]=clearance(link_map[link],tree)
                r.update(link=link,link_distance=link_cache[link])
            else:
                r.update(link='',link_distance=0.)
            for field in ('image','semantic'):
                path=r[field]
                if static:
                    r[field]='ml_quality_study_20261123/'+path
                else:
                    assert path.startswith('/mnt/e/ml_quality_study_20261116/')
                    r[field]=path[len('/mnt/e/'):]
                assert (args.data/r[field]).is_file()
            registered.append(r)
    manifest=out/'registered.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in registered))
    refs={}
    for dim,name in PLAN['candidates'].items():
        record=read(args.models/('followup_lock.json' if dim in ('97','102') else 'lock.json'))
        for seed in PLAN['seeds']:
            key=f'{name}_{seed}'
            digest=sha(args.models/'models'/(key+'.joblib'))
            assert digest==record['models'][name]['seeds'][str(seed)]['model_sha256']
            refs[key]=digest
    primary=[r for r in registered if r['static'] and min(r['position_distance'],r['link_distance'])>=25]
    dump(out/'lock.json',dict(protocol_sha=sha(HERE/'protocol.json'),map_sha=sha(args.map),prior_locations=len(tree.data),
        prior_unique_frames=len(used_ids)-len(registered),frozen_models=refs,sources=sources,
        manifest_sha=sha(manifest),total=len(registered),primary_rows=len(primary),primary_links=len({r['link'] for r in primary})))
    print(len(registered),len(primary),len({r['link'] for r in primary}),flush=True)


if __name__=='__main__':
    main()
