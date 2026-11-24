"""Compute frozen features and direct native GT independently of the archived quality labels."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.bridge import DEFAULT_CHECKPOINT_SHA256, load_target_model, sha256_file
from quality_estimation.build_dataset import collate, worker_init
from quality_estimation.native import NativeSamples
from quality_estimation.labels import build_label
from quality_estimation.roi10 import reduce_heads
from quality_estimation.spatial_features import SpatialProfile, spatial_feature_schema, extract_spatial_features
from prepare import HERE, read, rows, sha, dump

sys.path.insert(0,str(HERE.parent/'roi7_features_20261124'))
from extract import extra_features, MORPH


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('data','checkpoint','out'):
        ap.add_argument(key,type=Path)
    a=ap.parse_args()
    lock=read(a.out/'lock.json')
    assert sha(a.out/'registered.jsonl')==lock['manifest_sha']
    assert sha256_file(a.checkpoint)==DEFAULT_CHECKPOINT_SHA256
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    model=load_target_model(a.checkpoint,torch.device('cuda'),weights='ema')
    prof=SpatialProfile.load()
    schema=spatial_feature_schema(prof,include_global=True)
    names=schema['ordered_features']+['global.'+k for k in MORPH+['top_entropy','patch_entropy','lane_margin']]
    registered=rows(a.out/'registered.jsonl')
    receipts={}
    for key in lock['sources']:
        batch_rows=[r for r in registered if r['session']==key]
        path=a.out/(key+'.npz')
        input_set=NativeSamples(a.data,batch_rows)
        loader=DataLoader(input_set,batch_size=12,num_workers=4,collate_fn=collate,worker_init_fn=worker_init,pin_memory=True)
        vectors,counts=[],[]
        with torch.inference_mode():
            for images,items in loader:
                drive,mark=model.model(images.cuda(non_blocking=True))
                for item,d,p in zip(items,drive.argmax(1).cpu().numpy(),mark.float().softmax(1).cpu().numpy()):
                    assert item['index']==len(vectors)
                    assert item['pixel_sha256']==batch_rows[item['index']]['pixel_sha256']
                    heads=reduce_heads(d,p,item['geometry'])
                    base=extract_spatial_features(item['rgb'],heads,prof,include_global=True)
                    v=extra_features(p[:,12:372],[np.ones((360,640),bool)])
                    vectors.append(np.r_[base,v].astype('float32'))
                    g=build_label(p.argmax(0).astype('uint8'),item['target'],item['gt_valid'],item['gt_valid'])
                    counts.append([g['tp'],g['fp'],g['fn']])
        assert np.isfinite(vectors).all() and len(vectors)==len(batch_rows)
        np.savez_compressed(path,x=vectors,counts=counts,names=names,ids=[r['sample_id'] for r in batch_rows])
        receipts[key]=sha(path)
        print(key,len(vectors),flush=True)
    dump(a.out/'features.json',dict(packs=receipts,roi_sha=prof.regions_sha256,checkpoint=model.checkpoint_sha256,
                                     schema=names,registration=sha(a.out/'lock.json')))


if __name__=='__main__':
    main()
