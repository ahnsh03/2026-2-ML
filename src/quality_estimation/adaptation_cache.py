"""Content-addressed, lossless training cache; original RGB/Semantic stay in place."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from camera_semantic_perception.geometry import letterbox_bgr
from common.bridge import sha256_file
from .build_dataset import native_marking
from data_collection.mask_baker import semantic_policy_masks
from .artifacts import write_json
from .features import geometric_valid
from .image_io import color_channels, decoded_pixel_sha256, read_image
from .perception_data import POLICY_PATH, POLICY_SHA256


def prepare_one(root, row, cache):
    """Build once using verified source bytes, recording original and target hashes."""
    bgr, rgb_sha = read_image(root, row['image'], cv2.IMREAD_UNCHANGED)
    semantic, sem_sha = read_image(root, row['semantic'], cv2.IMREAD_UNCHANGED)
    for key, actual in [('rgb', rgb_sha), ('semantic', sem_sha)]:
        if row.get('expected_sha256', {}).get(key, actual) != actual:
            raise ValueError('source hash mismatch: '+row['sample_id'])
    bgr = color_channels(bgr, 'RGB'); semantic = color_channels(semantic, 'Semantic')
    if bgr.shape != semantic.shape:
        raise ValueError('unaligned source pair')
    native = native_marking(semantic)  # Reject unknown colors and unsupported blue markings.
    key = hashlib.sha256((rgb_sha+sem_sha+POLICY_SHA256).encode()).hexdigest()
    out = Path(cache)/key[:2]/key
    out.mkdir(parents=True, exist_ok=True)
    if (out/'receipt.json').exists():
        receipt = json.loads((out/'receipt.json').read_text())
        for name, digest in receipt['artifacts'].items():
            if sha256_file(out/name) != digest:
                raise ValueError('training cache changed: '+str(out/name))
    else:
        if sha256_file(POLICY_PATH) != POLICY_SHA256:
            raise ValueError('target policy changed')
        image, geometry = letterbox_bgr(bgr)
        masks = semantic_policy_masks(semantic, json.loads(POLICY_PATH.read_text()))
        if not np.array_equal(masks['road_marking'],native):
            raise ValueError('independent target policy disagrees with native marking')
        if masks['drivable_ignore'].any():
            raise ValueError('actor-negative targets must not ignore actors')
        x,y,_,_ = geometry.pad_ltrb; h,w = geometry.resized_hw
        targets = np.zeros((*geometry.target_hw,3), np.uint8)
        for channel, name in enumerate(['drivable','road_marking']):
            targets[y:y+h,x:x+w,channel] = cv2.resize(masks[name],(w,h),interpolation=cv2.INTER_NEAREST)
        targets[:,:,2] = geometric_valid(geometry)
        rgb8 = np.rint(image.transpose(1,2,0)*255).astype(np.uint8)
        if not np.array_equal(rgb8.transpose(2,0,1).astype(np.float32)*(1/255),image):
            raise ValueError('lossless input cache does not match native preprocessing')
        for name, value in [('input_rgb.png',rgb8),('targets.png',targets),('native_marking.png',masks['road_marking'])]:
            if not cv2.imwrite(str(out/name),value,[cv2.IMWRITE_PNG_COMPRESSION,1]):
                raise OSError('cache write failed')
        receipt = dict(rgb_sha256=rgb_sha,semantic_sha256=sem_sha,target_policy_sha256=POLICY_SHA256,
                       pixel_sha256=decoded_pixel_sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),
                       geometry=geometry.__dict__, artifacts={p.name:sha256_file(p) for p in out.glob('*.png')},
                       input_encoding='RGB uint8 stored losslessly; cv2 read without color conversion')
        write_json(out/'receipt.json',receipt)
    return dict(row,cache_key=key,cache_relative=str(out.relative_to(cache)),pixel_sha256=receipt['pixel_sha256'],
                expected_sha256={'rgb':rgb_sha,'semantic':sem_sha},cache_receipt_sha256=sha256_file(out/'receipt.json'))


class CachedAdaptationSamples(Dataset):
    def __init__(self, cache, rows, *, verify=False):
        self.cache, self.rows = Path(cache), rows
        if verify:
            cache_root = self.cache.resolve()
            def verify_one(row):
                directory=(cache_root/row['cache_relative']).resolve()
                directory.relative_to(cache_root)
                if sha256_file(directory/'receipt.json')!=row['cache_receipt_sha256']:
                    raise ValueError('cache receipt changed')
                receipt=json.loads((directory/'receipt.json').read_text())
                if receipt['target_policy_sha256']!=POLICY_SHA256 or receipt['pixel_sha256']!=row['pixel_sha256']:
                    raise ValueError('cache lineage differs')
                for name,digest in receipt['artifacts'].items():
                    artifact = (directory/name).resolve()
                    artifact.relative_to(directory)
                    if sha256_file(artifact)!=digest:raise ValueError('cache tensor changed')
            # Storage latency dominates verification of large lossless caches.
            # Check every receipt and tensor; propagate exceptions before loading.
            with ThreadPoolExecutor(max_workers=8) as pool:
                for _ in pool.map(verify_one, rows):
                    pass

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        out=self.cache/self.rows[index]['cache_relative']
        rgb=cv2.imread(str(out/'input_rgb.png'),cv2.IMREAD_UNCHANGED)
        target=cv2.imread(str(out/'targets.png'),cv2.IMREAD_UNCHANGED)
        if rgb is None or target is None:
            raise ValueError('missing cache input')
        if rgb.shape!=(384,640,3) or target.shape!=(384,640,3) or target[:,:,0].max()>1 or target[:,:,1].max()>3 or target[:,:,2].max()>1:
            raise ValueError('invalid cached target encoding')
        return dict(image=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float().mul_(1/255),
                    targets={name:torch.from_numpy(target[:,:,j].copy()).long() for j,name in enumerate(['drivable','road_marking'])},
                    valid_mask=torch.from_numpy(target[:,:,2].copy()).bool(),index=index)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','data-root','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); cv2.setNumThreads(1)
    if a.output.exists():raise ValueError('output manifest already exists')
    a.cache.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    with ThreadPoolExecutor(a.workers) as pool, a.output.open('x') as stream:
        for j,row in enumerate(pool.map(lambda row:prepare_one(a.data_root,row,a.cache),rows)):
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            if j%200==0:stream.flush();print('prepared',j+1,'/',len(rows),flush=True)
    write_json(a.output.with_suffix('.receipt.json'),dict(rows=len(rows),manifest_sha256=sha256_file(a.manifest),
               cached_manifest_sha256=sha256_file(a.output),target_policy_sha256=POLICY_SHA256))


if __name__=='__main__':main()
