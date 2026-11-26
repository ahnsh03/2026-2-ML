"""Verify a frozen spatial-review bundle and installed numerical runtime sources."""
from pathlib import Path
import argparse,json
from .drive_review import DriveReviewPredictor,file_sha
from .spatial_features import SpatialProfile

def verify_bundle(directory,*,simulator_release):
    root=Path(directory).resolve();manifest=json.loads((root/'bundle_manifest.json').read_text())
    if manifest['version']!='drive-review-bundle-1' or manifest['simulator_release']!=simulator_release:
        raise ValueError('bundle format/simulator release differs')
    for name,digest in manifest['artifact_sha256'].items():
        path=(root/name).resolve();path.relative_to(root)
        if file_sha(path)!=digest:raise ValueError('review artifact checksum differs: '+name)
    repo=Path(__file__).resolve().parents[2]
    for name,digest in manifest['source_sha256'].items():
        path=(repo/name).resolve();path.relative_to(repo)
        if file_sha(path)!=digest:raise ValueError('installed review source differs: '+name)
    profile=SpatialProfile.load()
    predictor=DriveReviewPredictor(root/'feature_study',root/'calibration',root/'policy/policy_frozen.json',checkpoint_sha256=manifest['checkpoint_sha256'],regions_sha256=profile.regions_sha256,simulator_release=simulator_release)
    return dict(status='verified',primary_dimensions=len(predictor.columns),far_dimensions=len(predictor.auxiliary['columns']) if predictor.auxiliary else 0,primary_training_rows=len(predictor.model.X_),far_training_rows=len(predictor.far_model.X_) if predictor.far_model else 0,RGB_smoke_rows=manifest['RGB_smoke_rows'],scope=manifest['scope'])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--simulator-release',required=True);args=parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle,simulator_release=args.simulator_release),indent=2))
if __name__=='__main__':main()
