"""Resolve an explicit development release while keeping v6 immutable for controls."""
import argparse
import json
import os
from pathlib import Path

from .perception_identity import resolve_quality_perception


def resolve_active_perception(registry=None, adaptation_run=None, public_release=None):
    registry = Path(registry) if registry else Path(__file__).resolve().parents[2]/'config/perception_releases.json'
    document = json.loads(registry.read_text())
    if document.get('schema_version') != 'perception-development-releases-1':
        raise ValueError('unknown perception release registry')
    version = document['active_development_version']
    release = document['releases'][version]
    if release['kind'] == 'baseline':
        path, identity, membership = resolve_quality_perception()
    elif release['kind'] == 'adaptation':
        run = adaptation_run or os.environ.get('ML_ACTIVE_PERCEPTION_RUN')
        deployed = public_release
        if deployed is None and not run and release.get('repository_model_release'):
            relative = Path(release['repository_model_release'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('repository model release must stay within the checkout')
            deployed = registry.resolve().parent.parent / relative
        if deployed is not None:
            if run:raise ValueError('choose a deployment release or adaptation run')
            from .perception_identity import resolve_public_perception
            path,identity,membership = resolve_public_perception(deployed,registry)
        elif not run:
            raise ValueError('active adapted release requires ML_ACTIVE_PERCEPTION_RUN or explicit adaptation_run')
        else:
            path, identity, membership = resolve_quality_perception(run)
        if identity['adaptation_run_sha256'] != release['adaptation_run_sha256']:
            raise ValueError('active release adaptation lineage differs')
    else:
        raise ValueError('unknown release kind')
    if identity['checkpoint_sha256'] != release['checkpoint_sha256']:
        raise ValueError('active release checkpoint differs')
    return version, path, identity, membership


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path)
    parser.add_argument('--adaptation-run',type=Path)
    parser.add_argument('--perception-release',type=Path)
    args=parser.parse_args()
    version,path,identity,membership=resolve_active_perception(args.registry,args.adaptation_run,args.perception_release)
    manifest=path.parent/'release.json'
    declared=json.loads(manifest.read_text()).get('training_rows')if manifest.exists()else None
    print(json.dumps(dict(version=version,checkpoint=str(path),perception_identity=identity,
                          training_members=len(membership)if membership else None,declared_training_rows=declared,
                          training_membership_available=bool(membership),review_policy_implicitly_adopted=False),indent=2))


if __name__=='__main__':
    main()
