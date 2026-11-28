"""Resolve an explicit development release while keeping v6 immutable for controls."""
import argparse
import json
import os
from pathlib import Path

from .perception_identity import resolve_quality_perception


def resolve_active_perception(registry=None, adaptation_run=None):
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
        if not run:
            raise ValueError('active adapted release requires ML_ACTIVE_PERCEPTION_RUN or explicit adaptation_run')
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
    args=parser.parse_args()
    version,path,identity,membership=resolve_active_perception(args.registry,args.adaptation_run)
    print(json.dumps(dict(version=version,checkpoint=str(path),perception_identity=identity,
                          training_members=len(membership),review_policy_implicitly_adopted=False),indent=2))


if __name__=='__main__':
    main()
