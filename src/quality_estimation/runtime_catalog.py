"""Verify pinned manifests for repository-managed runtime bundles."""
from pathlib import Path
import json
from common.bridge import sha256_file


def verify_repository_bundle(path,repo_root=None):
    root=Path(repo_root)if repo_root is not None else Path(__file__).resolve().parents[2]
    path=Path(path).resolve()
    try:relative=str(path.relative_to(root.resolve()))
    except ValueError:return  # Explicit external research inputs retain their own bundle checks.
    if not relative.startswith('models/'):return
    catalog=json.loads((root/'models/catalog.json').read_text());pins={catalog['reference_review_bundle']:catalog['review_bundle_manifest_sha256']}
    for spec in catalog.get('comparison_review_bundles',{}).values():pins[spec['path']]=spec['bundle_manifest_sha256']
    if relative not in pins or sha256_file(path/'bundle_manifest.json')!=pins[relative]:raise ValueError('repository runtime bundle is not the pinned catalog release')
