"""Explicit checkpoint identity for a baseline or a hash-checked local update."""
from pathlib import Path
import json
import re

from common.bridge import DEFAULT_CHECKPOINT_SHA256, resolve_checkpoint, sha256_file

TARGET_POLICY_SHA256 = "21521b2b87962bdfdc8c616221826616f4691431689088a2b33c960bb85f75e7"


def _require(value, message):
    if not value:
        raise ValueError(message)


def baseline_identity():
    return dict(schema_version="ml-perception-identity-1", kind="baseline",
                checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                initial_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                target_policy_sha256=TARGET_POLICY_SHA256,
                adaptation_run_sha256=None)


def validate_identity(identity):
    _require(isinstance(identity, dict), "perception identity must be explicit")
    expected = set(baseline_identity())
    _require(set(identity) == expected, "perception identity fields differ")
    _require(identity["schema_version"] == "ml-perception-identity-1", "unknown perception identity schema")
    _require(identity["initial_checkpoint_sha256"] == DEFAULT_CHECKPOINT_SHA256
             and identity["target_policy_sha256"] == TARGET_POLICY_SHA256,
             "perception identity baseline/policy differs")
    _require(isinstance(identity["checkpoint_sha256"], str)
             and re.fullmatch(r"[0-9a-f]{64}", identity["checkpoint_sha256"]),
             "invalid perception checkpoint hash")
    if identity["kind"] == "baseline":
        _require(identity == baseline_identity(), "baseline identity differs")
    else:
        _require(identity["kind"] == "adaptation", "unknown perception identity kind")
        digest = identity["adaptation_run_sha256"]
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                 "updated perception requires adaptation lineage")
        _require(identity["checkpoint_sha256"] != DEFAULT_CHECKPOINT_SHA256,
                 "adaptation cannot masquerade as baseline")
    return identity


def load_adaptation(directory):
    directory = Path(directory).resolve()
    run = json.loads((directory / "run.json").read_text())
    _require(run.get("status") == "complete", "adaptation is incomplete")
    _require((directory / "_SUCCESS").read_text().strip() == sha256_file(directory / "run.json"),
             "adaptation completion hash differs")
    _require(run["initial_checkpoint_sha256"] == DEFAULT_CHECKPOINT_SHA256
             and run["target_policy_sha256"] == TARGET_POLICY_SHA256,
             "adaptation baseline/policy differs")
    _require({"checkpoint.pt", "training_membership.json", "protocol.json"}.issubset(run["artifact_sha256"]),
             "adaptation inventory incomplete")
    for relative, digest in run["artifact_sha256"].items():
        path = (directory / relative).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("adaptation artifact escapes run") from exc
        _require(sha256_file(path) == digest, "adaptation artifact changed: " + relative)
    checkpoint = directory / "checkpoint.pt"
    declared = run.get("checkpoint_sha256", run["artifact_sha256"]["checkpoint.pt"])
    _require(sha256_file(checkpoint) == declared, "adaptation checkpoint identity differs")
    return checkpoint, run, json.loads((directory / "training_membership.json").read_text())


def identity_from_adaptation(directory):
    checkpoint, run, _ = load_adaptation(directory)
    identity = dict(baseline_identity(), kind="adaptation",
                    checkpoint_sha256=sha256_file(checkpoint),
                    adaptation_run_sha256=sha256_file(Path(directory) / "run.json"))
    return validate_identity(identity)


def resolve_public_perception(directory, registry=None):
    """Load an inference-only release pinned to the reviewed repository registry.

    This is not an adaptation run: membership is deliberately unavailable and
    training still requires load_adaptation() plus the full replay ledger.
    """
    directory = Path(directory).resolve()
    manifest = directory / "release.json"
    spec = json.loads(manifest.read_text())
    _require(spec.get("schema_version") == "public-perception-release-1", "unknown public model release")
    registry = Path(registry) if registry else Path(__file__).resolve().parents[2]/"config/perception_releases.json"
    catalog = json.loads(registry.read_text())
    _require(catalog.get("schema_version") == "perception-development-releases-1", "unknown perception registry")
    entry = catalog["releases"].get(spec.get("version"))
    _require(entry is not None and entry["kind"] == "adaptation", "public release is not registered")
    _require(entry.get("repository_release_manifest_sha256") == sha256_file(manifest), "public release manifest differs")
    identity = validate_identity(spec["perception_identity"])
    _require(identity["checkpoint_sha256"] == entry["checkpoint_sha256"]
             and identity["adaptation_run_sha256"] == entry["adaptation_run_sha256"], "public release lineage differs")
    _require(spec.get("perception_weights") == "model_state_dict" and spec.get("inference_only") is True,
             "public release is not an inference-only model state")
    _require(spec.get("checkpoint") == "checkpoint.pt", "public checkpoint path differs")
    path = (directory / spec["checkpoint"]).resolve()
    _require(path.parent == directory and sha256_file(path) == identity["checkpoint_sha256"],
             "public checkpoint bytes differ")
    return path, identity, []


def resolve_quality_perception(adaptation_run=None, checkpoint=None, *, release=None):
    if release is not None:
        _require(adaptation_run is None and checkpoint is None, "choose exactly one perception source")
        return resolve_public_perception(release)
    if adaptation_run is not None:
        _require(checkpoint is None, "choose adaptation run or baseline checkpoint, not both")
        path, _, membership = load_adaptation(adaptation_run)
        return path, identity_from_adaptation(adaptation_run), membership
    path = resolve_checkpoint(checkpoint)
    _require(sha256_file(path) == DEFAULT_CHECKPOINT_SHA256,
             "custom checkpoint requires a verified adaptation run")
    return path, baseline_identity(), []


def require_matching_identity(checkpoint_sha256, recorded=None, expected=None):
    expected = validate_identity(expected if expected is not None else baseline_identity())
    recorded = validate_identity(recorded if recorded is not None else baseline_identity())
    _require(recorded == expected and checkpoint_sha256 == expected["checkpoint_sha256"],
             "perception checkpoint/identity mismatch; specify the matching adaptation run explicitly")
    return recorded
