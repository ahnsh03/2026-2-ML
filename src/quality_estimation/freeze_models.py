"""Freeze a completed development run before inspecting final evaluation data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from common.bridge import sha256_file
from .dataset_io import require
from .features import feature_schema
from .model_io import load_model_run
from .perception_identity import baseline_identity, identity_from_adaptation


def create_freeze(model_run, evaluation_groups, selection_evidence=(), perception_identity=None):
    """Read model/development artifacts only; never load final labels or features.

    The prior-access declaration is an experimenter's statement, not a technical
    guarantee. Preserve the external collection and decision logs alongside it.
    """
    require(isinstance(evaluation_groups, list) and bool(evaluation_groups), "evaluation groups must be a nonempty list")
    require(all(isinstance(group, str) and group for group in evaluation_groups), "invalid evaluation group")
    require(len(set(evaluation_groups)) == len(evaluation_groups), "duplicate evaluation groups")
    directory = Path(model_run)
    bundle, run = load_model_run(directory, perception_identity=perception_identity)
    require("selection_membership.json" in run["artifact_sha256"], "final freeze requires hashed selection membership")
    used = set(bundle["train_parent_groups"]) | set(bundle["selection_parent_groups"])
    require(not used.intersection(evaluation_groups), "evaluation group was used for training or selection")
    schema_bytes = (json.dumps(feature_schema(), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    evidence = [{"path": str(Path(path)), "sha256": sha256_file(Path(path))} for path in selection_evidence]
    return dict(schema_version="ml-quality-final-freeze-1", created_utc=datetime.now(timezone.utc).isoformat(),
                model_run_sha256=sha256_file(directory / "run.json"),
                feature_schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
                checkpoint_sha256=bundle["checkpoint_sha256"], selected_model=run["selected_model"],
                perception_identity=bundle.get("perception_identity", baseline_identity()),
                feature_set=bundle["feature_set"], evaluation_role="final_test",
                evaluation_parent_groups=sorted(evaluation_groups), selection_evidence=evidence,
                final_test_accessed_before_freeze=False,
                prior_access_declaration="operator assertion; verify collection/decision logs separately",
                policy="no feature, hyperparameter, model or threshold retuning after final evaluation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-run", type=Path, required=True)
    p.add_argument("--evaluation-groups", type=Path, required=True, help="JSON list of reserved parent group IDs")
    p.add_argument("--selection-evidence", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adaptation-run", type=Path)
    a = p.parse_args()
    require(not a.output.exists(), "freeze output already exists")
    identity = identity_from_adaptation(a.adaptation_run) if a.adaptation_run else None
    value = create_freeze(a.model_run, json.loads(a.evaluation_groups.read_text()), a.selection_evidence, identity)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(selected_model=value["selected_model"], groups=value["evaluation_parent_groups"], output=str(a.output))))


if __name__ == "__main__":
    main()
