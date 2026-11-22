"""Register fixed f0/f1 checkpoints without reading final images or labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from common.bridge import DEFAULT_CHECKPOINT_SHA256, resolve_checkpoint, sha256_file
from .dataset_io import require
from .evaluate_perception import load_adaptation
from .perception_data import POLICY_SHA256


def create_perception_freeze(evaluation_groups, adaptation_runs=(), selection_evidence=(), checkpoint=None):
    require(isinstance(evaluation_groups, list) and bool(evaluation_groups), "nonempty final group list required")
    require(all(isinstance(g, str) and g for g in evaluation_groups), "invalid final group")
    require(len(set(evaluation_groups)) == len(evaluation_groups), "duplicate final groups")
    initial = resolve_checkpoint(checkpoint)
    require(sha256_file(initial) == DEFAULT_CHECKPOINT_SHA256, "f0 checkpoint differs")
    checkpoints = {"f0": DEFAULT_CHECKPOINT_SHA256}
    provenance = {}
    for directory in map(Path, adaptation_runs):
        require(directory.name not in checkpoints, "duplicate checkpoint registry name")
        path, run, membership = load_adaptation(directory)
        trained = set(run["train_parent_groups"]) | {r["parent_session_id"] for r in membership}
        require(not trained.intersection(evaluation_groups), "final parent used for adaptation")
        checkpoints[directory.name] = sha256_file(path)
        provenance[directory.name] = dict(run_sha256=sha256_file(directory / "run.json"),
            protocol_sha256=run["protocol_sha256"], arm=run["arm"], seed=run["seed"],
            steps=run["steps"], train_parent_groups=sorted(trained),
            checkpoint_selection=run["selection"])
    return dict(schema_version="ml-perception-final-freeze-1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        perception_checkpoints=checkpoints, adaptation_provenance=provenance,
        target_policy_sha256=POLICY_SHA256, evaluation_role="final_test",
        evaluation_parent_groups=sorted(evaluation_groups),
        selection_evidence=[dict(path=str(Path(p)), sha256=sha256_file(Path(p))) for p in selection_evidence],
        final_test_accessed_before_freeze=False,
        prior_access_declaration="operator assertion; verify experiment and collection logs separately",
        policy="evaluate every registered checkpoint; no best-seed or model selection using final results")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-groups", type=Path, required=True)
    p.add_argument("--adaptation-run", type=Path, action="append", default=[])
    p.add_argument("--selection-evidence", type=Path, action="append", default=[])
    p.add_argument("--checkpoint")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "freeze output already exists")
    value = create_perception_freeze(json.loads(a.evaluation_groups.read_text()),
        a.adaptation_run, a.selection_evidence, a.checkpoint)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(checkpoints=list(value["perception_checkpoints"]),
                         groups=value["evaluation_parent_groups"], output=str(a.output))))


if __name__ == "__main__":
    main()
