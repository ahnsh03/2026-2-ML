"""Two-head native targets for an optional controlled perception-update pilot."""
from __future__ import annotations

import json

import cv2
import numpy as np
import torch

from common.bridge import sha256_file
from .build_dataset import REPO_ROOT, color_channels, read_image
from .dataset_io import require
from .native import NativeSamples
from data_collection.mask_baker import semantic_policy_masks

POLICY_PATH = REPO_ROOT / "config/perception_target_policy_v3_actor_negative.json"
POLICY_SHA256 = "21521b2b87962bdfdc8c616221826616f4691431689088a2b33c960bb85f75e7"


class NativePerceptionSamples(NativeSamples):
    def __init__(self, root, rows):
        super().__init__(root, rows)
        require(sha256_file(POLICY_PATH) == POLICY_SHA256, "frozen perception target policy changed")
        self.policy = json.loads(POLICY_PATH.read_text())

    def __getitem__(self, index):
        item = super().__getitem__(index)
        semantic, digest = read_image(self.root, self.rows[index]["semantic"], cv2.IMREAD_UNCHANGED)
        require(digest == item["hashes"]["semantic"], "Semantic changed between target reads")
        masks = semantic_policy_masks(color_channels(semantic, "Semantic"), self.policy)
        require(np.array_equal(masks["road_marking"], item["native_target"]), "quality/perception marking targets differ")
        require(not np.any(masks["drivable_ignore"]), "actor-negative policy unexpectedly contains ignored pixels")
        geometry = item["geometry"]
        x,y,_,_ = geometry.pad_ltrb; h,w = geometry.resized_hw
        drivable = np.zeros(geometry.target_hw, np.uint8)
        drivable[y:y+h,x:x+w] = cv2.resize(masks["drivable"],(w,h),interpolation=cv2.INTER_NEAREST)
        return dict(index=index, image=item["image"], targets={"drivable":drivable,"road_marking":item["target"]},
                    valid_mask=item["gt_valid"], geometry=geometry, hashes=item["hashes"], pixel_sha256=item["pixel_sha256"])


def perception_batch(items, device):
    """Make one batch without carrying source RGB or paths onto the GPU."""
    return dict(image=torch.from_numpy(np.stack([r["image"] for r in items])).to(device),
                targets={head:torch.from_numpy(np.stack([r["targets"][head] for r in items])).long().to(device)
                         for head in ("drivable","road_marking")},
                valid_mask=torch.from_numpy(np.stack([r["valid_mask"] for r in items])).bool().to(device))


def validate_adaptation_roles(new_rows, replay_rows, replay_only=False):
    require(bool(replay_rows) and (not new_rows if replay_only else bool(new_rows)),
            "replay-only control requires empty new data; other arms require new and replay data")
    require(all(r.get("preregistered_quality_role") == "candidate_pool" for r in new_rows),
            "new adaptation labels must come from the reserved candidate pool")
    require(all(r.get("source_role") == "f0_train" and r.get("preregistered_quality_role") not in ("meta_dev","final_test")
                for r in replay_rows), "replay must be known f0 training data")
    ids = [r["sample_id"] for r in new_rows + replay_rows]
    require(len(set(ids)) == len(ids), "adaptation sample IDs overlap")
    require(not {r["parent_session_id"] for r in new_rows}.intersection(r["parent_session_id"] for r in replay_rows),
            "new and replay parent groups overlap")


def validate_adaptation_selection(new_rows, arm, seed, budget):
    """Prevent a declared experiment arm from silently using another admission list."""
    if arm == "ReplayOnly":
        require(not new_rows, "replay-only control cannot contain new admissions")
        return
    require(arm in ("Random", "FrozenQualityRank", "Confidence", "Entropy", "EnvironmentAudit") and bool(new_rows),
            "unknown or empty selection arm")
    selections = [row.get("selection", {}) for row in new_rows]
    require(all(s.get("method") == arm for s in selections), "admission method differs from training arm")
    require(all(s.get("budget") == budget for s in selections), "admission budget differs from protocol")
    require(all(s.get("seed") == (seed if arm == "Random" else None) for s in selections),
            "admission seed differs from training arm")
    require([s.get("rank") for s in selections] == list(range(1, len(selections)+1)),
            "admission ranks must be complete and ordered")
    fields = ["inference_run_sha256", "model_run_sha256"]
    if arm == "EnvironmentAudit":
        fields.extend(["queue_run_sha256", "environment_centers_sha256"])
        seed_values = {s.get("review_seed") for s in selections}
        require(len(seed_values) == 1 and all(type(seed) is int for seed in seed_values), "inconsistent review seed")
        require(all(type(s.get("environment_cluster")) is int and s["environment_cluster"] >= 0 and
                    s.get("review_reason") in ("environment_round_robin", "random_remainder_audit") for s in selections),
                "environment review evidence is incomplete")
    for field in fields:
        values = {s.get(field) for s in selections}
        require(len(values) == 1 and all(isinstance(v, str) and len(v) == 64
                                       and all(c in "0123456789abcdef" for c in v) for v in values),
                "admissions must share one valid " + field)
