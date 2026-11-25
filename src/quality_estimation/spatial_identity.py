"""Bind nine-region inputs to the verified perception update and selected state dict."""

from __future__ import annotations
import json
import numpy as np
from .perception_identity import (
    baseline_identity,
    require_matching_identity,
    resolve_quality_perception,
    validate_identity,
)


def recipe_perception(specification):
    identity = validate_identity(
        specification.get("perception_identity", baseline_identity())
    )
    if specification["checkpoint_sha256"] != identity["checkpoint_sha256"]:
        raise ValueError("spatial recipe perception identity/checkpoint differs")
    expected = (
        "ema_state_dict" if identity["kind"] == "baseline" else "model_state_dict"
    )
    weights = specification.get(
        "perception_weights",
        "ema_state_dict" if identity["kind"] == "baseline" else None,
    )
    if weights != expected:
        raise ValueError(
            "spatial recipe must identify the matching perception weight state"
        )
    return identity, weights


def resolve_spatial_perception(specification, *, checkpoint=None, adaptation_run=None):
    expected, weights = recipe_perception(specification)
    path, actual, membership = resolve_quality_perception(adaptation_run, checkpoint)
    require_matching_identity(actual["checkpoint_sha256"], actual, expected)
    return path, actual, membership, weights


def bind_model_perception(model, identity, weights):
    validate_identity(identity)
    model.checkpoint_sha256_ = np.asarray(identity["checkpoint_sha256"])
    model.perception_identity_json_ = np.asarray(json.dumps(identity, sort_keys=True))
    model.perception_weights_ = np.asarray(weights)


def validate_model_perception(model, specification):
    identity, weights = recipe_perception(specification)
    recorded = getattr(model, "perception_identity_json_", None)
    state = getattr(model, "perception_weights_", None)
    if identity["kind"] != "baseline" and (recorded is None or state is None):
        raise ValueError(
            "updated GP requires embedded perception lineage and weight state"
        )
    if (
        recorded is not None
        and validate_identity(json.loads(str(recorded))) != identity
    ):
        raise ValueError("embedded GP perception lineage differs")
    if state is not None and str(state) != weights:
        raise ValueError("embedded GP perception weight state differs")
    return identity, weights


def validate_pack_perception(receipt, identity, weights):
    recorded = validate_identity(
        receipt.get("perception_identity", baseline_identity())
    )
    if (
        receipt["checkpoint_sha256"] != identity["checkpoint_sha256"]
        or recorded != identity
        or receipt.get("perception_weights", "ema_state_dict") != weights
    ):
        raise ValueError("feature/GT pack perception lineage or weight state differs")


def validate_prediction_perception(predictions, run):
    identity, weights = recipe_perception(run)
    for prediction in predictions:
        validate_pack_perception(prediction, identity, weights)
        for key in ["profile", "regions_sha256", "model_sha256", "recipe_sha256"]:
            if prediction[key] != run[key]:
                raise ValueError("prediction/replay provenance differs: " + key)
    return identity, weights
