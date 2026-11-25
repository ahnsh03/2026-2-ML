"""Compare fixed-budget spatial review and GP covariance diversification on offline replay."""

from __future__ import annotations
import argparse, math, json, time
from pathlib import Path
import numpy as np
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_risk import spatial_scores, conditional_batch
from study import load, read, save, sha


def top(score, ids, budget, excluded=()):
    available = np.array([i for i in np.lexsort((ids, -score)) if i not in excluded])
    return available[:budget]


def condition(cov, selected, noise):
    cov = cov.copy()
    for i in selected:
        cov -= np.outer(cov[:, i], cov[i, :]) / (max(cov[i, i], 0) + noise)
    return cov


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("packs", "index", "run", "evaluation", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(exist_ok=True)
    frozen = read(a.run / "frozen.json")
    record = next(r for r in frozen["models"] if r["name"] == frozen["selected"])
    gp = SpatialQualityGP.load(a.run / record["model_file"])
    keys = [k for k in read(a.index) if k.startswith("archive_")]
    data, schema = load(a.packs, keys)
    x = feature_matrix(data, schema, record["ordered_features"])
    ids = data["ids"]
    source = a.evaluation / "predictions" / (record["name"] + ".npz")
    with np.load(source) as z:
        lookup = {sid: i for i, sid in enumerate(z["ids"])}
        ix = [lookup[sid] for sid in ids]
        mean = z["mean"][ix]
        cal = z["calibrated_mean"][ix]
        cov = z["calibrated_covariance"][ix]
    scores = spatial_scores(cal, cov)
    point_far = np.minimum.reduce(
        [mean[:, 0] - 0.7, mean[:, 1] - 0.8, 0.5 - mean[:, 2]]
    )
    choices = {
        name: []
        for name in [
            "global_mean",
            "global_probability",
            "global_conditional",
            "dual_point",
            "dual_probability",
            "dual_conditional",
            "mixed_probability",
            "sigma",
            "random23411",
            "random23412",
            "random23413",
        ]
    }
    rf_record = next(r for r in frozen["models"] if r["kind"] == "RF")
    with np.load(a.evaluation / "predictions" / (rf_record["name"] + ".npz")) as z:
        lookup = {sid: i for i, sid in enumerate(z["ids"])}
        rf = np.asarray([z["mean"][lookup[sid], 0] for sid in ids])
    choices["rf_global"] = []
    windows = []
    save(
        a.output / "protocol.json",
        dict(
            model_sha256=record["model_sha256"],
            prediction_sha256=sha(source),
            source_sha256=sha(Path(__file__)),
            window=64,
            budget="ceil(0.10*actual window length)",
            dual_rule="ceil(70% budget) whole-frame low-IoU, remainder predicted normal-near-good-far-weak; no duplicate IDs",
            uncertainty_use="Anticipate variance reduction after future GT labelling; no unselected GT or fictional mean updates.",
            role="Already-open archival driving diagnostics. Far-weak band labels are visibility/structure diagnostics, not operational failure truth.",
        ),
    )
    for key in keys:
        indices = np.flatnonzero(data["group"] == key)
        for start in range(0, len(indices), 64):
            w = indices[start : start + 64]
            n = len(w)
            budget = math.ceil(0.1 * n)
            low_budget = math.ceil(0.7 * budget)
            far_budget = budget - low_budget
            covariance = gp.posterior_covariance(x[w])
            contrast = gp.posterior_covariance(x[w], [0, 1, -1])
            qnoise = gp.noise_variance * gp.y_scale_[0] ** 2
            dnoise = gp.noise_variance * float((gp.y_scale_[1:] ** 2).sum())
            ordinary = top(scores["global_low"][w], ids[w], budget)
            choices["global_probability"].extend(w[ordinary])
            choices["global_mean"].extend(w[top(-mean[w, 0], ids[w], budget)])
            choices["rf_global"].extend(w[top(-rf[w], ids[w], budget)])
            choices["global_conditional"].extend(
                w[
                    conditional_batch(
                        scores["global_low"][w],
                        covariance,
                        budget,
                        ids[w],
                        noise_variance=qnoise,
                    )
                ]
            )
            for name, low, far in [
                ("dual_point", -mean[w, 0], point_far[w]),
                (
                    "dual_probability",
                    scores["global_low"][w],
                    scores["normal_far_only"][w],
                ),
            ]:
                first = top(low, ids[w], low_budget)
                last = top(far, ids[w], far_budget, first)
                choices[name].extend(w[np.r_[first, last]])
            first = np.asarray(
                conditional_batch(
                    scores["global_low"][w],
                    covariance,
                    low_budget,
                    ids[w],
                    noise_variance=qnoise,
                )
            )
            remaining = np.array([i for i in range(n) if i not in first])
            conditioned = condition(contrast, first, dnoise)
            last = conditional_batch(
                scores["normal_far_only"][w][remaining],
                conditioned[np.ix_(remaining, remaining)],
                far_budget,
                ids[w][remaining],
                noise_variance=dnoise,
            )
            choices["dual_conditional"].extend(w[np.r_[first, remaining[last]]])
            choices["mixed_probability"].extend(
                w[
                    top(
                        0.7 * scores["global_low"][w]
                        + 0.3 * scores["normal_far_only"][w],
                        ids[w],
                        budget,
                    )
                ]
            )
            choices["sigma"].extend(w[top(scores["sigma"][w], ids[w], budget)])
            for seed in [23411, 23412, 23413]:
                index = len(windows)
                rng = np.random.default_rng(seed + index * 1009)
                choices["random" + str(seed)].extend(
                    w[rng.choice(n, budget, replace=False)]
                )
            windows.append(dict(session=key, rows=n, budget=budget))
        print("reviewed session", key, flush=True)
    low = data["y"] < 0.7
    far = (data["error_type"] == 2) & ~low
    normal = ~low
    summary = {}
    for name, indices in choices.items():
        ix = np.asarray(indices, int)
        assert len(ix) == len(set(ix))
        cells = np.floor(data["xy"][ix] / 25).astype(int)
        runs = []
        for key in keys:
            mask = data["group"] == key
            chosen = ix[data["group"][ix] == key]
            runs.append(
                dict(
                    session=key,
                    selected=len(chosen),
                    low_hits=int(low[chosen].sum()),
                    far_hits=int(far[chosen].sum()),
                    low_total=int(low[mask].sum()),
                    far_total=int(far[mask].sum()),
                )
            )
        summary[name] = dict(
            selected=len(ix),
            global_low_hits=int(low[ix].sum()),
            global_low_total=int(low.sum()),
            normal_far_hits=int(far[ix].sum()),
            normal_far_total=int(far.sum()),
            other_selected=int((~(low | far))[ix].sum()),
            unsupported_selected=int((data["error_type"][ix] < 0).sum()),
            unique_25m_cells=len(set(map(tuple, cells))),
            night_selected=int(np.isin(data["hour"][ix], [5, 20, 23]).sum()),
            sessions=runs,
        )
    assert len(set(v["selected"] for v in summary.values())) == 1
    np.savez_compressed(
        a.output / "selections.npz",
        ids=ids,
        **{name: np.asarray(ix, int) for name, ix in choices.items()},
    )
    save(
        a.output / "summary.json",
        dict(
            rows=len(ids),
            sessions=len(keys),
            windows=len(windows),
            low_prevalence=float(low.mean()),
            normal_far_prevalence=float(far.mean()),
            policies=summary,
            selections_sha256=sha(a.output / "selections.npz"),
        ),
    )
    print(
        {
            k: (v["global_low_hits"], v["normal_far_hits"], v["unique_25m_cells"])
            for k, v in summary.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
