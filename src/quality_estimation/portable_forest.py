"""NumPy-only single-output random-forest inference across the two GPU environments.

Export uses the fitted sklearn estimator; runtime loads arrays without pickle.
Inputs are finite float32 values, matching sklearn's tree input conversion.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class PortableForest:
    def __init__(
        self, metadata, offsets, left, right, features, thresholds, values, classes
    ):
        self.metadata = metadata
        self.offsets = np.asarray(offsets, np.int64)
        self.left = np.asarray(left, np.int64)
        self.right = np.asarray(right, np.int64)
        self.features = np.asarray(features, np.int64)
        self.thresholds = np.asarray(thresholds, np.float64)
        self.values = np.asarray(values, np.float64)
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(metadata["n_features"])
        self.feature_names = metadata["ordered_features"]
        self.kind = metadata["kind"]
        self._validate()

    def _validate(self):
        if self.metadata["version"] != "portable-random-forest-1" or self.kind not in [
            "regression",
            "classification",
        ]:
            raise ValueError("unsupported portable forest format")
        n = len(self.left)
        if (
            len(self.offsets) < 2
            or self.offsets[0] != 0
            or self.offsets[-1] != n
            or (np.diff(self.offsets) <= 0).any()
            or self.n_features_in_ < 1
            or len(self.feature_names) != self.n_features_in_
            or len(set(self.feature_names)) != self.n_features_in_
            or any(
                a.shape != (n,) for a in [self.right, self.features, self.thresholds]
            )
            or self.values.ndim != 2
            or self.values.shape[0] != n
            or not np.isfinite(self.thresholds).all()
            or not np.isfinite(self.values).all()
        ):
            raise ValueError("invalid forest arrays or ordered inputs")
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            left = self.left[start:end]
            right = self.right[start:end]
            leaf = left == -1
            if not np.array_equal(leaf, right == -1):
                raise ValueError("inconsistent leaf children")
            parent = np.arange(start, end)[~leaf]
            if (
                (left[~leaf] <= parent).any()
                or (right[~leaf] <= parent).any()
                or (left[~leaf] >= end).any()
                or (right[~leaf] >= end).any()
                or (self.features[start:end][~leaf] < 0).any()
                or (self.features[start:end][~leaf] >= self.n_features_in_).any()
            ):
                raise ValueError("invalid or cyclic tree graph")
        if self.kind == "classification":
            leaf = self.left == -1
            if (
                len(self.classes_) != self.values.shape[1]
                or (self.values < 0).any()
                or not np.allclose(self.values[leaf].sum(1), 1, atol=1e-10, rtol=0)
            ):
                raise ValueError("invalid leaf class probabilities")
        elif self.values.shape[1] != 1:
            raise ValueError("single-output regression required")

    @classmethod
    def from_sklearn(cls, estimator, ordered_features, context=None):
        name = type(estimator).__name__
        if (
            name not in ["RandomForestClassifier", "RandomForestRegressor"]
            or estimator.n_outputs_ != 1
        ):
            raise ValueError("fitted single-output sklearn random forest required")
        if estimator.n_features_in_ != len(ordered_features):
            raise ValueError("feature order length differs")
        kind = "classification" if name.endswith("Classifier") else "regression"
        classes = (
            np.asarray(estimator.classes_)
            if kind == "classification"
            else np.asarray([], np.int64)
        )
        if classes.dtype.kind not in "biuUS":
            raise ValueError("integer or string class labels required")
        offsets = [0]
        left = []
        right = []
        features = []
        thresholds = []
        values = []
        for model in estimator.estimators_:
            tree = model.tree_
            offset = offsets[-1]
            left.append(
                np.where(tree.children_left < 0, -1, tree.children_left + offset)
            )
            right.append(
                np.where(tree.children_right < 0, -1, tree.children_right + offset)
            )
            features.append(tree.feature)
            thresholds.append(tree.threshold)
            value = tree.value[:, 0, :].copy()
            if kind == "classification":
                leaf = tree.children_left < 0
                # Older sklearn stores weighted counts, newer versions leaf
                # probabilities. Preserve already-normalized values exactly.
                if not np.allclose(value[leaf].sum(1), 1, atol=1e-12, rtol=0):
                    total = value.sum(1, keepdims=True)
                    value /= np.where(total > 0, total, 1)
            values.append(value)
            offsets.append(offset + tree.node_count)
        import sklearn

        metadata = dict(
            version="portable-random-forest-1",
            kind=kind,
            n_features=len(ordered_features),
            ordered_features=list(ordered_features),
            context=context or {},
            source_sklearn=sklearn.__version__,
            input_dtype="float32",
            missing_values="rejected",
            aggregation="unweighted mean of fitted trees",
        )
        return cls(
            metadata,
            offsets,
            np.concatenate(left),
            np.concatenate(right),
            np.concatenate(features),
            np.concatenate(thresholds),
            np.concatenate(values),
            classes,
        )

    def validate_context(
        self, ordered_features, profile, regions_sha256, checkpoint_sha256=None
    ):
        context = self.metadata["context"]
        if (
            self.feature_names != list(ordered_features)
            or context.get("profile") != profile
            or context.get("regions_sha256") != regions_sha256
            or context.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise ValueError("portable forest input/perception context differs")

    def _mean(self, X):
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError("finite ordered forest input required")
        output = np.zeros((len(x), self.values.shape[1]), np.float64)
        roots = self.offsets[:-1]
        # Traverse all trees together, with bounded sample chunks. Keep the
        # original tree accumulation order so exported probabilities stay exact.
        for start in range(0, len(x), 256):
            batch = x[start : start + 256]
            node = np.broadcast_to(roots, (len(batch), len(roots))).copy()
            active = self.left[node] >= 0
            while active.any():
                rows, trees = np.nonzero(active)
                current = node[rows, trees]
                go_left = (
                    batch[rows, self.features[current]] <= self.thresholds[current]
                )
                node[rows, trees] = np.where(
                    go_left, self.left[current], self.right[current]
                )
                active = self.left[node] >= 0
            for tree in range(len(roots)):
                output[start : start + len(batch)] += self.values[node[:, tree]]
        return output / (len(self.offsets) - 1)

    def predict(self, X):
        mean = self._mean(X)
        return (
            self.classes_[mean.argmax(1)]
            if self.kind == "classification"
            else mean[:, 0]
        )

    def predict_proba(self, X):
        if self.kind != "classification":
            raise ValueError("class probabilities require a classifier")
        return self._mean(X)

    def save(self, path):
        np.savez_compressed(
            path,
            metadata=json.dumps(self.metadata),
            offsets=self.offsets,
            left=self.left,
            right=self.right,
            features=self.features,
            thresholds=self.thresholds,
            values=self.values,
            classes=self.classes_,
        )

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                json.loads(str(z["metadata"])),
                *(
                    z[name]
                    for name in [
                        "offsets",
                        "left",
                        "right",
                        "features",
                        "thresholds",
                        "values",
                        "classes",
                    ]
                ),
            )
