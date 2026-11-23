"""Weather classification from ten fixed views with spatial out-of-fold probability stacking."""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def band_evidence(probabilities):
    """Keep the strongest class evidence in each of the upper, far, near and hood bands."""
    return np.concatenate([probabilities[:, start:end].max(axis=1)
                           for start, end in ((0, 3), (3, 6), (6, 9), (9, 10))], axis=1)


@dataclass
class BandWeatherModel:
    regional_models: list
    combiner: object

    @classmethod
    def fit(cls, views, weather, split_groups, *, jobs=1):
        if (views.shape != (len(weather), 10, 67) or len(split_groups) != len(weather)
                or not np.isfinite(views).all()):
            raise ValueError("expected ten 67-dimensional views and a label and split group per observation")
        groups = np.unique(split_groups)
        classes = np.unique(weather)
        if len(groups) < 3 or len(classes) < 2:
            raise ValueError("training requires at least three independent groups and two weather classes")
        folds = [np.isin(split_groups, groups[fold::3]) for fold in range(3)]
        if any(not np.array_equal(np.unique(weather[~held]), classes) for held in folds):
            raise ValueError("each held-group fold must retain every training weather class")
        oof = np.empty((len(weather), 10, len(classes)), dtype=np.float64)
        models = []
        for i in range(10):
            rf = RandomForestClassifier(n_estimators=72, min_samples_leaf=3, max_features="sqrt",
                                        random_state=23409, n_jobs=jobs)
            for held in folds:
                rf.fit(views[~held, i], weather[~held])
                oof[held, i] = rf.predict_proba(views[held, i])
            rf.fit(views[:, i], weather)
            rf.n_jobs = 1
            models.append(rf)
        combiner = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=1., max_iter=1500, random_state=23409))
        combiner.fit(band_evidence(oof), weather)
        return cls(models, combiner)

    def predict_proba(self, views):
        if views.ndim != 3 or views.shape[1:] != (10, 67) or not np.isfinite(views).all():
            raise ValueError("expected ten finite 67-dimensional weather views per observation")
        regional = np.stack([rf.predict_proba(views[:, i]) for i, rf in enumerate(self.regional_models)], axis=1)
        return self.combiner.predict_proba(band_evidence(regional))
