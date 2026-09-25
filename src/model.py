"""Whiff prediction: will this swing miss? (binary classification)"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import FEATURES_CATEGORICAL, FEATURES_NUMERIC


def make_features(d: pd.DataFrame):
    X = d[FEATURES_NUMERIC + FEATURES_CATEGORICAL].copy()
    y = d["whiff"].values
    return X, y


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), FEATURES_NUMERIC),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]),
             FEATURES_CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(d: pd.DataFrame, test_size=0.25, seed=42):
    X, y = make_features(d)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    models = {
        "logistic": LogisticRegression(max_iter=2000, C=1.0),
        "gradboost": HistGradientBoostingClassifier(max_iter=300,
                                                    learning_rate=0.06,
                                                    random_state=seed),
    }
    results = {}
    for name, m in models.items():
        pipe = build_pipeline(m)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        results[name] = {
            "roc_auc": roc_auc_score(yte, p),
            "avg_precision": average_precision_score(yte, p),
            "base_rate": yte.mean(),
            "pipeline": pipe,
        }
        print(f"{name:10s} ROC-AUC={results[name]['roc_auc']:.3f}  "
              f"AP={results[name]['avg_precision']:.3f}  (base whiff rate={yte.mean():.3f})")
    return results, (Xte, yte)


def permutation_importance_top(pipe: Pipeline, Xte, yte, top_n=10):
    from sklearn.inspection import permutation_importance
    r = permutation_importance(pipe, Xte, yte, n_repeats=5,
                               random_state=0, scoring="roc_auc", n_jobs=-1)
    # permutation_importance shuffles the *input* columns of the pipeline
    feat_names = FEATURES_NUMERIC + FEATURES_CATEGORICAL
    imp = pd.Series(r.importances_mean, index=feat_names).sort_values(ascending=False)
    return imp.head(top_n)
