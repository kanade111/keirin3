"""Model training and inference utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import get_logger

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None


NUMERIC_FEATURES = [
    "age",
    "score",
    "backs",
    "homes",
    "starts",
    "win_rate",
    "quinella_rate",
    "top3_rate",
    "gear",
    "lane_no",
    "field_size",
    "line_count",
]

CATEGORICAL_FEATURES = ["class", "stadium", "style", "grade", "track", "prefecture"]

TARGET_COLUMN = "finish_pos"


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    feature_columns: List[str]


class Model:
    """Wrapper around the underlying scikit-learn pipeline."""

    def __init__(self, artifacts: ModelArtifacts) -> None:
        self.artifacts = artifacts
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train_from_csv(cls, races_csv: str) -> "Model":
        logger = get_logger(__name__)
        df = pd.read_csv(races_csv, encoding="utf-8-sig")
        if TARGET_COLUMN not in df.columns:
            raise ValueError("Training CSV must contain finish_pos column")
        df = df.copy()

        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        df = df[df[TARGET_COLUMN].notna()]
        df["target"] = (df[TARGET_COLUMN] == 1).astype(int)

        feature_columns = list({*NUMERIC_FEATURES, *CATEGORICAL_FEATURES})
        for column in feature_columns:
            if column not in df.columns:
                df[column] = np.nan

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ]
        )

        classifier = (
            lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
            )
            if lgb is not None
            else RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
        )

        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])
        pipeline.fit(df[feature_columns], df["target"])

        logger.info("Trained model on %s rows", len(df))
        artifacts = ModelArtifacts(pipeline=pipeline, feature_columns=feature_columns)
        return cls(artifacts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, out_dir: str | Path) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.artifacts.pipeline, out_path / "model.joblib")
        metadata = {
            "feature_columns": self.artifacts.feature_columns,
        }
        (out_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.logger.info("Saved model to %s", out_path)

    @classmethod
    def load(cls, out_dir: str | Path) -> "Model":
        out_path = Path(out_dir)
        pipeline = joblib.load(out_path / "model.joblib")
        metadata = json.loads((out_path / "metadata.json").read_text(encoding="utf-8"))
        artifacts = ModelArtifacts(pipeline=pipeline, feature_columns=metadata["feature_columns"])
        return cls(artifacts)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, cards_df: pd.DataFrame) -> np.ndarray:
        df = cards_df.copy()
        for column in self.artifacts.feature_columns:
            if column not in df.columns:
                df[column] = np.nan
        probs = self.artifacts.pipeline.predict_proba(df[self.artifacts.feature_columns])
        if probs.ndim == 2:
            positive = probs[:, -1]
        else:
            positive = probs

        # Softmax normalisation per race
        race_ids = df.get("race_id")
        if race_ids is None:
            return positive

        normalised = np.zeros_like(positive, dtype=float)
        for race_id in pd.unique(race_ids):
            mask = race_ids == race_id
            group = positive[mask]
            if group.size == 0:
                continue
            shifted = group - np.max(group)
            exp = np.exp(shifted)
            total = exp.sum()
            if total == 0:
                normalised[mask] = 1.0 / group.size
            else:
                normalised[mask] = exp / total
        return normalised
