"""Simple model interface for Chariloto race predictions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
MODEL_FILENAME = "model.joblib"


class Model:
    """Wrapper around a scikit-learn pipeline with graceful fallbacks."""

    def __init__(
        self,
        pipeline: Optional[Pipeline],
        feature_columns: List[str],
        fallback_key: str = "score",
    ) -> None:
        self._pipeline = pipeline
        self.feature_columns = feature_columns
        self.fallback_key = fallback_key

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def train_from_csv(cls, races_csv: str) -> "Model":
        df = pd.read_csv(races_csv, encoding="utf-8-sig")
        if df.empty:
            LOGGER.warning("Training dataset is empty; using fallback model")
            return cls(pipeline=None, feature_columns=[], fallback_key="score")

        df = df.copy()
        df["finish_pos"] = pd.to_numeric(df.get("finish_pos"), errors="coerce")
        df["score"] = pd.to_numeric(df.get("score"), errors="coerce")
        df["win_rate"] = pd.to_numeric(df.get("win_rate"), errors="coerce")
        df["lane_no"] = pd.to_numeric(df.get("lane_no"), errors="coerce")

        df = df.dropna(subset=["finish_pos", "lane_no"])
        if df.empty:
            LOGGER.warning("No labelled data available; using fallback model")
            return cls(pipeline=None, feature_columns=[], fallback_key="score")

        df["target"] = (df["finish_pos"] == 1).astype(int)
        positives = df["target"].sum()
        negatives = len(df) - positives
        if positives == 0 or negatives == 0:
            LOGGER.warning("Imbalanced labels (pos=%s, neg=%s); using fallback model", positives, negatives)
            return cls(pipeline=None, feature_columns=[], fallback_key="score")

        feature_columns = ["lane_no", "score", "win_rate"]
        available_features = [col for col in feature_columns if col in df.columns]
        if not available_features:
            LOGGER.warning("No usable features found; using fallback model")
            return cls(pipeline=None, feature_columns=[], fallback_key="score")

        df_features = df[available_features].fillna(0)

        transformer = ColumnTransformer(
            transformers=[("num", StandardScaler(), available_features)],
            remainder="drop",
        )
        pipeline = Pipeline(
            steps=[
                ("transform", transformer),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        pipeline.fit(df_features, df["target"])
        LOGGER.info("Model trained with %s samples", len(df))
        return cls(pipeline=pipeline, feature_columns=available_features, fallback_key="score")

    # ------------------------------------------------------------------
    def save(self, out_dir: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / MODEL_FILENAME
        joblib.dump({
            "pipeline": self._pipeline,
            "feature_columns": self.feature_columns,
            "fallback_key": self.fallback_key,
        }, filepath)
        LOGGER.info("Model saved to %s", filepath)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path_or_dir: str) -> "Model":
        path = Path(path_or_dir)
        if path.is_dir():
            filepath = path / MODEL_FILENAME
        else:
            filepath = path
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        payload: Dict[str, object] = joblib.load(filepath)
        pipeline = payload.get("pipeline")
        feature_columns = payload.get("feature_columns", [])
        fallback_key = payload.get("fallback_key", "score")
        return cls(pipeline=pipeline, feature_columns=list(feature_columns), fallback_key=fallback_key)

    # ------------------------------------------------------------------
    def predict_proba(self, cards_df: pd.DataFrame) -> np.ndarray:
        if cards_df.empty:
            return np.array([])
        if self._pipeline is None or not self.feature_columns:
            LOGGER.info("Using fallback probability based on %s", self.fallback_key)
            return self._fallback_probability(cards_df)

        features = self._prepare_features(cards_df)
        try:
            probabilities = self._pipeline.predict_proba(features)[:, 1]
        except Exception as exc:  # pragma: no cover - safety net
            LOGGER.error("Model prediction failed: %s", exc)
            return self._fallback_probability(cards_df)
        return probabilities

    # ------------------------------------------------------------------
    def _prepare_features(self, cards_df: pd.DataFrame) -> pd.DataFrame:
        df = cards_df.copy()
        for column in self.feature_columns:
            if column not in df.columns:
                df[column] = 0
        df = df[self.feature_columns]
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
        return df

    def _fallback_probability(self, cards_df: pd.DataFrame) -> np.ndarray:
        df = cards_df.copy()
        key = self.fallback_key
        if key not in df.columns:
            if "score" in df.columns:
                key = "score"
            elif "win_rate" in df.columns:
                key = "win_rate"
            else:
                key = None
        if key:
            values = pd.to_numeric(df.get(key), errors="coerce").fillna(0)
        else:
            values = pd.Series([1.0] * len(df))

        df["_weight"] = values
        if "race_id" not in df.columns:
            df["race_id"] = ""

        weights: List[float] = []
        for _, group in df.groupby("race_id", dropna=False):
            group_weights = group["_weight"].astype(float)
            total = group_weights.sum()
            if total <= 0:
                normalized = np.full(len(group_weights), 1.0 / len(group_weights))
            else:
                normalized = group_weights / total
            weights.extend(normalized.tolist())
        return np.array(weights)


__all__ = ["Model"]
