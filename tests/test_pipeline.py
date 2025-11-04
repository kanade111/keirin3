from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bets_from_csv import build_bets
from model import Model
from scrape.chariloto_cards import fetch_cards_for_ids
from scrape.chariloto_results import fetch_results_for_ids
from scrape.normalize import TRAINING_COLUMNS, to_training_csv


class DummyResponse:
    def __init__(self, text: str, url: str = "http://test") -> None:
        self.text = text
        self.url = url
        self.status_code = 200

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None


class DummySession:
    def __init__(self, text: str) -> None:
        self.text = text

    def get(self, url: str, params=None, **kwargs):  # pragma: no cover - simple stub
        return DummyResponse(self.text, url)


def load_html(name: str) -> str:
    return (Path(__file__).parent / "data" / name).read_text(encoding="utf-8")


def test_fetch_cards_parses_entries():
    session = DummySession(load_html("card_sample.html"))
    info_df, entry_df = fetch_cards_for_ids(
        ["20240101CL2401"], session=session, rate_limit=0
    )
    assert not entry_df.empty
    assert entry_df.loc[0, "rider_name"] == "選手 太郎"
    assert info_df.loc[0, "stadium"] == "サンプルバンク"


def test_fetch_results_parses_tables():
    session = DummySession(load_html("result_sample.html"))
    info_df, entry_df, payout_df = fetch_results_for_ids(
        ["20240101CL2401"], session=session, rate_limit=0
    )
    assert not entry_df.empty
    assert int(entry_df.loc[0, "finish_pos"]) == 1
    assert not payout_df.empty
    assert payout_df.loc[0, "combination"] == "1-2"


def test_normalize_outputs_expected_columns(tmp_path: Path):
    cards_session = DummySession(load_html("card_sample.html"))
    results_session = DummySession(load_html("result_sample.html"))
    cards_info, cards_entries = fetch_cards_for_ids(
        ["20240101CL2401"], session=cards_session, rate_limit=0
    )
    results_info, results_entries, payouts = fetch_results_for_ids(
        ["20240101CL2401"], session=results_session, rate_limit=0
    )
    combined_info = pd.concat([cards_info, results_info], ignore_index=True)
    combined_entries = cards_entries.merge(
        results_entries[["race_id", "lane_no", "finish_pos"]],
        on=["race_id", "lane_no"],
        how="left",
    )
    out_path = tmp_path / "races.csv"
    df = to_training_csv(out_path, combined_info, combined_entries, payouts)
    assert out_path.exists()
    assert list(df.columns) == TRAINING_COLUMNS
    assert int(df.loc[0, "finish_pos"]) == 1


def test_model_training_and_prediction(tmp_path: Path):
    data = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "race_no": [1, 1, 2, 2],
            "stadium": ["A", "A", "B", "B"],
            "track": [None, None, None, None],
            "title": ["Series", "Series", "Series", "Series"],
            "race_name": ["Race1", "Race1", "Race2", "Race2"],
            "grade": ["G3", "G3", "G2", "G2"],
            "class": ["S1", "S2", "S1", "S2"],
            "lane_no": [1, 2, 1, 2],
            "rider_id": ["r1", "r2", "r3", "r4"],
            "rider_name": ["A", "B", "C", "D"],
            "age": [30, 31, 29, 32],
            "prefecture": ["Tokyo", "Osaka", "Tokyo", "Osaka"],
            "score": [100, 98, 95, 93],
            "style": ["逃げ", "差し", "逃げ", "差し"],
            "backs": [5, 3, 4, 2],
            "homes": [2, 1, 3, 2],
            "starts": [10, 11, 12, 13],
            "win_rate": [0.5, 0.3, 0.6, 0.2],
            "quinella_rate": [0.6, 0.4, 0.5, 0.3],
            "top3_rate": [0.7, 0.5, 0.6, 0.4],
            "kimarite_nige": [1, 0, 1, 0],
            "kimarite_makuri": [0, 1, 0, 1],
            "kimarite_sashi": [0, 1, 0, 1],
            "kimarite_mark": [0, 0, 0, 0],
            "finish_pos": [1, 2, 1, 2],
            "line_id": [None, None, None, None],
            "line_pos": [1, 2, 1, 2],
            "gear": [3.83, 3.83, 3.92, 3.92],
            "bank_code": ["24", "24", "11", "11"],
            "source": ["chariloto"] * 4,
            "field_size": [9, 9, 9, 9],
            "line_count": [3, 3, 3, 3],
            "line_pattern": [None, None, None, None],
            "start_time": [None, None, None, None],
            "weather": [None, None, None, None],
            "wind": [None, None, None, None],
            "kaizai_no": [None, None, None, None],
            "term": ["Series", "Series", "Series", "Series"],
        }
    )
    races_path = tmp_path / "races.csv"
    data.to_csv(races_path, index=False, encoding="utf-8-sig")
    model = Model.train_from_csv(str(races_path))
    model_dir = tmp_path / "model"
    model.save(model_dir)
    loaded = Model.load(model_dir)
    cards_df = data.drop(columns=["finish_pos", "title", "kimarite_nige", "kimarite_makuri", "kimarite_sashi", "kimarite_mark"])
    probs = loaded.predict_proba(cards_df)
    assert probs.shape[0] == len(cards_df)
    for race_id in cards_df["race_id"].unique():
        mask = cards_df["race_id"] == race_id
        assert np.isclose(probs[mask].sum(), 1.0)


def test_build_bets_generates_rows():
    cards = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "race_name": ["Race1"] * 3,
            "stadium": ["A"] * 3,
            "lane_no": [1, 2, 3],
            "rider_name": ["A", "B", "C"],
        }
    )
    preds = cards.copy()
    preds["p_win"] = [0.5, 0.3, 0.2]
    merged = cards.merge(preds[["race_id", "lane_no", "p_win"]], on=["race_id", "lane_no"], how="inner")
    bets = build_bets(merged, budget=1000, policy="flat", ev_threshold=0.5)
    assert not bets.empty
    assert "combination" in bets.columns
