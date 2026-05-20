from __future__ import annotations

import pandas as pd

from trading_lab.annual_prediction import (
    AnnualBeamConfig,
    AnnualCandidate,
    build_annual_examples,
    evaluate_annual_candidate,
    run_annual_beam_search,
)


def _daily_sample() -> pd.DataFrame:
    dates = pd.bdate_range("1979-12-20", "2020-01-10")
    index = range(len(dates))
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100 + value for value in index],
            "high": [101 + value for value in index],
            "low": [99 + value for value in index],
            "close": [100 + value for value in index],
            "volume": [1000] * len(dates),
            "credit_spread": [2.0 + (value % 7) * 0.1 for value in index],
            "yield_curve": [1.0 - (value % 5) * 0.05 for value in index],
        }
    )
    return data.set_index("timestamp")


def test_build_annual_examples_uses_previous_year_close_only() -> None:
    data = _daily_sample()

    examples = build_annual_examples(data, start_year=1981, end_year=1983)

    assert examples["target_year"].tolist() == [1981, 1982, 1983]
    assert examples["decision_date"].dt.year.tolist() == [1980, 1981, 1982]
    assert "spy_return_next_year" in examples.columns
    assert examples["target_positive"].dtype == bool


def test_annual_candidate_evaluates_train_and_validation_without_locked() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=2020)
    candidate = AnnualCandidate(
        specs=("spy_return_12m|threshold|0|1", "credit_spread_z_3y|threshold|0|-1"),
        min_votes=1,
    )

    row = evaluate_annual_candidate(examples, candidate)

    assert row["locked_opened"] is False
    assert row["train_total"] > 0
    assert row["validation_total"] > 0
    assert row["locked_hits"] == 0
    assert row["locked_total"] == 0
    assert "candidate_id" in row


def test_annual_beam_search_returns_ranked_candidates() -> None:
    examples = build_annual_examples(_daily_sample(), start_year=1981, end_year=2020)
    config = AnnualBeamConfig(
        stage=0,
        total_stages=2,
        seed_pool=12,
        beam_width=4,
        generations=2,
        mutations_per_parent=3,
    )

    rows = run_annual_beam_search(examples, config)

    assert rows
    assert rows[0]["annual_score"] >= rows[-1]["annual_score"]
    assert all(row["locked_opened"] is False for row in rows)
    assert all(int(row["feature_count"]) <= 4 for row in rows)
