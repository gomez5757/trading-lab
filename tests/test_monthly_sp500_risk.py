from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from trading_lab.monthly_risk import (
    MonthlyRiskCandidate,
    MonthlyRiskSearchConfig,
    build_monthly_examples,
    evaluate_monthly_candidate,
    merge_monthly_risk_leaderboards,
    run_monthly_risk_search,
    write_monthly_risk_outputs,
)


def _daily_sample() -> pd.DataFrame:
    dates = pd.bdate_range("1979-12-03", "2021-03-31")
    index = range(len(dates))
    close = [100.0 + value * 0.08 + (value % 23) * 0.15 for value in index]
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": [value * 1.01 for value in close],
            "low": [value * 0.99 for value in close],
            "close": close,
            "volume": [1000] * len(dates),
            "credit_spread": [2.0 + (value % 11) * 0.05 for value in index],
            "yield_curve": [1.0 - (value % 7) * 0.03 for value in index],
            "cape": [15.0 + (value % 17) * 0.1 for value in index],
            "earnings_yield": [0.05 + (value % 13) * 0.001 for value in index],
            "dividend_yield": [0.02 + (value % 19) * 0.0005 for value in index],
        }
    )
    return data.set_index("timestamp")


def test_build_monthly_examples_uses_previous_month_close_only() -> None:
    data = _daily_sample()

    examples = build_monthly_examples(data, start_year=1980, end_year=1980)

    first = examples.iloc[0]
    assert int(first["target_year"]) == 1980
    assert int(first["target_month"]) == 1
    assert first["decision_date"].year == 1979
    assert first["decision_date"].month == 12
    december = data.loc[(data.index.year == 1979) & (data.index.month == 12), "close"].iloc[-1]
    january = data.loc[(data.index.year == 1980) & (data.index.month == 1), "close"].iloc[-1]
    assert first["spy_return_next_month"] == january / december - 1.0


def test_monthly_candidate_has_continuous_bounded_exposure() -> None:
    examples = build_monthly_examples(_daily_sample(), start_year=1980, end_year=1980)
    candidate = MonthlyRiskCandidate(("month_number|threshold|0|1|1",), intercept=0.0, scale=0.37)

    row, positions, _ = evaluate_monthly_candidate(examples, candidate)

    assert row["min_exposure"] >= -1.0
    assert row["max_exposure"] <= 1.0
    assert any(math.isclose(value, 0.37, abs_tol=1e-9) for value in positions["exposure"])
    assert row["locked_opened"] is False


def test_monthly_cash_return_is_exactly_zero() -> None:
    examples = build_monthly_examples(_daily_sample(), start_year=1980, end_year=1980)
    candidate = MonthlyRiskCandidate(("month_number|threshold|0|1|1",), intercept=0.0, scale=0.0)

    _, positions, _ = evaluate_monthly_candidate(examples, candidate)

    assert positions["strategy_return"].abs().max() == 0.0


def test_monthly_candidate_rejects_train_year_below_10pct() -> None:
    examples = pd.DataFrame(
        {
            "decision_date": pd.date_range("1999-12-31", periods=24, freq="ME"),
            "target_year": [2000] * 12 + [2001] * 12,
            "target_month": list(range(1, 13)) * 2,
            "spy_return_next_month": [0.001] * 24,
            "month_number": list(range(1, 13)) * 2,
        }
    )
    candidate = MonthlyRiskCandidate(("month_number|threshold|0|1|1",), intercept=1.0, scale=0.0)

    row, _, year_by_year = evaluate_monthly_candidate(examples, candidate)

    assert row["accepted"] is False
    assert row["rejection_reason"] == "train_year_below_10pct"
    assert year_by_year["strategy_return"].max() < 0.10


def test_monthly_score_uses_train_and_reports_validation_only() -> None:
    base = {
        "decision_date": pd.date_range("2006-01-31", periods=36, freq="ME"),
        "target_year": [2006] * 12 + [2007] * 12 + [2008] * 12,
        "target_month": list(range(1, 13)) * 3,
        "spy_return_next_month": [-0.02] * 24 + [0.03] * 12,
        "month_number": list(range(1, 13)) * 3,
    }
    examples = pd.DataFrame(base)
    candidate = MonthlyRiskCandidate(("month_number|threshold|0|1|1",), intercept=1.0, scale=0.0)

    row, _, _ = evaluate_monthly_candidate(examples, candidate)

    assert row["accepted"] is False
    assert row["validation_years_ge_10pct"] == 1
    assert row["rejection_reason"] == "train_year_below_10pct"


def test_monthly_locked_is_closed_and_not_scored() -> None:
    examples = build_monthly_examples(_daily_sample(), start_year=2006, end_year=2021)
    candidate = MonthlyRiskCandidate(("month_number|threshold|0|1|1",), intercept=0.5, scale=0.0)

    row, positions, year_by_year = evaluate_monthly_candidate(examples, candidate)

    assert row["locked_opened"] is False
    assert row["locked_months"] == 0
    assert positions["period"].isin(["locked"]).sum() == 0
    assert year_by_year["period"].isin(["locked"]).sum() == 0


def test_monthly_search_returns_required_fields() -> None:
    examples = build_monthly_examples(_daily_sample(), start_year=1981, end_year=2010)
    rows = run_monthly_risk_search(
        examples,
        MonthlyRiskSearchConfig(stage=0, total_stages=2, seed_pool=12, beam_width=4, generations=1),
    )

    required = {
        "candidate_id",
        "rules",
        "features",
        "monthly_exposure_formula",
        "train_years_ge_10pct",
        "validation_years_ge_10pct",
        "train_cagr",
        "validation_cagr",
        "train_mdd",
        "validation_mdd",
        "average_exposure",
        "min_exposure",
        "max_exposure",
        "months_long",
        "months_cash_like",
        "months_short",
        "locked_opened",
    }
    assert rows
    assert required.issubset(rows[0])
    assert rows[0]["monthly_risk_score"] >= rows[-1]["monthly_risk_score"]


def test_monthly_output_files_are_written(tmp_path: Path) -> None:
    examples = build_monthly_examples(_daily_sample(), start_year=1981, end_year=2010)
    rows = run_monthly_risk_search(
        examples,
        MonthlyRiskSearchConfig(stage=0, total_stages=2, seed_pool=8, beam_width=3, generations=1),
    )

    write_monthly_risk_outputs(rows, examples, tmp_path, stage=0)

    assert (tmp_path / "monthly_risk_leaderboard.csv").exists()
    assert (tmp_path / "monthly_risk_candidates.jsonl").exists()
    assert (tmp_path / "monthly_risk_summary.json").exists()
    assert (tmp_path / "monthly_risk_year_by_year.csv").exists()
    assert (tmp_path / "monthly_risk_monthly_positions.csv").exists()


def test_monthly_merge_counts_only_train_and_validation_10pct_verified(tmp_path: Path) -> None:
    stage = tmp_path / "stage.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "ok",
                "monthly_risk_score": 10.0,
                "accepted": True,
                "train_years_ge_10pct": 28,
                "train_years_total": 28,
                "validation_years_ge_10pct": 12,
                "validation_years_total": 12,
                "locked_opened": False,
            },
            {
                "candidate_id": "fails_validation",
                "monthly_risk_score": 9.0,
                "accepted": True,
                "train_years_ge_10pct": 28,
                "train_years_total": 28,
                "validation_years_ge_10pct": 11,
                "validation_years_total": 12,
                "locked_opened": False,
            },
        ]
    ).to_csv(stage, index=False)

    summary = merge_monthly_risk_leaderboards([stage], tmp_path / "merged")

    verified = pd.read_csv(tmp_path / "merged" / "monthly_risk_train_validation_10pct.csv")
    assert summary["unique_verified_train_validation_10pct"] == 1
    assert verified["candidate_id"].tolist() == ["ok"]
