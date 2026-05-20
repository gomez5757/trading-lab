from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Iterable

import numpy as np
import pandas as pd


TRAIN_END_YEAR = 2004
VALIDATION_START_YEAR = 2005
VALIDATION_END_YEAR = 2015
LOCKED_START_YEAR = 2016


@dataclass(frozen=True)
class AnnualCandidate:
    specs: tuple[str, ...]
    min_votes: int = 1


@dataclass(frozen=True)
class AnnualBeamConfig:
    stage: int
    total_stages: int = 64
    seed_pool: int = 400
    beam_width: int = 32
    generations: int = 5
    mutations_per_parent: int = 10
    max_features: int = 4
    random_seed: int = 173_000


def build_annual_examples(
    daily: pd.DataFrame,
    *,
    start_year: int = 1980,
    end_year: int | None = None,
) -> pd.DataFrame:
    data = _normalize_daily(daily)
    end_year = end_year or int(data.index.max().year)
    features = _build_daily_feature_frame(data)
    rows: list[dict[str, object]] = []
    for target_year in range(start_year, end_year + 1):
        previous_year = data.loc[data.index.year == target_year - 1]
        target = data.loc[data.index.year == target_year]
        if previous_year.empty or target.empty:
            continue
        decision_date = previous_year.index[-1]
        feature_row = features.loc[decision_date]
        spy_return = float(target["close"].iloc[-1] / target["close"].iloc[0] - 1.0)
        row: dict[str, object] = {
            "target_year": int(target_year),
            "decision_date": decision_date,
            "spy_return_next_year": spy_return,
            "target_positive": bool(spy_return > 0.0),
        }
        for name, value in feature_row.items():
            row[name] = float(value) if pd.notna(value) else np.nan
        rows.append(row)
    examples = pd.DataFrame(rows)
    if not examples.empty:
        examples["decision_date"] = pd.to_datetime(examples["decision_date"])
        examples["target_positive"] = examples["target_positive"].astype(bool)
    return examples.replace([np.inf, -np.inf], np.nan)


def evaluate_annual_candidate(
    examples: pd.DataFrame,
    candidate: AnnualCandidate,
) -> dict[str, object]:
    predictions = _predict_positive(examples, candidate)
    train_mask = examples["target_year"].astype(int) <= TRAIN_END_YEAR
    validation_mask = (examples["target_year"].astype(int) >= VALIDATION_START_YEAR) & (
        examples["target_year"].astype(int) <= VALIDATION_END_YEAR
    )
    target = examples["target_positive"].astype(bool).to_numpy()
    returns = examples["spy_return_next_year"].astype(float).to_numpy()
    train = _period_metrics(predictions, target, returns, train_mask.to_numpy())
    validation = _period_metrics(predictions, target, returns, validation_mask.to_numpy())
    train_predicted_returns = _fit_predicted_returns(predictions[train_mask.to_numpy()], returns[train_mask.to_numpy()])
    train_mae = _mae_from_mapping(predictions[train_mask.to_numpy()], returns[train_mask.to_numpy()], train_predicted_returns)
    validation_mae = _mae_from_mapping(
        predictions[validation_mask.to_numpy()],
        returns[validation_mask.to_numpy()],
        train_predicted_returns,
    )
    row = {
        "candidate_id": _candidate_id(candidate),
        "specs": ";".join(candidate.specs),
        "min_votes": int(candidate.min_votes),
        "feature_count": len(candidate.specs),
        "train_hits": train["hits"],
        "train_total": train["total"],
        "train_accuracy": train["accuracy"],
        "train_negative_hits": train["negative_hits"],
        "train_negative_total": train["negative_total"],
        "validation_hits": validation["hits"],
        "validation_total": validation["total"],
        "validation_accuracy": validation["accuracy"],
        "validation_negative_hits": validation["negative_hits"],
        "validation_negative_total": validation["negative_total"],
        "train_return_mae": train_mae,
        "validation_return_mae": validation_mae,
        "always_positive_validation_accuracy": _always_positive_accuracy(examples, validation_mask),
        "annual_score": 0.0,
        "accepted": False,
        "rejection_reason": "",
        "locked_opened": False,
        "locked_hits": 0,
        "locked_total": 0,
    }
    row["annual_score"] = annual_score(row)
    row["rejection_reason"] = _rejection_reason(row)
    row["accepted"] = row["rejection_reason"] == ""
    return row


def annual_score(row: dict[str, object]) -> float:
    validation_accuracy = float(row.get("validation_accuracy", 0.0) or 0.0)
    train_accuracy = float(row.get("train_accuracy", 0.0) or 0.0)
    validation_negative_hits = int(row.get("validation_negative_hits", 0) or 0)
    train_negative_hits = int(row.get("train_negative_hits", 0) or 0)
    validation_mae = float(row.get("validation_return_mae", 1.0) or 1.0)
    complexity_penalty = max(0, int(row.get("feature_count", 1) or 1) - 2) * 0.5
    baseline = float(row.get("always_positive_validation_accuracy", 0.0) or 0.0)
    baseline_bonus = max(0.0, validation_accuracy - baseline) * 20.0
    return float(
        validation_accuracy * 100.0
        + train_accuracy * 30.0
        + validation_negative_hits * 8.0
        + train_negative_hits * 2.0
        + baseline_bonus
        - validation_mae * 10.0
        - complexity_penalty
    )


def run_annual_beam_search(
    examples: pd.DataFrame,
    config: AnnualBeamConfig,
) -> list[dict[str, object]]:
    catalog = _build_spec_catalog(examples)
    if not catalog:
        return []
    rng = np.random.default_rng(config.random_seed + config.stage)
    rows: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, ...], int]] = set()
    seeds = _seed_candidates(catalog, config=config, rng=rng)
    seed_rows = _evaluate_unique(examples, seeds, seen=seen)
    rows.extend(seed_rows)
    beam = _select_beam(seed_rows, config.beam_width)
    for _ in range(config.generations):
        children: list[AnnualCandidate] = []
        for parent in beam:
            candidate = _candidate_from_row(parent)
            for _ in range(config.mutations_per_parent):
                children.append(_mutate_candidate(candidate, catalog, config=config, rng=rng))
        child_rows = _evaluate_unique(examples, children, seen=seen)
        rows.extend(child_rows)
        beam = _select_beam([*beam, *child_rows], config.beam_width)
    return sorted(rows, key=lambda row: float(row["annual_score"]), reverse=True)


def _normalize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.set_index("timestamp")
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    if "close" not in data.columns:
        raise ValueError("annual prediction data needs a close column")
    return data


def _build_daily_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(data["close"], errors="coerce")
    out: dict[str, pd.Series] = {
        "spy_return_3m": close.pct_change(63),
        "spy_return_6m": close.pct_change(126),
        "spy_return_12m": close.pct_change(252),
        "spy_return_36m": close.pct_change(756),
        "spy_vol_3m": close.pct_change().rolling(63, min_periods=21).std(ddof=0),
        "spy_vol_12m": close.pct_change().rolling(252, min_periods=63).std(ddof=0),
        "spy_drawdown_12m": close / close.rolling(252, min_periods=63).max() - 1.0,
    }
    skip = {"open", "high", "low", "close", "volume"}
    for column in data.columns:
        if column in skip:
            continue
        series = pd.to_numeric(data[column], errors="coerce")
        if series.notna().sum() < 120:
            continue
        out[column] = series
        out[f"{column}_change_12m"] = series.diff(252)
        mean = series.rolling(756, min_periods=126).mean()
        std = series.rolling(756, min_periods=126).std(ddof=0).replace(0, np.nan)
        out[f"{column}_z_3y"] = (series - mean) / std
    return pd.DataFrame(out, index=data.index)


def _build_spec_catalog(examples: pd.DataFrame) -> list[str]:
    train = examples.loc[examples["target_year"].astype(int) <= TRAIN_END_YEAR]
    reserved = {"target_year", "decision_date", "spy_return_next_year", "target_positive"}
    specs: list[str] = []
    for column in examples.columns:
        if column in reserved:
            continue
        series = pd.to_numeric(train[column], errors="coerce").dropna()
        if len(series) < 8 or series.nunique() < 4:
            continue
        for quantile in (0.25, 0.50, 0.75):
            value = float(series.quantile(quantile))
            if not isfinite(value):
                continue
            specs.append(_encode_spec(column, value, 1))
            specs.append(_encode_spec(column, value, -1))
    return sorted(set(specs))


def _seed_candidates(
    catalog: list[str],
    *,
    config: AnnualBeamConfig,
    rng: np.random.Generator,
) -> list[AnnualCandidate]:
    candidates: list[AnnualCandidate] = []
    offset_catalog = [spec for index, spec in enumerate(catalog) if index % config.total_stages == config.stage]
    base = offset_catalog or catalog
    for spec in base[: config.seed_pool]:
        candidates.append(AnnualCandidate((spec,), min_votes=1))
    while len(candidates) < config.seed_pool:
        size = int(rng.integers(1, config.max_features + 1))
        specs = tuple(sorted(rng.choice(catalog, size=size, replace=False).tolist()))
        min_votes = int(rng.integers(1, size + 1))
        candidates.append(AnnualCandidate(specs, min_votes=min_votes))
    return candidates


def _evaluate_unique(
    examples: pd.DataFrame,
    candidates: Iterable[AnnualCandidate],
    *,
    seen: set[tuple[tuple[str, ...], int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        clean = AnnualCandidate(tuple(sorted(set(candidate.specs))), max(1, min(candidate.min_votes, len(candidate.specs))))
        key = (clean.specs, clean.min_votes)
        if key in seen or not clean.specs:
            continue
        seen.add(key)
        rows.append(evaluate_annual_candidate(examples, clean))
    return rows


def _select_beam(rows: list[dict[str, object]], width: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    signatures: set[str] = set()
    for row in sorted(rows, key=lambda item: float(item["annual_score"]), reverse=True):
        signature = _signature(str(row["specs"]))
        if signature in signatures:
            continue
        signatures.add(signature)
        selected.append(row)
        if len(selected) >= width:
            break
    return selected


def _mutate_candidate(
    candidate: AnnualCandidate,
    catalog: list[str],
    *,
    config: AnnualBeamConfig,
    rng: np.random.Generator,
) -> AnnualCandidate:
    specs = list(candidate.specs)
    action = str(rng.choice(["replace", "replace", "add", "remove", "votes"]))
    if action == "replace" and specs:
        specs[int(rng.integers(0, len(specs)))] = str(catalog[int(rng.integers(0, len(catalog)))])
    elif action == "add" and len(specs) < config.max_features:
        specs.append(str(catalog[int(rng.integers(0, len(catalog)))]))
    elif action == "remove" and len(specs) > 1:
        del specs[int(rng.integers(0, len(specs)))]
    min_votes = candidate.min_votes
    if action == "votes":
        min_votes = int(rng.integers(1, len(specs) + 1))
    return AnnualCandidate(tuple(sorted(set(specs)))[: config.max_features], min_votes=min_votes)


def _candidate_from_row(row: dict[str, object]) -> AnnualCandidate:
    specs = tuple(part for part in str(row["specs"]).split(";") if part)
    return AnnualCandidate(specs, int(row.get("min_votes", 1) or 1))


def _predict_positive(examples: pd.DataFrame, candidate: AnnualCandidate) -> np.ndarray:
    votes = np.zeros(len(examples), dtype=int)
    for spec in candidate.specs:
        feature, threshold, direction = _decode_spec(spec)
        values = pd.to_numeric(examples[feature], errors="coerce").to_numpy(dtype=float)
        if direction >= 0:
            vote = values > threshold
        else:
            vote = values < threshold
        votes += np.nan_to_num(vote, nan=False).astype(int)
    return votes >= max(1, min(candidate.min_votes, len(candidate.specs)))


def _period_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    if not mask.any():
        return {"hits": 0, "total": 0, "accuracy": 0.0, "negative_hits": 0, "negative_total": 0}
    period_predictions = predictions[mask]
    period_target = target[mask]
    hits = period_predictions == period_target
    negative = returns[mask] < 0
    return {
        "hits": int(hits.sum()),
        "total": int(mask.sum()),
        "accuracy": float(hits.mean()),
        "negative_hits": int((hits & negative).sum()),
        "negative_total": int(negative.sum()),
    }


def _fit_predicted_returns(predictions: np.ndarray, returns: np.ndarray) -> dict[bool, float]:
    default = float(np.nanmean(returns)) if len(returns) else 0.0
    mapping = {True: default, False: default}
    for value in (True, False):
        subset = returns[predictions == value]
        if len(subset):
            mapping[value] = float(np.nanmean(subset))
    return mapping


def _mae_from_mapping(
    predictions: np.ndarray,
    returns: np.ndarray,
    mapping: dict[bool, float],
) -> float:
    if len(returns) == 0:
        return 1.0
    predicted = np.array([mapping[bool(value)] for value in predictions], dtype=float)
    return float(np.nanmean(np.abs(predicted - returns)))


def _always_positive_accuracy(examples: pd.DataFrame, mask: pd.Series) -> float:
    if not mask.any():
        return 0.0
    return float(examples.loc[mask, "target_positive"].astype(bool).mean())


def _rejection_reason(row: dict[str, object]) -> str:
    if int(row["validation_total"]) < 8:
        return "too_few_validation_years"
    if float(row["validation_accuracy"]) < 0.70:
        return "validation_accuracy"
    if float(row["train_accuracy"]) < 0.70:
        return "train_accuracy"
    if float(row["validation_accuracy"]) <= float(row["always_positive_validation_accuracy"]):
        return "baseline"
    if int(row["validation_negative_total"]) and int(row["validation_negative_hits"]) < 1:
        return "misses_validation_stress"
    if int(row["feature_count"]) > 4:
        return "too_many_features"
    return ""


def _encode_spec(feature: str, threshold: float, direction: int) -> str:
    return f"{feature}|threshold|{threshold:.8g}|{int(direction)}"


def _decode_spec(spec: str) -> tuple[str, float, int]:
    parts = spec.split("|")
    if len(parts) != 4 or parts[1] != "threshold":
        raise ValueError(f"invalid annual spec: {spec}")
    return parts[0], float(parts[2]), int(float(parts[3]))


def _candidate_id(candidate: AnnualCandidate) -> str:
    raw = f"annual_beam|min_votes={candidate.min_votes}|specs={';'.join(candidate.specs)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"annual_beam_{digest}"


def _signature(specs: str) -> str:
    names = sorted({part.split("|", 1)[0] for part in specs.split(";") if part})
    return ";".join(names[:4])
