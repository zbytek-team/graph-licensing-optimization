from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy import stats

try:
    import scikit_posthocs as sp
except ImportError:  # pragma: no cover
    sp = None

LATEX_STYLE = {
    "figure.figsize": (6.2, 4.0),  # ~\textwidth, proporcje 3:2
    "axes.titlesize": 12,  # tytuł lekko większy niż label
    "axes.labelsize": 11,  # jak tekst główny
    "xtick.labelsize": 10,  # odrobinę mniejsze
    "ytick.labelsize": 10,
    "legend.fontsize": 9,  # żeby legenda nie dominowała
    "grid.alpha": 0.3,  # subtelna siatka
    "lines.linewidth": 1.5,  # dobrze widoczne w druku
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
    "savefig.dpi": 300,  # wysoka jakość (tylko gdybyś zapisywał do png)
}


BASE_COLOR_SEQUENCE = [
    "#1f77b4",  # niebieski
    "#ff7f0e",  # pomarańczowy
    "#2ca02c",  # zielony
    "#d62728",  # czerwony
    "#9467bd",  # fioletowy
    "#8c564b",  # brązowy
    "#e377c2",  # różowy
    "#7f7f7f",  # szary
    "#bcbd22",  # oliwkowy
    "#17becf",  # turkusowy
    "#aec7e8",  # jasnoniebieski
    "#ffbb78",  # jasnopomarańczowy
]

ALGORITHM_CANONICAL_ORDER = [
    "ILPSolver",
    "GreedyAlgorithm",
    "GeneticAlgorithm",
    "SimulatedAnnealing",
    "TabuSearch",
    "AntColonyOptimization",
    "DominatingSetAlgorithm",
    "RandomizedAlgorithm",
]

ALGORITHM_DISPLAY = {
    "AntColonyOptimization": "Algorytm mrówkowy",
    "DominatingSetAlgorithm": "Zbiór dominujący",
    "GeneticAlgorithm": "Algorytm genetyczny",
    "GreedyAlgorithm": "Algorytm zachłanny",
    "ILPSolver": "Solver ILP",
    "RandomizedAlgorithm": "Algorytm losowy",
    "SimulatedAnnealing": "Wyżarzanie symulowane",
    "TabuSearch": "Przeszukiwanie tabu",
}

REVERSE_ALGORITHM_DISPLAY = {label: key for key, label in ALGORITHM_DISPLAY.items()}

ALGORITHM_COLOR_MAP: dict[str, str] = {}
for idx, canonical in enumerate(ALGORITHM_CANONICAL_ORDER):
    color = BASE_COLOR_SEQUENCE[idx % len(BASE_COLOR_SEQUENCE)]
    display = ALGORITHM_DISPLAY.get(canonical, canonical)
    ALGORITHM_COLOR_MAP[display] = color
    ALGORITHM_COLOR_MAP[canonical] = color

POLISH_UNITS = [
    "zero",
    "jeden",
    "dwa",
    "trzy",
    "cztery",
    "pięć",
    "sześć",
    "siedem",
    "osiem",
    "dziewięć",
]

POLISH_TEENS = {
    10: "dziesięć",
    11: "jedenaście",
    12: "dwanaście",
    13: "trzynaście",
    14: "czternaście",
    15: "piętnaście",
    16: "szesnaście",
    17: "siedemnaście",
    18: "osiemnaście",
    19: "dziewiętnaście",
}

POLISH_TENS = [
    "",
    "",
    "dwadzieścia",
    "trzydzieści",
    "czterdzieści",
    "pięćdziesiąt",
    "sześćdziesiąt",
    "siedemdziesiąt",
    "osiemdziesiąt",
    "dziewięćdziesiąt",
]

POLISH_HUNDREDS = [
    "",
    "sto",
    "dwieście",
    "trzysta",
    "czterysta",
    "pięćset",
    "sześćset",
    "siedemset",
    "osiemset",
    "dziewięćset",
]

LICENSE_SYNONYMS = {
    "basic": "individual",
    "cohort": "group",
    "duo": "group",
    "family": "group",
    "group": "group",
    "groups": "group",
    "individual": "individual",
    "premium": "group",
    "solo": "individual",
    "singles": "individual",
    "standard": "group",
    "team": "group",
}


def apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(LATEX_STYLE)
    plt.rcParams["axes.prop_cycle"] = cycler(color=BASE_COLOR_SEQUENCE)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def algorithm_display_name(name: object) -> str:
    if name is None:
        return "nieznany"
    if isinstance(name, float) and np.isnan(name):
        return "nieznany"
    text = str(name)
    return ALGORITHM_DISPLAY.get(text, text)


def apply_algorithm_labels(
    df: pd.DataFrame, column: str = "algorithm", new_column: str | None = None
) -> pd.DataFrame:
    result = df.copy()
    target = new_column or column
    result[target] = result[column].map(algorithm_display_name)
    return result


def _color_for_label(label: str) -> str:
    color = ALGORITHM_COLOR_MAP.get(label)
    if color:
        return color
    canonical = REVERSE_ALGORITHM_DISPLAY.get(label)
    if canonical:
        color = ALGORITHM_COLOR_MAP.get(canonical)
        if color:
            ALGORITHM_COLOR_MAP[label] = color
            return color
    idx = int(hashlib.sha1(label.encode("utf-8")).hexdigest(), 16) % len(
        BASE_COLOR_SEQUENCE
    )
    color = BASE_COLOR_SEQUENCE[idx]
    ALGORITHM_COLOR_MAP[label] = color
    return color


def algorithm_palette(labels: Iterable[str]) -> dict[str, str]:
    palette: dict[str, str] = {}
    for label in labels:
        palette[label] = _color_for_label(str(label))
    return palette


def _number_to_words_under_thousand(value: int) -> str:
    if value == 0:
        return POLISH_UNITS[0]
    parts: list[str] = []
    hundreds = value // 100
    if hundreds:
        parts.append(POLISH_HUNDREDS[hundreds])
    remainder = value % 100
    if remainder:
        if remainder < 10:
            parts.append(POLISH_UNITS[remainder])
        elif remainder < 20:
            parts.append(POLISH_TEENS[remainder])
        else:
            tens = remainder // 10
            parts.append(POLISH_TENS[tens])
            units = remainder % 10
            if units:
                parts.append(POLISH_UNITS[units])
    return " ".join(part for part in parts if part)


def number_to_polish_words(value: int) -> str:
    if not isinstance(value, (int, np.integer)):
        raise TypeError("value must be an integer")
    if value < 0:
        raise ValueError("value must be non-negative")
    if value < 1000:
        return _number_to_words_under_thousand(int(value))
    if value < 1_000_000:
        thousands = value // 1000
        remainder = value % 1000
        parts: list[str] = []
        if thousands == 1:
            parts.append("tysiąc")
        else:
            thousands_words = _number_to_words_under_thousand(int(thousands))
            if thousands % 10 in {2, 3, 4} and thousands % 100 not in {12, 13, 14}:
                suffix = "tysiące"
            else:
                suffix = "tysięcy"
            parts.append(f"{thousands_words} {suffix}")
        if remainder:
            parts.append(_number_to_words_under_thousand(int(remainder)))
        return " ".join(parts)
    return str(int(value))


def _safe_parse_struct(value: object) -> Mapping[str, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, Mapping):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return {}
    if isinstance(parsed, Mapping):
        return parsed
    return {}


def load_dataset(path: Path | str, parse_json: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if parse_json:
        json_cols = [c for c in df.columns if c.endswith("_json")]
        for column in json_cols:
            df[column] = df[column].apply(_safe_parse_struct)
    if "time_ms" in df.columns and "time_s" not in df.columns:
        df = df.assign(time_s=df["time_ms"] / 1000.0)
    return df


def expand_license_counts(
    df: pd.DataFrame, column: str = "license_counts_json", prefix: str = "license"
) -> pd.DataFrame:
    if column not in df.columns:
        return df
    result = df.copy()
    parsed = result[column].apply(_safe_parse_struct)

    def _accumulate(mapping: Mapping[str, float]) -> dict:
        counts = {"group": 0.0, "individual": 0.0, "other": 0.0}
        for key, value in mapping.items():
            if value is None:
                continue
            canon = LICENSE_SYNONYMS.get(str(key).lower())
            if canon == "group":
                counts["group"] += float(value)
            elif canon == "individual":
                counts["individual"] += float(value)
            else:
                counts["other"] += float(value)
        counts["total"] = counts["group"] + counts["individual"] + counts["other"]
        return counts

    aggregated = parsed.apply(_accumulate)
    for key in ["group", "individual", "other", "total"]:
        result[f"{prefix}_{key}"] = aggregated.apply(lambda x: x.get(key, 0.0))
    return result


def _resolve_unit_costs(
    df: pd.DataFrame,
    total_col: str = "total_cost",
    individual_col: str = "license_individual",
    group_col: str = "license_group",
) -> dict[str, dict[str, float]]:
    if (
        total_col not in df.columns
        or individual_col not in df.columns
        or group_col not in df.columns
    ):
        return {}

    unit_costs: dict[str, dict[str, float]] = {}
    grouped = df.groupby("license_config", dropna=True)
    for license_config, frame in grouped:
        sub = frame[[total_col, individual_col, group_col]].copy()
        sub = sub.dropna(subset=[total_col])
        if sub.empty:
            continue
        A = sub[[individual_col, group_col]].to_numpy(dtype=float)
        b = sub[total_col].to_numpy(dtype=float)
        valid = np.isfinite(b) & np.isfinite(A).all(axis=1) & (A.sum(axis=1) > 0)
        if not valid.any():
            continue
        A = A[valid]
        b = b[valid]
        solo_cost = np.nan
        group_cost = np.nan
        try:
            coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            if len(coef) >= 1:
                solo_cost = float(coef[0])
            if len(coef) >= 2:
                group_cost = float(coef[1])
        except np.linalg.LinAlgError:
            solo_cost = np.nan
            group_cost = np.nan

        if not np.isfinite(solo_cost) or solo_cost <= 0:
            ind_mask = A[:, 0] > 0
            if ind_mask.any():
                solo_candidates = b[ind_mask] / A[ind_mask, 0]
                solo_candidates = solo_candidates[
                    np.isfinite(solo_candidates) & (solo_candidates > 0)
                ]
                if solo_candidates.size:
                    solo_cost = float(np.mean(solo_candidates))

        if not np.isfinite(group_cost) or group_cost <= 0:
            grp_mask = A[:, 1] > 0
            if grp_mask.any():
                group_candidates = b[grp_mask] / A[grp_mask, 1]
                group_candidates = group_candidates[
                    np.isfinite(group_candidates) & (group_candidates > 0)
                ]
                if group_candidates.size:
                    group_cost = float(np.mean(group_candidates))

        if not np.isfinite(solo_cost) or solo_cost <= 0:
            continue

        unit_costs[str(license_config)] = {"solo": solo_cost}
        if np.isfinite(group_cost) and group_cost > 0:
            unit_costs[str(license_config)]["group"] = group_cost
    return unit_costs


def normalize_cost_columns(
    df: pd.DataFrame,
    cost_columns: Sequence[str] = ("total_cost", "cost_per_node"),
    unit_costs: Mapping[str, Mapping[str, float]] | None = None,
    attach_group_multiplier: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    result = df.copy()
    if unit_costs is None:
        unit_costs = _resolve_unit_costs(result)
    if not unit_costs:
        return result, {}

    solo_map = {
        cfg: vals.get("solo") for cfg, vals in unit_costs.items() if vals.get("solo")
    }
    if not solo_map:
        return result, {}

    scale_series = result["license_config"].map(solo_map).astype(float)
    scale_series = scale_series.where((scale_series > 0) & np.isfinite(scale_series))
    scale_values = scale_series.to_numpy()
    mask = np.isfinite(scale_values) & (scale_values > 0)

    for column in cost_columns:
        if column not in result.columns:
            continue
        series = result[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        values = series.to_numpy(dtype=float, copy=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(mask, values / scale_values, values)
        result[column] = values

    if attach_group_multiplier:
        multiplier_map = {}
        for cfg, vals in unit_costs.items():
            solo = vals.get("solo")
            group_cost = vals.get("group")
            if solo and group_cost and solo > 0:
                multiplier_map[cfg] = group_cost / solo
        if multiplier_map:
            result["license_group_multiplier"] = result["license_config"].map(
                multiplier_map
            )

    return result, dict(unit_costs)


def describe_numeric(
    df: pd.DataFrame, value_cols: Sequence[str], group_cols: Sequence[str] | None = None
) -> pd.DataFrame:
    if group_cols:
        grouped = df.groupby(list(group_cols), dropna=False)
        records: list[dict] = []
        for keys, frame in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            for value_col in value_cols:
                series = frame[value_col].dropna()
                if series.empty:
                    continue
                record = {col: key for col, key in zip(group_cols, keys)}
                record.update(
                    {
                        "metric": value_col,
                        "count": int(series.count()),
                        "mean": float(series.mean()),
                        "std": float(series.std(ddof=1)) if series.count() > 1 else 0.0,
                        "min": float(series.min()),
                        "max": float(series.max()),
                    }
                )
                records.append(record)
        return pd.DataFrame(records)
    series_records = []
    for value_col in value_cols:
        series = df[value_col].dropna()
        if series.empty:
            continue
        series_records.append(
            {
                "metric": value_col,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if series.count() > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(series_records)


def compute_pareto_front(
    df: pd.DataFrame, cost_col: str, time_col: str, objective: str = "min"
) -> pd.DataFrame:
    subset = df.dropna(subset=[cost_col, time_col]).copy()
    if subset.empty:
        return subset
    ascending = objective == "min"
    subset = subset.sort_values([cost_col, time_col], ascending=ascending)
    pareto_mask = []
    best_time = np.inf if ascending else -np.inf
    comparator = (
        (lambda current, best: current <= best)
        if ascending
        else (lambda current, best: current >= best)
    )
    for _, row in subset.iterrows():
        current = row[time_col]
        if comparator(current, best_time):
            pareto_mask.append(True)
            best_time = current
        else:
            pareto_mask.append(False)
    return subset.loc[pareto_mask]


@dataclass
class FriedmanResult:
    statistic: float
    pvalue: float
    mean_ranks: pd.Series
    nemenyi: pd.DataFrame | None


def run_friedman_nemenyi(pivot: pd.DataFrame) -> FriedmanResult | None:
    pivot = pivot.dropna()
    if pivot.empty or pivot.shape[1] < 3:
        return None
    samples = [pivot[col].to_numpy() for col in pivot.columns]
    statistic, pvalue = stats.friedmanchisquare(*samples)
    ranks = pivot.rank(axis=1, ascending=True).mean(axis=0)
    nemenyi = None
    if sp is not None:
        nemenyi = sp.posthoc_nemenyi_friedman(pivot.to_numpy())
        nemenyi.index = pivot.columns
        nemenyi.columns = pivot.columns
    return FriedmanResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        mean_ranks=ranks,
        nemenyi=nemenyi,
    )


def pivot_complete_blocks(
    df: pd.DataFrame, index_cols: Sequence[str], column_col: str, value_col: str
) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=list(index_cols), columns=column_col, values=value_col, aggfunc="mean"
    )
    pivot = pivot.dropna()
    return pivot


def save_table(df: pd.DataFrame, path: Path | str) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=True)


def write_text(path: Path | str, lines: Iterable[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
