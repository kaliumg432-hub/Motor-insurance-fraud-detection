"""Shared helpers for the audit scripts: file paths, data loading, metrics and chart style."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
KAGGLE_CSV = HERE / "data_set_2_kaggle.csv"
KENYA_CSV = HERE / "Kenyan_Motor_Insurance_2023_2024.csv"
SEED = 42

# Chart style (light surface, colour-blind-checked slots)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a8a69f"


def need(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(
            f"\nMissing data file: {path.name}\n"
            f"Put it in {HERE} (see the README section 'Data') and run again.\n"
        )
    return path


def load_kaggle() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(need(KAGGLE_CSV))
    y = (df.pop("FraudFound") == "Yes").astype(int).to_numpy()
    return df, y


def as_categories(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).astype("category")
    return X


def read_rds(name: str) -> np.ndarray:
    import pyreadr

    obj = list(pyreadr.read_r(str(HERE / name)).values())[0]
    return obj if isinstance(obj, pd.DataFrame) and obj.shape[1] > 1 else np.asarray(obj).ravel()


def scores(y, p, threshold=None) -> dict:
    """AUC-ROC, AUC-PR and best-threshold F1 (threshold picked on the same data,
    so the F1 here is optimistic, which only strengthens a 'no signal' result)."""
    out = {"AUC_ROC": roc_auc_score(y, p), "AUC_PR": average_precision_score(y, p), "base_rate": float(np.mean(y))}
    if threshold is None:
        grid = np.quantile(p, np.linspace(0.01, 0.99, 99))
        out["F1"] = max(f1_score(y, p >= t) for t in grid)
    else:
        out["F1"] = f1_score(y, p >= threshold)
    return out


def bar_chart(path, title, subtitle, labels, series, xlabel="AUC-ROC  (0.5 = coin flip, 1.0 = perfect)", xmax=1.05, ref=0.5):
    """Horizontal grouped bars with direct value labels. series = [(name, colour, values)]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, k = len(labels), len(series)
    h = 0.8 / k
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * n * k + 1.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y0 = np.arange(n)[::-1]
    for j, (name, colour, vals) in enumerate(series):
        ys = y0 + 0.4 - h * (j + 0.5)
        cols = colour if isinstance(colour, list) else [colour] * n
        for yy, v, c in zip(ys, vals, cols):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                ax.text(0.01, yy, "no output file", va="center", fontsize=8.5, color=INK2, style="italic")
                continue
            ax.barh(yy, v, height=h - 0.04, color=c, edgecolor=SURFACE, linewidth=1)
            ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=9, color=INK)
        if k > 1:
            ax.barh(np.nan, 0, color=cols[0], label=name)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.text(ref, n - 0.45, " coin flip", fontsize=8, color=INK2, va="bottom")
    ax.set_yticks(y0)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.3)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    H = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / H, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.45 / H, subtitle, ha="left", va="top", fontsize=9, color=INK2)
    if k > 1:
        ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / H))
    fig.savefig(HERE / path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {path}")
