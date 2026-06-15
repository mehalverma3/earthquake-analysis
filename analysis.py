"""
Seismicity analysis.

Implements the Gutenberg-Richter frequency-magnitude analysis:

    log10(N) = a - b * M

where N is the cumulative number of events with magnitude >= M. The b-value
(slope) is typically near 1.0 for tectonic regions; a is the productivity.

Provides two independent b-value estimators (least-squares fit and the Aki 1965
maximum-likelihood estimator), a magnitude-of-completeness estimate via the
maximum-curvature method, and simple spatial clustering with DBSCAN-style
great-circle distances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOG10_E = np.log10(np.e)  # 0.4342944819...


def frequency_magnitude_distribution(
    magnitudes: np.ndarray | pd.Series, bin_width: float = 0.1
) -> pd.DataFrame:
    """
    Build the cumulative and non-cumulative frequency-magnitude distribution.

    Returns a DataFrame with columns:
        mag         - left edge of each magnitude bin
        incremental - count of events in that bin
        cumulative  - count of events with magnitude >= mag
    """
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[~np.isnan(mags)]
    if mags.size == 0:
        return pd.DataFrame(columns=["mag", "incremental", "cumulative"])

    # Build edges as integer multiples of bin_width to avoid floating-point
    # drift from np.arange, and extend one extra bin past the max so the
    # largest event is always captured (np.histogram's final bin is closed).
    eps = 1e-9
    n_lo = int(np.floor(mags.min() / bin_width + eps))
    n_hi = int(np.ceil(mags.max() / bin_width - eps))
    edges = np.arange(n_lo, n_hi + 2) * bin_width
    incremental, _ = np.histogram(mags, bins=edges)

    centers = edges[:-1]
    # Cumulative count of events with magnitude >= each bin's left edge.
    cumulative = incremental[::-1].cumsum()[::-1]

    return pd.DataFrame(
        {"mag": np.round(centers, 4), "incremental": incremental, "cumulative": cumulative}
    )


def magnitude_of_completeness(
    magnitudes: np.ndarray | pd.Series, bin_width: float = 0.1
) -> float:
    """
    Estimate the magnitude of completeness (Mc) with the maximum-curvature method.

    Mc is taken as the magnitude bin with the most events (the peak of the
    non-cumulative distribution), which marks where the catalog starts losing
    small events. A common refinement adds a +0.2 correction; we return the raw
    maximum-curvature value and let the caller adjust if desired.
    """
    fmd = frequency_magnitude_distribution(magnitudes, bin_width)
    if fmd.empty:
        return float("nan")
    return float(fmd.loc[fmd["incremental"].idxmax(), "mag"])


def b_value_mle(
    magnitudes: np.ndarray | pd.Series,
    mc: float | None = None,
    bin_width: float = 0.1,
) -> dict:
    """
    Aki (1965) maximum-likelihood b-value estimate.

        b = log10(e) / (mean(M) - (Mc - dM/2))

    Only events with magnitude >= Mc are used. Returns the b-value, its
    standard error (Shi & Bolt 1982), the a-value, and Mc.
    """
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[~np.isnan(mags)]
    if mc is None:
        mc = magnitude_of_completeness(mags, bin_width)

    sample = mags[mags >= mc - bin_width / 2]
    n = sample.size
    if n < 2:
        return {"b": float("nan"), "b_err": float("nan"), "a": float("nan"), "mc": mc, "n": n}

    mean_mag = sample.mean()
    b = LOG10_E / (mean_mag - (mc - bin_width / 2))

    # Shi & Bolt (1982) uncertainty on b.
    var = np.sum((sample - mean_mag) ** 2) / (n * (n - 1))
    b_err = 2.30 * b**2 * np.sqrt(var)

    # a-value from log10(N) = a - b*Mc  =>  a = log10(N) + b*Mc
    a = np.log10(n) + b * mc

    return {"b": b, "b_err": b_err, "a": a, "mc": mc, "n": n}


def b_value_lstsq(
    magnitudes: np.ndarray | pd.Series,
    mc: float | None = None,
    bin_width: float = 0.1,
) -> dict:
    """
    Least-squares b-value: linear fit of log10(cumulative count) vs magnitude,
    restricted to magnitudes >= Mc. Returns slope (b), intercept (a), and R^2.
    """
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[~np.isnan(mags)]
    if mc is None:
        mc = magnitude_of_completeness(mags, bin_width)

    fmd = frequency_magnitude_distribution(mags, bin_width)
    fit_region = fmd[(fmd["mag"] >= mc) & (fmd["cumulative"] > 0)]
    if len(fit_region) < 2:
        return {"b": float("nan"), "a": float("nan"), "r_squared": float("nan"), "mc": mc}

    x = fit_region["mag"].to_numpy()
    y = np.log10(fit_region["cumulative"].to_numpy())

    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"b": -slope, "a": intercept, "r_squared": r_squared, "mc": mc}


def summary_stats(df: pd.DataFrame) -> dict:
    """Headline numbers for a catalog: counts, magnitude/depth ranges, spans."""
    stats = {
        "n_events": int(len(df)),
        "mag_min": float(df["magnitude"].min()),
        "mag_max": float(df["magnitude"].max()),
        "mag_mean": float(df["magnitude"].mean()),
        "depth_min_km": float(df["depth_km"].min()),
        "depth_max_km": float(df["depth_km"].max()),
        "depth_median_km": float(df["depth_km"].median()),
    }
    if "datetime" in df.columns and len(df):
        stats["time_start"] = str(df["datetime"].min())
        stats["time_end"] = str(df["datetime"].max())
    return stats


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between arrays of points."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def cluster_events(
    df: pd.DataFrame, eps_km: float = 100.0, min_samples: int = 5
) -> pd.DataFrame:
    """
    Label spatial clusters with DBSCAN using great-circle distance.

    Adds a ``cluster`` column: -1 means noise (not part of any cluster).
    Falls back gracefully with a clear message if scikit-learn isn't installed.
    """
    df = df.copy()
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("scikit-learn not installed; skipping clustering. "
              "Install with: pip install scikit-learn")
        df["cluster"] = -1
        return df

    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    eps_rad = eps_km / 6371.0
    labels = DBSCAN(
        eps=eps_rad, min_samples=min_samples, metric="haversine"
    ).fit_predict(coords)
    df["cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} spatial clusters "
          f"({(labels == -1).sum()} events flagged as noise).")
    return df
