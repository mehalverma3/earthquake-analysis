#!/usr/bin/env python3
"""
Earthquake Pattern Analysis — command-line pipeline.

Examples
--------
    # Global M>=4.5 over the last 90 days
    python main.py --days 90 --min-mag 4.5

    # A fixed window, lower threshold (more events, slower)
    python main.py --start 2024-01-01 --end 2024-12-31 --min-mag 2.5

    # Re-use cached data and skip the slow interactive maps
    python main.py --days 30 --no-refresh --no-maps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import analysis, data_fetch, mapping, visualize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="USGS earthquake pattern analysis.")
    p.add_argument("--start", help="Start date (ISO, e.g. 2024-01-01).")
    p.add_argument("--end", help="End date (ISO).")
    p.add_argument("--days", type=int, default=90,
                   help="Look back this many days (ignored if --start/--end given).")
    p.add_argument("--min-mag", type=float, default=4.5, help="Minimum magnitude.")
    p.add_argument("--bin-width", type=float, default=0.1, help="Magnitude bin width.")
    p.add_argument("--outdir", default="outputs", help="Output directory.")
    p.add_argument("--cluster", action="store_true",
                   help="Run DBSCAN spatial clustering (needs scikit-learn).")
    p.add_argument("--no-maps", action="store_true", help="Skip Folium maps.")
    p.add_argument("--refresh", dest="refresh", action="store_true",
                   help="Force re-fetch, ignoring cache.")
    p.add_argument("--no-refresh", dest="refresh", action="store_false",
                   help="Use cached data if present (default).")
    p.set_defaults(refresh=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        start, end = data_fetch.date_range_days(args.days)

    df = data_fetch.fetch_earthquakes(
        start, end, min_magnitude=args.min_mag,
        cache_path=outdir / "earthquakes_cache.csv", refresh=args.refresh,
    )
    if df.empty:
        print("No data to analyze. Try a wider window or lower --min-mag.")
        return

    # ----- Analysis -----
    stats = analysis.summary_stats(df)
    mc = analysis.magnitude_of_completeness(df["magnitude"], args.bin_width)
    mle = analysis.b_value_mle(df["magnitude"], mc=mc, bin_width=args.bin_width)
    lsq = analysis.b_value_lstsq(df["magnitude"], mc=mc, bin_width=args.bin_width)

    report = {
        "query": {"start": start, "end": end, "min_magnitude": args.min_mag},
        "summary": stats,
        "gutenberg_richter": {
            "magnitude_of_completeness": mc,
            "b_value_mle": mle,
            "b_value_lstsq": lsq,
        },
    }

    if args.cluster:
        df = analysis.cluster_events(df)
        report["clustering"] = {
            "n_clusters": int(df["cluster"].nunique() - (1 if -1 in df["cluster"].values else 0)),
            "n_noise": int((df["cluster"] == -1).sum()),
        }

    (outdir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print("\n=== Gutenberg-Richter ===")
    print(f"  Mc          = {mc:.2f}")
    print(f"  b (MLE)     = {mle['b']:.3f} ± {mle['b_err']:.3f}  (n={mle['n']:,})")
    print(f"  b (LSQ)     = {lsq['b']:.3f}  (R²={lsq['r_squared']:.3f})")
    print(f"  a-value     = {mle['a']:.2f}")

    # ----- Static figures -----
    print("\nGenerating figures...")
    visualize.plot_gutenberg_richter(df, outdir, args.bin_width)
    visualize.plot_magnitude_histogram(df, outdir)
    visualize.plot_depth_distribution(df, outdir)
    visualize.plot_temporal(df, outdir)
    visualize.plot_global_scatter(df, outdir)

    # ----- Interactive maps -----
    if not args.no_maps:
        print("\nGenerating interactive maps...")
        mapping.marker_cluster_map(df, outdir / "earthquake_map.html")
        mapping.heatmap(df, outdir / "earthquake_heatmap.html")

    print(f"\nDone. Outputs in {outdir.resolve()}")


if __name__ == "__main__":
    main()
