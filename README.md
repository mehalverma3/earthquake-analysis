# Earthquake Pattern Analysis

Pull live global seismic data from the **USGS** earthquake API and analyze it with
**Pandas**, **NumPy**, and **Matplotlib**. The project reproduces the
**Gutenberg-Richter** power-law relationship on real-world data and maps geographic
clustering with **Folium**.

![Gutenberg-Richter distribution](assets/gutenberg_richter.png)

> *Example output from a synthetic catalog generated with a known b-value of 0.95;
> the maximum-likelihood estimator recovered 0.96 ± 0.01. Run the pipeline to
> regenerate every figure on live USGS data.*

## The science

**Gutenberg-Richter law.** The frequency of earthquakes scales with magnitude as a
power law:

```
log10(N) = a - b * M
```

where `N` is the cumulative number of events of magnitude ≥ `M`. The slope `b`
(the *b-value*) is close to **1.0** for most tectonic regions — there are roughly
ten times as many magnitude-4 quakes as magnitude-5, and so on. The intercept `a`
measures overall seismic productivity.

The project estimates `b` two independent ways and compares them:

| Method | How | Notes |
| --- | --- | --- |
| **Maximum likelihood** (Aki 1965) | `b = log10(e) / (mean(M) − (Mc − ΔM/2))` | Standard error from Shi & Bolt (1982). Statistically unbiased. |
| **Least squares** | Linear fit of `log10(N)` vs `M` above `Mc` | Intuitive, but over-weights the well-populated small-magnitude bins. |

**Magnitude of completeness (`Mc`).** Below some magnitude, catalogs miss events.
`Mc` is estimated with the maximum-curvature method (the peak of the
non-cumulative distribution) and the b-value fit is restricted to `M ≥ Mc`.

## Project layout

```
earthquake-pattern-analysis/
├── main.py              # CLI pipeline: fetch -> analyze -> visualize -> map
├── src/
│   ├── data_fetch.py    # USGS FDSN client (pagination + caching)
│   ├── analysis.py      # G-R b-value, Mc, summary stats, clustering
│   ├── visualize.py     # Matplotlib figures
│   └── mapping.py       # Folium interactive maps
├── notebooks/
│   └── exploration.ipynb
├── outputs/             # generated figures, maps, report.json
├── requirements.txt
└── README.md
```

## Quick start

```bash
git clone <your-repo-url>
cd earthquake-pattern-analysis
pip install -r requirements.txt

# Global M>=4.5 earthquakes over the last 90 days
python main.py --days 90 --min-mag 4.5
```

Outputs land in `outputs/`:

- `gutenberg_richter.png` — frequency-magnitude distribution with the fit
- `magnitude_histogram.png`, `depth_distribution.png`, `temporal.png`,
  `global_scatter.png`
- `earthquake_map.html` — clustered, interactive marker map
- `earthquake_heatmap.html` — density heatmap of epicenters
- `report.json` — all computed statistics

### Useful flags

```bash
python main.py --start 2024-01-01 --end 2024-12-31 --min-mag 2.5  # fixed window
python main.py --days 30 --no-refresh        # reuse cached data
python main.py --days 90 --cluster           # add DBSCAN spatial clustering
python main.py --days 90 --no-maps           # skip the (slower) Folium maps
```

| Flag | Default | Description |
| --- | --- | --- |
| `--days` | 90 | Look-back window in days |
| `--start` / `--end` | — | Fixed ISO date window (overrides `--days`) |
| `--min-mag` | 4.5 | Minimum magnitude (lower = far more events) |
| `--bin-width` | 0.1 | Magnitude bin width for the G-R fit |
| `--cluster` | off | Run DBSCAN spatial clustering |
| `--no-maps` | off | Skip Folium maps |
| `--refresh` / `--no-refresh` | refresh | Re-fetch vs. use cached CSV |

## Use it as a library

```python
from src import data_fetch, analysis, visualize

df = data_fetch.fetch_earthquakes("2024-01-01", "2024-12-31", min_magnitude=4.5)

mc  = analysis.magnitude_of_completeness(df["magnitude"])
mle = analysis.b_value_mle(df["magnitude"], mc=mc)
print(f"b = {mle['b']:.2f} ± {mle['b_err']:.2f}")

visualize.plot_gutenberg_richter(df)
```

## Notes on the data

- Source: [USGS FDSN Event web service](https://earthquake.usgs.gov/fdsnws/event/1/).
  No API key required.
- A single request is capped at 20,000 events; the client automatically bisects
  large time windows and stitches the results together.
- Results are cached to `outputs/earthquakes_cache.csv` so repeated runs don't
  re-hit the service (use `--refresh` to force an update).
- Magnitudes are reported to 0.1 — which is why the maximum-likelihood estimator
  uses the standard ΔM/2 binning correction.

## References

- Gutenberg, B. & Richter, C. F. (1944). *Frequency of earthquakes in California.*
- Aki, K. (1965). *Maximum likelihood estimate of b in the formula log N = a − bM.*
- Shi, Y. & Bolt, B. A. (1982). *The standard error of the magnitude-frequency b value.*

## License

MIT — see [LICENSE](LICENSE).
