# prairie-urban-safety

**An open-source Python pipeline integrating climate, crime, and social media sentiment data for longitudinal urban safety monitoring across Canadian Prairie cities.**

Submitted to: *Computers, Environment and Urban Systems* — Special Issue: Open Urban Data Science  
Repository: `prairie-urban-safety`  
Licence: MIT (code) | CC BY 4.0 (data)

---

## Overview

`prairie-urban-safety` is a reproducible Python pipeline that automates the acquisition, processing, normalisation, statistical analysis, and geospatial visualisation of three complementary open data streams:

- **Climate observations** — Environment and Climate Change Canada (ECCC) Historical Climate Data Portal
- **Crime statistics** — Statistics Canada Uniform Crime Reporting (UCR) Survey (Table 35-10-0177-01)
- **Social media sentiment** — Reddit posts from city-specific subreddits (via Pushshift archive and Reddit API)

Applied to four Canadian Prairie cities — **Edmonton**, **Saskatoon**, **Regina**, and **Winnipeg** — across sixteen years (2009–2025), the pipeline produces:

1. An **era-stratification framework** identifying when Reddit-derived sentiment is and is not a reliable proxy for public safety perception
2. Verified **climate-crime correlation and regression results** by city
3. **Geospatial visualisations** of crime intensity and sentiment change across the Prairie region
4. **Reproducible statistical test outputs** for all six tests reported in the article

---

## Repository Structure

```
prairie-urban-safety/
│
├── README.md
├── LICENSE
│
├── scripts/
│   ├── figures/
│   │   ├── generate_figures.py          # Produces Figures 2–8 from raw CSVs
│   │   └── fig1_pipeline_architecture.py # Produces Figure 1 (no data required)
│   │
│   └── analysis/
│       ├── statistical_analysis.py      # Runs all six statistical tests
│       └── run_statistical_tests.py     # Runs tests and prints formatted tables
│
├── data/
│   ├── climate/
│   │   ├── Master_All_Cities_Daily_Climate_2010_to_2025_imputed.csv
│   │   ├── Master_All_Cities_Monthly_Climate_2010_to_2025_imputed.csv
│   │   └── Master_All_Cities_Yearly_Climate_2010_to_2025_imputed.csv
│   │
│   ├── crime/
│   │   ├── Actual_Incidents_Edmonton.csv
│   │   ├── Actual_Incidents_Regina.csv
│   │   ├── Actual_Incidents_Winnipeg.csv
│   │   └── Actual_Incidents_incidents_Saskatoon.csv
│   │
│   └── sentiment/
│       ├── reddit_climate_crime_2009_2013_sentiment.csv
│       ├── reddit_climate_crime_2014_2019_sentiment.csv
│       ├── reddit_climate_crime_2020_2021_sentiment.csv
│       ├── reddit_climate_crime_2022_2023_sentiment.csv
│       ├── reddit_climate_crime_2024_2025_sentiment.csv
│       ├── reddit_2009_2013_for_gis.csv
│       ├── reddit_2014_2019_for_gis.csv
│       ├── reddit_2020_2021_for_gis.csv
│       ├── reddit_2022_2023_for_gis.csv
│       └── reddit_2024_2025_for_gis.csv
│
└── figures/
    ├── fig1_pipeline_architecture.png
    ├── fig2_rolling_sentiment.png
    ├── fig3_volatility_heatmap.png
    ├── fig4_climate_crime_scatter.png
    ├── fig5_proportional_maps.png
    ├── fig6_era_comparison.png
    ├── fig7_spatial_sentiment.png
    └── fig8_polar_seasonal.png
```

---

## Pipeline Architecture (M1–M7)

| Module | Name | Description |
|--------|------|-------------|
| M1 | Climate Acquisition | Multi-station averaging, variable imputation, HDD/CDD derivation |
| M2 | Crime Integration | UCR violation mapping, annual aggregation to city level |
| M3 | Reddit Corpus Collection | Keyword classification, subreddit mapping, 2009–2025 |
| M4 | Sentiment Scoring | VADER compound scoring, categorical classification, crime-relevance filtering |
| M5 | Normalisation & Bias Mitigation | Per-post weighting, per-user aggregation, z-score anomaly scoring, seasonal adjustment |
| M6 | Statistical Analysis | Kruskal-Wallis, Mann-Whitney U, Chi-square, Wilcoxon, Pearson, OLS regression |
| M7 | Geospatial Visualisation | Maps, polar charts, heatmaps, scatter panels, era comparison charts |

---

## Installation

```bash
pip install pandas numpy scipy matplotlib statsmodels
```

No additional packages are required. All scripts run with Python 3.8+.

---

## Reproducing the Figures

### Figure 1 — Pipeline Architecture Diagram
```bash
python scripts/figures/fig1_pipeline_architecture.py
```
No data files required. Outputs `figures/fig1_pipeline_architecture.png`.

### Figures 2–8 — All Data Figures
```bash
python scripts/figures/generate_figures.py
```

Before running, update the `DATA_DIR` and `OUT_DIR` variables at the top of the script to point to your local data directory and desired output directory respectively.

All seven figures are generated in a single run. Expected runtime: under 60 seconds on a standard laptop.

---

## Reproducing the Statistical Tests

```bash
python scripts/analysis/run_statistical_tests.py
```

Update `DATA_DIR` at the top of the script. The script runs all six tests in sequence and prints formatted results tables to the console, matching the numbers reported in Tables 4–10 of the article.

**Tests run:**
- Test 1: Kruskal-Wallis H (inter-city polarity distributions)
- Test 2: Mann-Whitney U with Bonferroni correction (pairwise city comparisons)
- Test 3: Chi-square test of independence (categorical sentiment composition)
- Test 4: Wilcoxon signed-rank (crime vs. non-crime sentiment divergence)
- Test 5a: Pearson correlations (bivariate climate-crime)
- Test 5b: OLS regression + Durbin-Watson (multivariate climate-crime)

---

## Data Sources

| Dataset | Source | Years | Access |
|---------|--------|-------|--------|
| Climate (daily/monthly/yearly) | Environment and Climate Change Canada | 2010–2025 | [climate.weather.gc.ca](https://climate.weather.gc.ca) |
| Violent crime (UCR) | Statistics Canada Table 35-10-0177-01 | 2010–2024 | [statcan.gc.ca](https://www.statcan.gc.ca) |
| Reddit sentiment (Eras 1–5) | Pushshift archive + Reddit API | 2009–2025 | Archived on Zenodo (DOI: [to be assigned]) |

All processed datasets used in this study are archived on Zenodo under CC BY 4.0:  
**DOI: [to be assigned upon acceptance]**

---

## Era-Stratification Framework

The pipeline identifies five analytically distinct eras in the 2009–2025 Reddit data:

| Era | Period | Status |
|-----|--------|--------|
| Era 1 | 2009–2013 | Baseline — sparse, monthly-aggregated data |
| Era 2 | 2014–2019 | First analytically tractable era |
| Era 3 | 2020–2021 | Excluded — pandemic disruption |
| Era 4 | 2022–2023 | Post-pandemic re-stabilisation |
| Era 5 | 2024–2025 | Most analytically stable era |

---

## Key Findings

- **Edmonton** is the only city with robust climate-crime associations: mean humidity (r = −0.694, p = 0.004) and cooling degree days (r = +0.701, p = 0.004) significantly predict annual violent crime, consistent with a hot-dry physiological stress mechanism
- A post-2019 humidity decline of −7.3 percentage points in Edmonton coincides with a 49.8% increase in violent crime (2010–2024)
- **Winnipeg** shows no climate-crime associations despite the highest cooling degree days of any city (203/yr), confirming that heat load alone is insufficient and pointing to socioeconomic drivers
- **Regina** maintains a persistently positive sentiment profile across all five eras despite high per-capita crime severity — a structural discourse anomaly confirmed by Mann-Whitney U Bonferroni-corrected comparisons in all analytically tractable eras
- The era-stratification framework demonstrates that Reddit sentiment collapses as a reliable crime perception proxy during exogenous shocks (Era 3, pandemic), with Wilcoxon divergence lost in Regina (p = 0.394) and Winnipeg (p = 0.597) while retained in Edmonton and Saskatoon

---

## Citation

If you use this pipeline or dataset in your research, please cite:

> [Author names withheld for peer review]. (2026). *Climate Variability, Urban Crime, and Public Perceptions of Safety: A Social Media Sentiment Analysis of Canadian Prairie Cities*. Computers, Environment and Urban Systems. [DOI: to be assigned]

---

## Licence

- **Code:** MIT Licence — see `LICENSE`
- **Data:** CC BY 4.0 — see Zenodo archive

---

## Contact

For questions about the pipeline, data, or methodology, please open an issue on this repository or contact the corresponding author.
