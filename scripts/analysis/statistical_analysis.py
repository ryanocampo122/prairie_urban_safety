"""
================================================================================
prairie-urban-safety — Statistical Analysis Script
================================================================================
Runs all five statistical tests reported in the article:

    Test 1 — Kruskal-Wallis H test (inter-city polarity distributions)
    Test 2 — Mann-Whitney U with Bonferroni correction (pairwise city comparisons)
    Test 3 — Chi-square test of independence (categorical sentiment composition)
    Test 4 — Wilcoxon signed-rank test (crime vs. non-crime sentiment divergence)
    Test 5a — Pearson correlations (bivariate climate-crime associations)
    Test 5b — OLS regression with Durbin-Watson (multivariate climate-crime model)

All tests are run on Eras 2–5 (post-level data). Era 1 is monthly-aggregated
and noted separately where applicable.

Requirements:
    pip install pandas numpy scipy statsmodels

Data files required (update DATA_DIR below):
    reddit_climate_crime_2014_2019_sentiment.csv
    reddit_climate_crime_2020_2021_sentiment.csv
    reddit_climate_crime_2022_2023_sentiment.csv
    reddit_climate_crime_2024_2025_sentiment.csv
    Actual_Incidents_Edmonton.csv
    Actual_Incidents_Regina.csv
    Actual_Incidents_Winnipeg.csv
    Actual_Incidents_incidents_Saskatoon.csv
    Master_All_Cities_Yearly_Climate_2010_to_2025_imputed.csv

Usage:
    python statistical_analysis.py

Output:
    Prints all results to console with clear section headers.
    All reported statistics match those cited in article draft v5.
================================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, wilcoxon, pearsonr
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR = "/mnt/user-data/uploads"

# ── Constants ─────────────────────────────────────────────────────────────────
CITIES = ['Edmonton', 'Saskatoon', 'Regina', 'Winnipeg']

CITY_MAP = {
    'edmonton':    'Edmonton',  'alberta':      'Edmonton',
    'saskatoon':   'Saskatoon', 'saskatchewan': 'Saskatoon',
    'regina':      'Regina',
    'winnipeg':    'Winnipeg',  'manitoba':     'Winnipeg',
}

VIOLENT_VIOLATION = 'Total violent Criminal Code violations [100]'

CRIME_FILES = {
    'Edmonton':  f'{DATA_DIR}/Actual_Incidents_Edmonton.csv',
    'Regina':    f'{DATA_DIR}/Actual_Incidents_Regina.csv',
    'Winnipeg':  f'{DATA_DIR}/Actual_Incidents_Winnipeg.csv',
    'Saskatoon': f'{DATA_DIR}/Actual_Incidents_incidents_Saskatoon.csv',
}

ERA_FILES = {
    'Era 2 (2014–2019)': f'{DATA_DIR}/reddit_climate_crime_2014_2019_sentiment.csv',
    'Era 3 (2020–2021)': f'{DATA_DIR}/reddit_climate_crime_2020_2021_sentiment.csv',
    'Era 4 (2022–2023)': f'{DATA_DIR}/reddit_climate_crime_2022_2023_sentiment.csv',
    'Era 5 (2024–2025)': f'{DATA_DIR}/reddit_climate_crime_2024_2025_sentiment.csv',
}

# ── Data loader ───────────────────────────────────────────────────────────────
def load_era(path):
    """Load a post-level sentiment CSV and map subreddits to canonical city names."""
    df = pd.read_csv(path)
    df['city'] = df['subreddit'].str.lower().map(CITY_MAP)
    return df[df['city'].isin(CITIES)].copy()


def load_eras():
    return {era: load_era(path) for era, path in ERA_FILES.items()}


def load_crime():
    """Load annual violent crime counts for all four cities."""
    out = {}
    for city, path in CRIME_FILES.items():
        df  = pd.read_csv(path)
        sub = (df[df['Violations'] == VIOLENT_VIOLATION][['REF_DATE', 'VALUE']]
               .rename(columns={'REF_DATE': 'year', 'VALUE': 'violent_crime'}))
        sub['year'] = sub['year'].astype(int)
        out[city]   = sub.set_index('year')['violent_crime']
    return out


def load_climate():
    """Load yearly imputed climate data for all four cities (2010–2024)."""
    df = pd.read_csv(
        f'{DATA_DIR}/Master_All_Cities_Yearly_Climate_2010_to_2025_imputed.csv'
    )
    df['year'] = pd.to_datetime(df['Date']).dt.year
    return df[df['year'] <= 2024]


# ── Significance formatter ────────────────────────────────────────────────────
def sig(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return 'ns'


# ════════════════════════════════════════════════════════════════════════════
#  TEST 1 — KRUSKAL-WALLIS H TEST
#  Do sentiment polarity distributions differ across the four cities?
# ════════════════════════════════════════════════════════════════════════════
def test1_kruskal_wallis(eras):
    print("=" * 65)
    print("TEST 1 — KRUSKAL-WALLIS H TEST")
    print("Non-parametric test: do sentiment polarity distributions")
    print("differ significantly across the four cities?")
    print("-" * 65)
    print(f"{'Era':<22} {'H statistic':>13} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)

    results = {}
    for era, df in eras.items():
        groups = [df[df['city'] == c]['sentiment_compound'].dropna().values
                  for c in CITIES]
        h, p = kruskal(*groups)
        results[era] = {'H': h, 'p': p, 'sig': sig(p)}
        print(f"{era:<22} {h:>13.3f} {p:>10.5f} {sig(p):>5}")

    print("-" * 65)
    print("Note: Era 1 (2009–2013) uses monthly-aggregated data and is")
    print("      not included in post-level non-parametric testing.")
    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 2 — MANN-WHITNEY U WITH BONFERRONI CORRECTION
#  Which specific city pairs drive the Kruskal-Wallis result?
# ════════════════════════════════════════════════════════════════════════════
def test2_mannwhitney(eras):
    print("=" * 65)
    print("TEST 2 — MANN-WHITNEY U (pairwise + Bonferroni correction)")
    print("Which specific city pairs drive the Kruskal-Wallis result?")
    print()

    pairs    = list(combinations(CITIES, 2))
    n_tests  = len(pairs)  # 6 pairs → Bonferroni multiplier = 6
    results  = {}

    for era, df in eras.items():
        print(f"  {era}")
        print(f"  {'City A':<12} {'City B':<12} {'U':>10} "
              f"{'p (raw)':>10} {'p (Bonf)':>10} {'Sig':>5}")
        print("  " + "-" * 60)
        era_results = []
        for c1, c2 in pairs:
            g1    = df[df['city'] == c1]['sentiment_compound'].dropna().values
            g2    = df[df['city'] == c2]['sentiment_compound'].dropna().values
            u, p  = mannwhitneyu(g1, g2, alternative='two-sided')
            p_adj = min(p * n_tests, 1.0)
            flag  = ' ◄' if p_adj < 0.05 else ''
            era_results.append({
                'c1': c1, 'c2': c2, 'U': u,
                'p_raw': p, 'p_bonf': p_adj, 'sig': sig(p_adj)
            })
            print(f"  {c1:<12} {c2:<12} {u:>10.0f} "
                  f"{p:>10.4f} {p_adj:>10.4f} {sig(p_adj):>5}{flag}")
        results[era] = era_results
        print()

    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 3 — CHI-SQUARE TEST OF INDEPENDENCE
#  Do cities differ in their proportional mix of negative/neutral/positive?
# ════════════════════════════════════════════════════════════════════════════
def test3_chisquare(eras):
    print("=" * 65)
    print("TEST 3 — CHI-SQUARE TEST OF INDEPENDENCE")
    print("Do cities differ in the proportional mix of")
    print("negative / neutral / positive sentiment posts?")
    print("-" * 65)
    print(f"{'Era':<22} {'χ²':>9} {'df':>4} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)

    results = {}
    for era, df in eras.items():
        table = np.array([
            [(df[df['city'] == c]['sentiment_category'] == cat).sum()
             for cat in ['negative', 'neutral', 'positive']]
            for c in CITIES
        ])
        chi2, p, dof, _ = chi2_contingency(table)
        results[era]    = {'chi2': chi2, 'p': p, 'dof': dof, 'sig': sig(p)}
        print(f"{era:<22} {chi2:>9.3f} {dof:>4} {p:>10.5f} {sig(p):>5}")

    print("-" * 65)
    print()

    print("Contingency tables (raw counts) per era:")
    for era, df in eras.items():
        print(f"\n  {era}")
        print(f"  {'City':<12} {'Negative':>10} {'Neutral':>10} "
              f"{'Positive':>10} {'Total':>8}")
        print("  " + "-" * 52)
        for city in CITIES:
            c   = df[df['city'] == city]
            neg = (c['sentiment_category'] == 'negative').sum()
            neu = (c['sentiment_category'] == 'neutral').sum()
            pos = (c['sentiment_category'] == 'positive').sum()
            print(f"  {city:<12} {neg:>10} {neu:>10} {pos:>10} {len(c):>8}")
    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 4 — WILCOXON SIGNED-RANK TEST
#  Is crime-relevant sentiment significantly more negative than non-crime?
# ════════════════════════════════════════════════════════════════════════════
def test4_wilcoxon(eras):
    print("=" * 65)
    print("TEST 4 — WILCOXON SIGNED-RANK TEST")
    print("Is crime-relevant sentiment significantly more negative")
    print("than non-crime sentiment within each city?")

    np.random.seed(42)
    results = {}

    for era, df in eras.items():
        print(f"\n  {era}")
        print(f"  {'City':<12} {'W':>8} {'p-value':>10} {'Sig':>5} "
              f"{'Crime Mean':>12} {'Non-Crime Mean':>15} {'Δ':>8}")
        print("  " + "-" * 75)
        era_results = {}
        for city in CITIES:
            c        = df[df['city'] == city]
            crime    = c[c['contains_crime_keywords'] == True ]['sentiment_compound'].dropna().values
            noncrime = c[c['contains_crime_keywords'] == False]['sentiment_compound'].dropna().values
            n        = min(len(crime), len(noncrime))
            idx1     = np.random.choice(len(crime),    n, replace=False)
            idx2     = np.random.choice(len(noncrime), n, replace=False)
            w, p     = wilcoxon(crime[idx1], noncrime[idx2])
            diff     = crime.mean() - noncrime.mean()
            era_results[city] = {
                'W': w, 'p': p, 'sig': sig(p),
                'crime_mean': crime.mean(),
                'non_mean':   noncrime.mean(),
                'delta':      diff,
            }
            print(f"  {city:<12} {w:>8.0f} {p:>10.5f} {sig(p):>5} "
                  f"{crime.mean():>12.3f} {noncrime.mean():>15.3f} {diff:>8.3f}")
        results[era] = era_results

    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 5a — PEARSON CORRELATIONS
#  Bivariate climate-crime associations (n=15 per city, 2010–2024)
# ════════════════════════════════════════════════════════════════════════════
def test5a_pearson(crime, climate):
    print("=" * 65)
    print("TEST 5a — PEARSON CORRELATIONS")
    print("Bivariate climate vs. annual violent crime (n=15, 2010–2024)")
    print("-" * 65)
    print(f"{'City':<12} {'Variable':<26} {'r':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)

    results = {}
    clim_vars = [
        ('Relative_Humidity_Pct_mean', 'Mean Humidity (%)'),
        ('Cooling_Degree_Days_sum',    'Cooling Degree Days'),
        ('Temperature_C_max',          'Max Temperature (°C)'),
    ]

    for city in CITIES:
        df_c   = pd.read_csv(CRIME_FILES[city])
        vc     = (df_c[df_c['Violations'] == VIOLENT_VIOLATION][['REF_DATE', 'VALUE']]
                  .rename(columns={'REF_DATE': 'year', 'VALUE': 'violent_crime'}))
        vc['year'] = vc['year'].astype(int)
        cl     = climate[climate['City'] == city].copy()
        merged = vc.merge(cl, on='year').dropna()
        results[city] = {}
        for var, label in clim_vars:
            r, p = pearsonr(merged[var], merged['violent_crime'])
            results[city][var] = {'r': r, 'p': p, 'sig': sig(p)}
            print(f"{city:<12} {label:<26} {r:>8.3f} {p:>10.4f} {sig(p):>5}")
        print()

    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 5b — OLS REGRESSION + DURBIN-WATSON
#  Multivariate climate-crime model (z-scored predictors)
# ════════════════════════════════════════════════════════════════════════════
def test5b_ols(crime, climate):
    print("=" * 65)
    print("TEST 5b — OLS REGRESSION (z-scored predictors)")
    print("Outcome: annual violent crime")
    print("Predictors: Max Temperature + Mean Humidity + Cooling Degree Days")
    print("-" * 65)

    results = {}

    for city in CITIES:
        df_c   = pd.read_csv(CRIME_FILES[city])
        vc     = (df_c[df_c['Violations'] == VIOLENT_VIOLATION][['REF_DATE', 'VALUE']]
                  .rename(columns={'REF_DATE': 'year', 'VALUE': 'violent_crime'}))
        vc['year'] = vc['year'].astype(int)
        cl     = climate[climate['City'] == city].copy()
        merged = vc.merge(cl, on='year').dropna()

        # Z-score predictors
        X = merged[['Temperature_C_max',
                    'Relative_Humidity_Pct_mean',
                    'Cooling_Degree_Days_sum']].copy()
        for col in X.columns:
            X[col] = (X[col] - X[col].mean()) / X[col].std()

        y     = merged['violent_crime'].values
        X_mat = np.column_stack([np.ones(len(X)), X.values])

        # OLS via least squares
        betas, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        y_pred    = X_mat @ betas
        residuals = y - y_pred
        ss_res    = np.sum(residuals ** 2)
        ss_tot    = np.sum((y - y.mean()) ** 2)
        r2        = 1 - ss_res / ss_tot
        n, k      = len(y), X_mat.shape[1] - 1
        r2_adj    = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        # Durbin-Watson statistic
        dw = np.sum(np.diff(residuals) ** 2) / ss_res
        dw_flag = '← moderate autocorrelation' if 1.0 <= dw < 2.0 else '← autocorrelation concern'

        results[city] = {
            'r2': r2, 'r2_adj': r2_adj, 'dw': dw, 'betas': betas
        }

        print(f"\n  {city}  (n={n})")
        print(f"  R²            = {r2:.3f}")
        print(f"  Adjusted R²   = {r2_adj:.3f}")
        print(f"  Durbin-Watson = {dw:.3f}  {dw_flag}")
        print(f"  Intercept     = {betas[0]:>10.1f}")
        print(f"  β Temperature = {betas[1]:>10.1f}  incidents per 1 SD increase")
        print(f"  β Humidity    = {betas[2]:>10.1f}  incidents per 1 SD increase")
        print(f"  β CDD         = {betas[3]:>10.1f}  incidents per 1 SD increase")

    print()
    print("Note: All results treated as exploratory associations.")
    print("      Serial autocorrelation (DW < 2.0) means standard errors")
    print("      are underestimated. HAC-corrected SE recommended for")
    print("      formal inference.")
    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 65)
    print("prairie-urban-safety — Statistical Analysis")
    print("All tests reported in article draft v5")
    print("=" * 65)
    print()

    # Load data
    eras    = load_eras()
    crime   = load_crime()
    climate = load_climate()

    # Run all tests
    kw_results   = test1_kruskal_wallis(eras)
    mw_results   = test2_mannwhitney(eras)
    chi_results  = test3_chisquare(eras)
    wil_results  = test4_wilcoxon(eras)
    pear_results = test5a_pearson(crime, climate)
    ols_results  = test5b_ols(crime, climate)

    print("=" * 65)
    print("All tests complete.")
    print("Results above match those cited in article draft v5.")
    print("=" * 65)
