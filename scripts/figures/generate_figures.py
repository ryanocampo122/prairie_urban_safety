"""
================================================================================
UrbanSentinel-Prairie — Figure Generation Script
================================================================================
Produces all 7 publication-quality figures for the article:

    "Climate Variability, Urban Crime, and Public Perceptions of Safety:
     A Social Media Sentiment Analysis of Canadian Prairie Cities"

Figures generated:
    Fig 2 — Rolling mean sentiment trajectory (2009–2025)
    Fig 3 — Sentiment volatility heatmap (city × era)
    Fig 4 — Climate-crime scatter panels (humidity + CDD vs. violent crime)
    Fig 5 — Proportional symbol maps — violent crime growth (2010, 2017, 2024)
    Fig 6 — Era comparison chart (volume, crime share, sentiment)
    Fig 7 — Spatial sentiment change map (pre vs. post 2019)
    Fig 8 — Polar seasonal sentiment charts

Requirements:
    pip install pandas numpy matplotlib scipy

Data files required (place in DATA_DIR or update paths below):
    reddit_climate_crime_2009_2013_sentiment.csv  (Era 1 — monthly aggregated)
    reddit_climate_crime_2014_2019_sentiment.csv  (Era 2 — post level)
    reddit_climate_crime_2020_2021_sentiment.csv  (Era 3 — post level)
    reddit_climate_crime_2022_2023_sentiment.csv  (Era 4 — post level)
    reddit_climate_crime_2024_2025_sentiment.csv  (Era 5 — post level)
    reddit_2014_2019_for_gis.csv
    reddit_2022_2023_for_gis.csv
    reddit_2024_2025_for_gis.csv
    Actual_Incidents_Edmonton.csv
    Actual_Incidents_Regina.csv
    Actual_Incidents_Winnipeg.csv
    Actual_Incidents_incidents_Saskatoon.csv
    Master_All_Cities_Yearly_Climate_2010_to_2025_imputed.csv

Usage:
    python generate_figures.py

Output:
    All figures saved as high-resolution PNG (300 DPI) to OUT_DIR.
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats

warnings.filterwarnings('ignore')

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR = "/mnt/user-data/uploads"
OUT_DIR  = "/home/claude/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared constants ──────────────────────────────────────────────────────────
CITIES = ['Edmonton', 'Saskatoon', 'Regina', 'Winnipeg']

CITY_COLORS = {
    'Edmonton':  '#E07B3A',
    'Saskatoon': '#4A90C4',
    'Regina':    '#9B59B6',
    'Winnipeg':  '#C0392B',
}

CITY_MARKERS = {
    'Edmonton':  'o',
    'Saskatoon': '^',
    'Regina':    'D',
    'Winnipeg':  's',
}

CITY_COORDS = {
    'Edmonton':  (53.5461, -113.4938),
    'Saskatoon': (52.1332, -106.6700),
    'Regina':    (50.4452, -104.6189),
    'Winnipeg':  (49.8951,  -97.1384),
}

CITY_LABEL_OFFSETS = {
    'Edmonton':  ( 2.2,  0.6),
    'Saskatoon': ( 1.8, -0.9),
    'Regina':    ( 1.8,  0.5),
    'Winnipeg':  (-7.8,  0.5),
}

ERA_SHADE = ['#ECEFF1', '#E3F2FD', '#FFEBEE', '#F1F8E9', '#EDE7F6']
ERA_LABELS = [
    'Era 1\n2009–2013',
    'Era 2\n2014–2019',
    'Era 3\n2020–2021',
    'Era 4\n2022–2023',
    'Era 5\n2024–2025',
]

# Correct aspect ratio for Prairie region (~52.5°N):
# 1 degree longitude = cos(52.5°) × 111 km ≈ 68 km
# 1 degree latitude  = 111 km
# → aspect = 111/68 ≈ 1.63
MAP_ASPECT = 1.0 / np.cos(np.radians(52.5))

# Subreddit → canonical city name
CITY_MAP = {
    'edmonton': 'Edmonton', 'alberta':      'Edmonton',
    'saskatoon':'Saskatoon', 'saskatchewan': 'Saskatoon',
    'regina':   'Regina',
    'winnipeg': 'Winnipeg', 'manitoba':     'Winnipeg',
}

# Violation string used in Statistics Canada UCR files
VIOLENT_VIOLATION = 'Total violent Criminal Code violations [100]'

# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_all_sentiment():
    """
    Merge all five era sentiment files into a single long-format DataFrame
    with columns: city, date (monthly timestamp), sentiment_compound, era.

    Era 1 is pre-aggregated at monthly resolution (mean_polarity used).
    Eras 2–5 are post-level; aggregated to monthly means before merging.
    """
    frames = []

    # ── Era 1: monthly aggregated ─────────────────────────────────
    e1 = pd.read_csv(f'{DATA_DIR}/reddit_climate_crime_2009_2013_sentiment.csv')
    e1['city']               = e1['city'].map(CITY_MAP).fillna(e1['city'])
    e1['date']               = pd.to_datetime(e1['year_month'] + '-01')
    e1['sentiment_compound'] = e1['mean_polarity']
    e1['era']                = 'Era 1'
    frames.append(e1[['city', 'date', 'sentiment_compound', 'era']])

    # ── Eras 2–5: post-level ──────────────────────────────────────
    era_files = [
        ('Era 2', 'reddit_climate_crime_2014_2019_sentiment.csv'),
        ('Era 3', 'reddit_climate_crime_2020_2021_sentiment.csv'),
        ('Era 4', 'reddit_climate_crime_2022_2023_sentiment.csv'),
        ('Era 5', 'reddit_climate_crime_2024_2025_sentiment.csv'),
    ]
    for era_name, fname in era_files:
        df = pd.read_csv(f'{DATA_DIR}/{fname}', parse_dates=['created_date'])
        df['city'] = df['subreddit'].str.lower().map(CITY_MAP)
        df = df[df['city'].isin(CITIES)].copy()
        df['era']  = era_name
        df['date'] = (df['created_date']
                      .dt.tz_localize(None)
                      .dt.to_period('M')
                      .dt.to_timestamp())
        frames.append(df[['city', 'date', 'sentiment_compound', 'era']])

    all_df = pd.concat(frames, ignore_index=True).sort_values('date')
    return all_df


def load_crime():
    """
    Load annual violent crime counts for all four cities.
    Returns dict: {city: pd.Series indexed by year}
    """
    crime_files = {
        'Edmonton':  'Actual_Incidents_Edmonton.csv',
        'Regina':    'Actual_Incidents_Regina.csv',
        'Winnipeg':  'Actual_Incidents_Winnipeg.csv',
        'Saskatoon': 'Actual_Incidents_incidents_Saskatoon.csv',
    }
    out = {}
    for city, fname in crime_files.items():
        df  = pd.read_csv(f'{DATA_DIR}/{fname}')
        sub = (df[df['Violations'] == VIOLENT_VIOLATION][['REF_DATE', 'VALUE']]
               .rename(columns={'REF_DATE': 'year', 'VALUE': 'violent_crime'}))
        sub['year'] = sub['year'].astype(int)
        out[city]   = sub.set_index('year')['violent_crime']
    return out


def load_climate():
    """
    Load yearly imputed climate data for all four cities (2010–2024).
    """
    df = pd.read_csv(
        f'{DATA_DIR}/Master_All_Cities_Yearly_Climate_2010_to_2025_imputed.csv'
    )
    df['year'] = pd.to_datetime(df['Date']).dt.year
    return df[df['year'] <= 2024]


# ════════════════════════════════════════════════════════════════════════════
#  MAP BASEMAP HELPER
# ════════════════════════════════════════════════════════════════════════════

def draw_prairie_basemap(ax, xlim=(-120, -94), ylim=(48.5, 60.5)):
    """
    Draw a geographically corrected Prairie basemap on ax.

    Applies MAP_ASPECT so that longitude degrees are correctly scaled
    relative to latitude degrees at ~52.5°N. Province boundaries are
    drawn from their precise legal coordinates. No external shapefiles
    or internet access required.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(MAP_ASPECT)
    ax.set_facecolor('#D6EAF8')

    # Province rectangles
    provinces = [
        # x0,   y0, width, height, facecolor,  edgecolor,  label,          lx,      ly
        (-120,   49,  10,    11,   '#EAF4EC', '#999999', 'Alberta',       -115.5,  55.5),
        (-110,   49,   8.5,  11,   '#FEF9E7', '#999999', 'Saskatchewan', -106.2,  55.5),
        (-101.5, 49,   6.5,  11,   '#FDEBD0', '#999999', 'Manitoba',      -98.5,  55.5),
    ]
    for x0, y0, w, h, fc, ec, label, lx, ly in provinces:
        rect = patches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle='square,pad=0',
            linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=1
        )
        ax.add_patch(rect)
        ax.text(lx, ly, label, ha='center', va='center',
                fontsize=8.5, color='#666666', style='italic', zorder=2)

    # US border at 49°N
    ax.axhline(49, color='#888888', linewidth=1.0,
               linestyle='--', alpha=0.6, zorder=3)
    ax.text(-107, 48.65, 'United States', ha='center',
            fontsize=7.5, color='#888888', style='italic')

    # Graticule
    for lon in range(-120, -93, 5):
        ax.axvline(lon, color='#cccccc', linewidth=0.4,
                   linestyle=':', alpha=0.5, zorder=0)
    for lat in range(50, 61, 5):
        ax.axhline(lat, color='#cccccc', linewidth=0.4,
                   linestyle=':', alpha=0.5, zorder=0)

    ax.set_xticks(range(-120, -93, 5))
    ax.set_xticklabels([f'{abs(x)}°W' for x in range(-120, -93, 5)], fontsize=7)
    ax.set_yticks(range(49, 61, 2))
    ax.set_yticklabels([f'{y}°N' for y in range(49, 61, 2)], fontsize=7)
    ax.tick_params(length=3, color='#aaaaaa')


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Rolling Mean Sentiment Trajectory
# ════════════════════════════════════════════════════════════════════════════

def fig2_rolling_sentiment():
    """
    Four-city rolling mean sentiment trajectory 2009–2025 with era shading.
    Data: all five reddit_climate_crime_*.csv files merged.
    """
    print("Fig 2 — Rolling mean sentiment trajectory...")
    all_df  = load_all_sentiment()
    monthly = (all_df.groupby(['city', 'date'])['sentiment_compound']
               .mean().reset_index().sort_values(['city', 'date']))

    fig, ax = plt.subplots(figsize=(14, 6))

    # Era shading bands
    era_dates = [
        ('2009-01-01', '2013-12-31', ERA_SHADE[0], ERA_LABELS[0]),
        ('2014-01-01', '2019-12-31', ERA_SHADE[1], ERA_LABELS[1]),
        ('2020-01-01', '2021-12-31', ERA_SHADE[2], ERA_LABELS[2]),
        ('2022-01-01', '2023-12-31', ERA_SHADE[3], ERA_LABELS[3]),
        ('2024-01-01', '2025-12-31', ERA_SHADE[4], ERA_LABELS[4]),
    ]
    ymax = 0.65
    for start, end, color, label in era_dates:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.35, color=color, zorder=0)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        short, period = label.split('\n')
        ax.text(mid, ymax - 0.04, short,   ha='center', va='top',
                fontsize=7.5, color='#555555', fontweight='bold')
        ax.text(mid, ymax - 0.11, period,  ha='center', va='top',
                fontsize=7.0, color='#777777')

    # 3-month rolling mean per city
    for city in CITIES:
        c = (monthly[monthly['city'] == city]
             .set_index('date')
             .resample('ME')['sentiment_compound']
             .mean())
        rolling = c.rolling(window=3, min_periods=1).mean()
        ax.plot(rolling.index, rolling.values,
                color=CITY_COLORS[city], linewidth=1.8,
                label=city, zorder=3)

    ax.axhline(0, color='#333333', linewidth=0.8, linestyle='-', alpha=0.5, zorder=2)
    ax.set_xlim(pd.Timestamp('2009-01-01'), pd.Timestamp('2025-12-31'))
    ax.set_ylim(-0.55, ymax)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Mean Sentiment Score (3-month rolling mean)', fontsize=11)
    ax.set_title(
        'Figure 2. Monthly Sentiment Score Trajectory by City and Era (2009–2025)',
        fontsize=12, fontweight='bold', pad=14
    )
    ax.legend(loc='lower left', fontsize=9.5, framealpha=0.9, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    path = f'{OUT_DIR}/fig2_rolling_sentiment.png'
    plt.savefig(path)
    plt.close()
    print(f"  ✓  {path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Sentiment Volatility Heatmap
# ════════════════════════════════════════════════════════════════════════════

def fig3_volatility_heatmap():
    """
    4 × 5 colour grid: monthly sentiment standard deviation by city and era.
    Darker cells = higher volatility.
    """
    print("Fig 3 — Sentiment volatility heatmap...")
    all_df  = load_all_sentiment()
    monthly = (all_df.groupby(['city', 'era', 'date'])['sentiment_compound']
               .mean().reset_index())
    vol     = (monthly.groupby(['city', 'era'])['sentiment_compound']
               .std().reset_index())

    era_order  = ['Era 1', 'Era 2', 'Era 3', 'Era 4', 'Era 5']
    era_labels = ERA_LABELS

    matrix = np.full((4, 5), np.nan)
    for i, city in enumerate(CITIES):
        for j, era in enumerate(era_order):
            row = vol[(vol['city'] == city) & (vol['era'] == era)]
            if len(row):
                matrix[i, j] = row['sentiment_compound'].values[0]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=0.55)

    ax.set_xticks(range(5));   ax.set_xticklabels(era_labels, fontsize=10)
    ax.set_yticks(range(4));   ax.set_yticklabels(CITIES, fontsize=10)

    for i in range(4):
        for j in range(5):
            val   = matrix[i, j]
            tcolor = 'white' if val > 0.30 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10.5, fontweight='bold', color=tcolor)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Std. Deviation', fontsize=10)
    ax.set_title('Figure 3. Sentiment Volatility (Std. Deviation) by City and Era',
                 fontsize=12, fontweight='bold', pad=12)

    plt.tight_layout()
    path = f'{OUT_DIR}/fig3_volatility_heatmap.png'
    plt.savefig(path)
    plt.close()
    print(f"  ✓  {path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Climate-Crime Scatter Panels
# ════════════════════════════════════════════════════════════════════════════

def fig4_climate_crime_scatter():
    """
    2 × 4 scatter grid: humidity (top row) and cooling degree days (bottom row)
    vs. annual violent crime per city. OLS fit line, Pearson r and p annotated.
    Key verified results:
        Edmonton humidity  r = -0.694  p = 0.004 **
        Edmonton CDD       r = +0.701  p = 0.004 **
    """
    print("Fig 4 — Climate-crime scatter panels...")
    crime = load_crime()
    clim  = load_climate()

    vars_to_plot = [
        ('Relative_Humidity_Pct_mean', 'Mean Annual Relative Humidity (%)'),
        ('Cooling_Degree_Days_sum',    'Annual Cooling Degree Days (sum)'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle(
        'Figure 4. Climate Variables vs. Annual Violent Crime (2010–2024)',
        fontsize=13, fontweight='bold', y=1.01
    )

    row_labels = ['Humidity → Violent Crime', 'Cooling Degree Days → Violent Crime']

    for row, (clim_var, xlabel) in enumerate(vars_to_plot):
        for col, city in enumerate(CITIES):
            ax = axes[row, col]
            cl = clim[clim['City'] == city][['year', clim_var]].dropna()
            cr = crime[city].reset_index()
            merged = cl.merge(cr, on='year').dropna()

            x     = merged[clim_var].values
            y     = merged['violent_crime'].values
            years = merged['year'].values

            r, p = stats.pearsonr(x, y)
            sig  = ('**' if p < 0.01 else
                    ('*'  if p < 0.05 else
                     ('†'  if p < 0.10 else 'ns')))

            ax.scatter(x, y, color=CITY_COLORS[city], s=55, zorder=4,
                       alpha=0.85, marker=CITY_MARKERS[city],
                       edgecolors='white', linewidth=0.5)

            for xi, yi, yr in zip(x, y, years):
                ax.annotate(f"'{str(yr)[2:]}", (xi, yi),
                            textcoords='offset points', xytext=(4, 3),
                            fontsize=7, color='#444444')

            if len(x) > 2:
                m, b   = np.polyfit(x, y, 1)
                xline  = np.linspace(x.min(), x.max(), 100)
                ax.plot(xline, m * xline + b, color=CITY_COLORS[city],
                        linewidth=1.5, linestyle='--', alpha=0.7, zorder=3)

            sig_color = ('#1a5276' if p < 0.05 else
                         '#7d6608' if p < 0.10 else '#555555')
            ax.annotate(
                f'r = {r:+.3f} {sig}\np = {p:.3f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=8.5, va='top', ha='left',
                color=sig_color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=sig_color, alpha=0.8)
            )

            ax.set_xlabel(xlabel, fontsize=8.5)
            if col == 0:
                ax.set_ylabel('Annual Violent Crime Incidents', fontsize=8.5)
            ax.set_title(city, fontsize=10.5, fontweight='bold',
                         color=CITY_COLORS[city])
            ax.tick_params(labelsize=8)

        fig.text(-0.005, 0.75 - row * 0.5, row_labels[row],
                 va='center', ha='right', fontsize=9,
                 fontweight='bold', rotation=90, color='#333333')

    plt.tight_layout()
    path = f'{OUT_DIR}/fig4_climate_crime_scatter.png'
    plt.savefig(path)
    plt.close()
    print(f"  ✓  {path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Proportional Symbol Maps (Crime Growth)
# ════════════════════════════════════════════════════════════════════════════

def fig5_proportional_maps():
    """
    Three-panel Prairie map showing violent crime bubble size at each city
    for 2010, 2017, and 2024. Bubble area proportional to incident count.
    Map uses MAP_ASPECT correction for geographic accuracy at 52.5°N.
    """
    print("Fig 5 — Proportional symbol maps...")
    crime      = load_crime()
    snap_years = [2010, 2017, 2024]
    all_vals   = [crime[c][y] for c in CITIES
                  for y in snap_years if y in crime[c].index]
    max_val    = max(all_vals)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        'Figure 5. Violent Crime Intensity Across Prairie Cities — 2010, 2017 and 2024\n'
        'Bubble area proportional to annual violent crime incidents',
        fontsize=12, fontweight='bold', y=1.01
    )

    for ax, yr in zip(axes, snap_years):
        draw_prairie_basemap(ax, xlim=(-121, -93), ylim=(48.0, 61.0))
        ax.set_title(str(yr), fontsize=14, fontweight='bold',
                     pad=10, color='#222222')

        # Per-figure label offsets — fan SE cities apart clearly
        fig5_offsets = {
            'Edmonton':  ( 1.8,  0.7),
            'Saskatoon': ( 1.8,  0.5),
            'Regina':    ( 1.8,  0.5),
            'Winnipeg':  ( 1.8, -0.8),
        }

        for city in CITIES:
            lat, lon = CITY_COORDS[city]
            if yr not in crime[city].index:
                continue
            val  = crime[city][yr]
            size = (val / max_val) * 2200

            ax.scatter(lon, lat, s=size, color=CITY_COLORS[city],
                       alpha=0.55, edgecolors=CITY_COLORS[city],
                       linewidth=1.8, zorder=6)
            ax.scatter(lon, lat, s=22, color='white', zorder=7)

            ox, oy = fig5_offsets[city]
            ax.annotate(
                f'{city}\n{val:,.0f}',
                xy=(lon, lat), xytext=(lon + ox, lat + oy),
                fontsize=8, fontweight='bold',
                ha='left', va='center',
                color=CITY_COLORS[city], zorder=8,
                arrowprops=dict(arrowstyle='-',
                                color=CITY_COLORS[city], lw=0.8)
            )

        if yr == snap_years[0]:
            ax.set_ylabel('Latitude', fontsize=8.5)
        ax.set_xlabel('Longitude', fontsize=8.5)

    # Bubble size legend — outside all map panels, top right of whole figure
    legend_handles = []
    for ref_val, label in [(5000, '5,000'), (12000, '12,000'), (22000, '22,000')]:
        size = (ref_val / max_val) * 2800
        h = axes[0].scatter([], [], s=size, color='#999999', alpha=0.55,
                            label=f'{label} incidents',
                            edgecolors='#999999', linewidth=1)
        legend_handles.append(h)

    fig.legend(
        handles=legend_handles,
        title='Annual Violent\nCrime Incidents',
        title_fontsize=8,
        fontsize=7.5,
        framealpha=0.92,
        edgecolor='#cccccc',
        loc='upper right',
        bbox_to_anchor=(0.995, 0.97),
        bbox_transform=fig.transFigure,
    )

    plt.tight_layout(w_pad=2.0)
    plt.subplots_adjust(right=0.87)
    path = f'{OUT_DIR}/fig5_proportional_maps.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓  {path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Era Comparison Chart
# ════════════════════════════════════════════════════════════════════════════

def fig6_era_comparison():
    """
    Grouped bar chart comparing post volume, % crime-relevant posts,
    and mean sentiment across all five eras and four cities.
    """
    print("Fig 6 — Era comparison chart...")

    rows = []

    # Era 1 — pre-aggregated
    e1 = pd.read_csv(f'{DATA_DIR}/reddit_climate_crime_2009_2013_sentiment.csv')
    for city in CITIES:
        c = e1[e1['city'] == city]
        rows.append({
            'era':       'Era 1\n2009–2013',
            'city':       city,
            'n_posts':    c['n_posts'].sum(),
            'pct_crime': (c['n_posts_crime'].sum() / c['n_posts'].sum()
                          if c['n_posts'].sum() > 0 else np.nan),
            'mean_sent':  c['mean_polarity'].mean(),
        })

    # Eras 2–5 — post-level
    era_files = [
        ('Era 2\n2014–2019', 'reddit_climate_crime_2014_2019_sentiment.csv'),
        ('Era 3\n2020–2021', 'reddit_climate_crime_2020_2021_sentiment.csv'),
        ('Era 4\n2022–2023', 'reddit_climate_crime_2022_2023_sentiment.csv'),
        ('Era 5\n2024–2025', 'reddit_climate_crime_2024_2025_sentiment.csv'),
    ]
    for era_label, fname in era_files:
        df = pd.read_csv(f'{DATA_DIR}/{fname}')
        df['city_clean'] = df['subreddit'].str.lower().map(CITY_MAP)
        df = df[df['city_clean'].isin(CITIES)]
        for city in CITIES:
            c = df[df['city_clean'] == city]
            rows.append({
                'era':       era_label,
                'city':       city,
                'n_posts':    len(c),
                'pct_crime':  c['contains_crime_keywords'].mean(),
                'mean_sent':  c['sentiment_compound'].mean(),
            })

    sumdf = pd.DataFrame(rows)
    eras  = sumdf['era'].unique()
    x     = np.arange(len(eras))
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        'Figure 6. Era Comparison: Post Volume, Crime Discussion Share, and Mean Sentiment',
        fontsize=12, fontweight='bold'
    )

    metrics = [
        ('n_posts',   'Total Posts',                False),
        ('pct_crime', '% Crime-Relevant Posts',     True),
        ('mean_sent', 'Mean Sentiment Score',        False),
    ]

    for ax, (metric, ylabel, as_pct) in zip(axes, metrics):
        for i, city in enumerate(CITIES):
            vals = []
            for e in eras:
                row = sumdf[(sumdf['era'] == e) & (sumdf['city'] == city)][metric]
                vals.append(row.values[0] if len(row) else 0)
            if as_pct:
                vals = [v * 100 for v in vals]
            ax.bar(x + i * width, vals, width,
                   label=city, color=CITY_COLORS[city],
                   alpha=0.85, edgecolor='white', linewidth=0.5)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(eras, fontsize=8)
        ax.set_ylabel(ylabel + (' (%)' if as_pct else ''), fontsize=9.5)
        ax.set_title(ylabel, fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8.5)
        if metric == 'mean_sent':
            ax.axhline(0, color='#333333', linewidth=0.8,
                       linestyle='--', alpha=0.6)

    axes[0].legend(fontsize=8.5, loc='upper left', ncol=2, framealpha=0.9)
    plt.tight_layout()
    path = f'{OUT_DIR}/fig6_era_comparison.png'
    plt.savefig(path)
    plt.close()
    print(f"  ✓  {path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Spatial Sentiment Change Map
# ════════════════════════════════════════════════════════════════════════════

def fig7_spatial_sentiment():
    """
    Two-panel Prairie map comparing % negative Reddit posts per city:
    2014–2019 (pre-pandemic baseline) vs. 2022–2025 (post-pandemic).
    Bubble colour encodes negativity level (RdYlGn_r colormap).
    Colourbar placed in a dedicated GridSpec column (not overlapping maps).
    Map uses MAP_ASPECT correction for geographic accuracy at 52.5°N.
    """
    print("Fig 7 — Spatial sentiment change map...")

    def get_pct_neg(fname):
        df = pd.read_csv(f'{DATA_DIR}/{fname}')
        df['city_clean'] = df['subreddit'].str.lower().map(CITY_MAP)
        df = df[df['city_clean'].isin(CITIES)]
        return (df.groupby('city_clean')
                  .apply(lambda x: (x['sentiment_category'] == 'negative').mean())
                  .to_dict())

    pre   = get_pct_neg('reddit_2014_2019_for_gis.csv')
    post4 = get_pct_neg('reddit_2022_2023_for_gis.csv')
    post5 = get_pct_neg('reddit_2024_2025_for_gis.csv')

    # Average post-pandemic eras
    post = {}
    for city in CITIES:
        vals = [v for v in [post4.get(city), post5.get(city)]
                if v is not None and not np.isnan(v)]
        post[city] = np.mean(vals) if vals else np.nan

    changes = {c: post.get(c, np.nan) - pre.get(c, np.nan) for c in CITIES}

    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0.28, vmax=0.52)

    # GridSpec: 2 map panels + 1 narrow colourbar column (no overlap)
    fig = plt.figure(figsize=(19, 7))
    fig.patch.set_facecolor('white')
    gs  = GridSpec(1, 3, figure=fig,
                   width_ratios=[1, 1, 0.045],
                   wspace=0.12, left=0.05, right=0.97,
                   top=0.88, bottom=0.10)
    ax_pre  = fig.add_subplot(gs[0, 0])
    ax_post = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])
    axes    = [ax_pre, ax_post]

    fig.suptitle(
        'Figure 7. Spatial Distribution of Reddit Negative Sentiment:\n'
        '2014–2019 (Pre-Pandemic) vs. 2022–2025 (Post-Pandemic)',
        fontsize=12, fontweight='bold', y=1.00
    )

    period_data = [
        (pre,  '2014–2019\n(Pre-Pandemic Baseline)', False),
        (post, '2022–2025\n(Post-Pandemic)',          True),
    ]

    # Per-figure label offsets — override global ones for fig7
    # because cities are close together in the SE of the map
    fig7_offsets = {
        'Edmonton':  ( 1.8,  0.7),
        'Saskatoon': ( 1.8, -0.7),
        'Regina':    ( 1.8,  0.5),
        'Winnipeg':  ( 1.8, -0.7),
    }

    for ax, (data, title, show_change) in zip(axes, period_data):
        draw_prairie_basemap(ax, xlim=(-121, -93), ylim=(48.0, 61.0))
        ax.set_title(title, fontsize=11, fontweight='bold',
                     pad=10, color='#222222')

        for city in CITIES:
            lat, lon = CITY_COORDS[city]
            val = data.get(city, np.nan)
            if np.isnan(val):
                continue

            ax.scatter(lon, lat, s=420,
                       color=cmap(norm(val)),
                       edgecolors='#333333', linewidth=1.2,
                       zorder=6, alpha=0.92)

            ox, oy = fig7_offsets[city]
            change_str = ''
            if show_change:
                ch = changes.get(city, np.nan)
                if not np.isnan(ch):
                    sign = '+' if ch >= 0 else ''
                    change_str = f'\n({sign}{ch:.1%} Δ)'

            ax.annotate(
                f'{city}\n{val:.1%} neg{change_str}',
                xy=(lon, lat), xytext=(lon + ox, lat + oy),
                fontsize=7.5, fontweight='bold',
                ha='left', va='center',
                color='#222222', zorder=8,
                arrowprops=dict(arrowstyle='-', color='#666666', lw=0.7)
            )

        ax.set_xlabel('Longitude', fontsize=8.5)
        if title.startswith('2014'):
            ax.set_ylabel('Latitude', fontsize=8.5)

    # Colourbar in its own dedicated axis — never overlaps maps
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('% Negative Posts', fontsize=9.5, labelpad=8)
    cbar.ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1, decimals=0)
    )
    cbar.ax.tick_params(labelsize=8.5)

    path = f'{OUT_DIR}/fig7_spatial_sentiment.png'
    plt.savefig(path)
    plt.close()
    print(f"  ✓  {path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — Polar Seasonal Sentiment Charts
# ════════════════════════════════════════════════════════════════════════════

def fig8_polar_seasonal():
    """
    Four polar (radar) charts — one per city — showing mean sentiment
    across the four seasons (Spring, Summer, Fall, Winter), using
    posts from Eras 2–5 (2014–2025) which contain a 'season' column.
    """
    print("Fig 8 — Polar seasonal sentiment charts...")

    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    frames = []
    for fname in [
        'reddit_climate_crime_2014_2019_sentiment.csv',
        'reddit_climate_crime_2020_2021_sentiment.csv',
        'reddit_climate_crime_2022_2023_sentiment.csv',
        'reddit_climate_crime_2024_2025_sentiment.csv',
    ]:
        df = pd.read_csv(f'{DATA_DIR}/{fname}')
        df['city_clean'] = df['subreddit'].str.lower().map(CITY_MAP)
        df = df[df['city_clean'].isin(CITIES) & df['season'].notna()]
        frames.append(df[['city_clean', 'season', 'sentiment_compound']])

    all_df  = pd.concat(frames, ignore_index=True)
    angles  = np.linspace(0, 2 * np.pi, len(season_order), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5),
                              subplot_kw={'projection': 'polar'})
    fig.suptitle(
        'Figure 8. Seasonal Sentiment Patterns by City (Eras 2–5, 2014–2025)',
        fontsize=12, fontweight='bold', y=1.03
    )

    for ax, city in zip(axes, CITIES):
        city_df = all_df[all_df['city_clean'] == city]
        means   = [city_df[city_df['season'] == s]['sentiment_compound'].mean()
                   for s in season_order]
        means   = [v if not np.isnan(v) else 0 for v in means]
        values  = means + means[:1]

        ax.plot(angles, values, color=CITY_COLORS[city], linewidth=2.5, zorder=3)
        ax.fill(angles, values, color=CITY_COLORS[city], alpha=0.20, zorder=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(season_order, fontsize=9.5)
        ax.set_ylim(-0.15, 0.20)
        ax.set_yticks([-0.1, 0, 0.1, 0.2])
        ax.set_yticklabels(['-0.10', '0', '0.10', '0.20'], fontsize=7)
        ax.axhline(y=0, color='#333333', linewidth=0.8,
                   linestyle='--', alpha=0.5)
        ax.set_title(city, fontsize=11, fontweight='bold',
                     color=CITY_COLORS[city], pad=14)

        for angle, val, season in zip(angles[:-1], means, season_order):
            ax.annotate(f'{val:+.2f}', (angle, val),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=7.5, color=CITY_COLORS[city])

    plt.tight_layout()
    path = f'{OUT_DIR}/fig8_polar_seasonal.png'
    plt.savefig(path)
    plt.close()
    print(f"  ✓  {path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("UrbanSentinel-Prairie — Figure Generation")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)
    fig2_rolling_sentiment()
    fig3_volatility_heatmap()
    fig4_climate_crime_scatter()
    fig5_proportional_maps()
    fig6_era_comparison()
    fig7_spatial_sentiment()
    fig8_polar_seasonal()
    print("\n" + "=" * 60)
    print("All 7 figures generated successfully.")
    print("=" * 60)
