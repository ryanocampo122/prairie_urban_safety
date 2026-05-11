"""
Figure 1 — prairie-urban-safety Pipeline Architecture Diagram
Publication-quality flowchart showing the seven processing modules (M1-M7)
with data sources at top and outputs at bottom. Black and white scheme
to match article styling.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Colour scheme — black and white only ─────────────────────────────────────
C_SOURCE   = '#1a1a1a'   # near-black  — input source boxes
C_MODULE   = '#333333'   # dark grey   — processing module boxes
C_OUTPUT   = '#555555'   # mid grey    — output boxes
C_ARROW    = '#222222'   # arrow colour
C_LABEL    = '#ffffff'   # text on dark boxes
C_TITLE    = '#000000'   # section labels above lanes
C_BORDER   = '#000000'   # box borders
C_LANE_BG  = '#f7f7f7'   # very light grey lane backgrounds
C_LANE_BD  = '#cccccc'   # lane border

# ── Helper: rounded rectangle ─────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel=None,
        fc='#333333', tc='white', fontsize=9, radius=0.18, lw=1.2):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f'round,pad=0.0,rounding_size={radius}',
        linewidth=lw, edgecolor=C_BORDER, facecolor=fc, zorder=4
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.12, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=tc,
                fontfamily='DejaVu Sans', zorder=5)
        ax.text(x, y - 0.18, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color=tc, alpha=0.85,
                fontfamily='DejaVu Sans', zorder=5)
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=tc,
                fontfamily='DejaVu Sans', zorder=5)

# ── Helper: arrow ─────────────────────────────────────────────────────────────
def arrow(ax, x1, y1, x2, y2, lw=1.5):
    ax.annotate('',
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=C_ARROW,
            lw=lw, mutation_scale=14,
        ), zorder=3)

# ── Helper: lane background ───────────────────────────────────────────────────
def lane(ax, y, h, label):
    rect = FancyBboxPatch(
        (0.3, y - h/2), 13.4, h,
        boxstyle='round,pad=0.0,rounding_size=0.1',
        linewidth=0.8, edgecolor=C_LANE_BD,
        facecolor=C_LANE_BG, zorder=1
    )
    ax.add_patch(rect)
    ax.text(0.62, y, label, ha='center', va='center',
            fontsize=7.5, color='#666666', fontweight='bold',
            fontfamily='DejaVu Sans', rotation=90, zorder=2)

# ════════════════════════════════════════════════════════════════════════════
#  LANE BACKGROUNDS
# ════════════════════════════════════════════════════════════════════════════
lane(ax, 9.0, 1.3,  'INPUT\nSOURCES')
lane(ax, 7.3, 1.1,  'ACQUISITION')
lane(ax, 5.8, 0.9,  'PROCESSING')
lane(ax, 4.4, 0.9,  'NORMALISATION')
lane(ax, 3.0, 0.9,  'ANALYSIS')
lane(ax, 1.55, 0.9, 'OUTPUTS')

# ════════════════════════════════════════════════════════════════════════════
#  ROW 1 — Input sources  (y = 9.0)
# ════════════════════════════════════════════════════════════════════════════
sources = [
    (3.0,  9.0, 'Environment & Climate\nChange Canada (ECCC)', 'Historical Climate\nData Portal'),
    (7.0,  9.0, 'Statistics Canada\nUCR Survey', 'Table 35-10-0177-01\n2010–2024'),
    (11.0, 9.0, 'Reddit API /\nPushshift Archive', 'r/Edmonton, r/Saskatoon\nr/Regina, r/Winnipeg + provincial'),
]
for x, y, label, sub in sources:
    box(ax, x, y, 3.4, 0.95, label, sub,
        fc=C_SOURCE, tc='white', fontsize=8.5)

# ════════════════════════════════════════════════════════════════════════════
#  ROW 2 — M1 / M2 / M3  (y = 7.3)  Acquisition
# ════════════════════════════════════════════════════════════════════════════
modules_r2 = [
    (3.0,  7.3, 'M1', 'Climate Acquisition',    'Multi-station averaging\nVariable imputation'),
    (7.0,  7.3, 'M2', 'Crime Integration',       'UCR violation mapping\nAnnual aggregation'),
    (11.0, 7.3, 'M3', 'Reddit Corpus Collection','Keyword classification\n2009–2025'),
]
for x, y, tag, label, sub in modules_r2:
    box(ax, x, y, 3.4, 0.90,
        f'{tag} — {label}', sub,
        fc=C_MODULE, tc='white', fontsize=8.5)

# ════════════════════════════════════════════════════════════════════════════
#  ROW 3 — M4  (y = 5.8)  Sentiment scoring — full width
# ════════════════════════════════════════════════════════════════════════════
box(ax, 7.0, 5.8, 10.8, 0.75,
    'M4 — Sentiment Scoring',
    'VADER lexicon-based scoring (−1 to +1)  |  Categorical classification (negative / neutral / positive)  |  Crime-relevance filtering',
    fc=C_MODULE, tc='white', fontsize=8.5)

# ════════════════════════════════════════════════════════════════════════════
#  ROW 4 — M5  (y = 4.4)  Normalisation — full width
# ════════════════════════════════════════════════════════════════════════════
box(ax, 7.0, 4.4, 10.8, 0.75,
    'M5 — Normalisation & Bias Mitigation',
    'Per-post weighting  |  Per-user aggregation  |  Within-city z-score anomaly scoring  |  City-month seasonal adjustment',
    fc=C_MODULE, tc='white', fontsize=8.5)

# ════════════════════════════════════════════════════════════════════════════
#  ROW 5 — M6  (y = 3.0)  Statistical testing — full width
# ════════════════════════════════════════════════════════════════════════════
box(ax, 7.0, 3.0, 10.8, 0.75,
    'M6 — Statistical Analysis',
    'Kruskal-Wallis  |  Mann-Whitney U (Bonferroni)  |  Wilcoxon signed-rank  |  Pearson r  |  OLS regression',
    fc=C_MODULE, tc='white', fontsize=8.5)

# ════════════════════════════════════════════════════════════════════════════
#  ROW 6 — Outputs  (y = 1.55)
# ════════════════════════════════════════════════════════════════════════════
outputs = [
    (2.2,  1.55, 'Climate-Crime\nCorrelation Results',   'Pearson r, OLS coefficients\nper city (n = 15)'),
    (5.4,  1.55, 'Era-Stratified\nSentiment Series',     'Monthly sentiment anomalies\n2009–2025'),
    (8.6,  1.55, 'Geospatial\nVisualisations',           'Maps, polar charts\nvolatility heatmaps'),
    (11.8, 1.55, 'Open Data\nArchive',                   'GitHub + Zenodo\nMIT / CC BY 4.0'),
]
for x, y, label, sub in outputs:
    box(ax, x, y, 2.8, 0.85, label, sub,
        fc=C_OUTPUT, tc='white', fontsize=8.2)

# ════════════════════════════════════════════════════════════════════════════
#  ARROWS — source → acquisition
# ════════════════════════════════════════════════════════════════════════════
for x in [3.0, 7.0, 11.0]:
    arrow(ax, x, 8.52, x, 7.76)

# acquisition → M4 (all three converge)
# Edmonton → M4
arrow(ax, 3.0, 6.85, 3.0, 6.18)
ax.annotate('', xy=(6.4, 5.99), xytext=(3.0, 6.18),
    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5,
                    mutation_scale=14,
                    connectionstyle='arc3,rad=0.0'), zorder=3)
# Statistics Canada → M4
arrow(ax, 7.0, 6.85, 7.0, 6.18)
# Reddit → M4
arrow(ax, 11.0, 6.85, 11.0, 6.18)
ax.annotate('', xy=(7.6, 5.99), xytext=(11.0, 6.18),
    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5,
                    mutation_scale=14,
                    connectionstyle='arc3,rad=0.0'), zorder=3)

# M4 → M5
arrow(ax, 7.0, 5.42, 7.0, 4.78)

# M5 → M6
arrow(ax, 7.0, 4.02, 7.0, 3.38)

# M6 → outputs (fan out)
arrow_targets = [2.2, 5.4, 8.6, 11.8]
for xt in arrow_targets:
    ax.annotate('', xy=(xt, 1.98), xytext=(7.0, 2.62),
        arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.4,
                        mutation_scale=13,
                        connectionstyle='arc3,rad=0.0'), zorder=3)

# ════════════════════════════════════════════════════════════════════════════
#  ALSO: M7 label — embedded in outputs lane as a label
# ════════════════════════════════════════════════════════════════════════════
ax.text(7.0, 0.78, 'M7 — Geospatial Visualisation module produces Figures 2–8',
        ha='center', va='center', fontsize=7.8, color='#555555',
        style='italic', fontfamily='DejaVu Sans')

# ════════════════════════════════════════════════════════════════════════════
#  TITLE
# ════════════════════════════════════════════════════════════════════════════
ax.text(7.0, 9.82,
        'Figure 1. prairie-urban-safety Pipeline Architecture (Modules M1–M7)',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color=C_TITLE, fontfamily='DejaVu Sans')
ax.text(7.0, 9.57,
        'Three open data sources are integrated through seven sequential processing modules to produce climate-crime correlations, era-stratified sentiment series, geospatial visualisations, and an open data archive.',
        ha='center', va='center', fontsize=8, color='#444444',
        fontfamily='DejaVu Sans')

plt.tight_layout(pad=0.3)
plt.savefig('/home/claude/figures/fig1_pipeline_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved fig1_pipeline_architecture.png")
