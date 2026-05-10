# ── Cell 1 ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

RED   = '#D7263D'
AMBER = '#F4A261'
GREEN = '#2A9D8F'
BLUE  = '#264653'
GREY  = '#E9ECEF'
print("Libraries loaded.")


# ── Cell 2 ──────────────────────────────────────────────────────────────
df = pd.read_excel('global-data-on-sustainable-energy (1).xlsx')

df.columns = (df.columns.str.strip().str.lower()
    .str.replace(' ','_').str.replace(r'[^\w]','_',regex=True)
    .str.replace(r'_+','_',regex=True))
df['year']   = pd.to_numeric(df['year'], errors='coerce')
df['entity'] = df['entity'].astype(str).str.strip()
df = df.drop_duplicates().drop_duplicates(subset=['entity','year'], keep='first')
df = df.sort_values(['entity','year'])
df = df[(df['year'] >= 2000) & (df['year'] <= 2020)]

num_cols = df.select_dtypes(include='number').columns.tolist()
df[num_cols] = df.groupby('entity')[num_cols].transform(
    lambda x: x.interpolate().ffill().bfill())

miss_ratio = df[num_cols].isnull().mean()
keep_cols  = miss_ratio[miss_ratio <= 0.4].index.tolist()
non_num    = [c for c in df.columns if c not in num_cols]
df = df[non_num + keep_cols]

num_cols2 = df.select_dtypes(include='number').columns.difference(['year'])
for c in num_cols2:
    df[c] = mstats.winsorize(df[c], limits=[0.01, 0.01])

TARGET = 'renewable_energy_share_in_the_total_final_energy_consumption_'
print(f"Dataset: {df.shape[0]} rows | {df.entity.nunique()} countries | {int(df.year.min())}–{int(df.year.max())}")
df.head(3)


# ── Cell 3 ──────────────────────────────────────────────────────────────
from sklearn.metrics import r2_score, mean_absolute_error

comparison = []
for country, grp in df.groupby('entity'):
    sub = grp.sort_values('year')[[TARGET,'year']].dropna()
    if len(sub) < 8:
        continue
    t = sub['year'].values - sub['year'].values.min()
    y = sub[TARGET].values
    lin  = np.polyfit(t, y, 1); y_lin  = np.polyval(lin, t)
    quad = np.polyfit(t, y, 2); y_quad = np.polyval(quad, t)
    comparison.append({
        'entity':       country,
        'mae_linear':   round(mean_absolute_error(y, y_lin),  3),
        'mae_quad':     round(mean_absolute_error(y, y_quad), 3),
        'r2_linear':    round(r2_score(y, y_lin),  3),
        'r2_quad':      round(r2_score(y, y_quad), 3),
        'quad_better':  mean_absolute_error(y,y_quad) < mean_absolute_error(y,y_lin),
    })

cdf = pd.DataFrame(comparison)
print(f"Quadratic improves MAE: {cdf.quad_better.sum()}/{len(cdf)} ({cdf.quad_better.mean():.1%})")
print(f"Median MAE  — Linear: {cdf.mae_linear.median():.3f} | Quad: {cdf.mae_quad.median():.3f}")
print(f"Median R²   — Linear: {cdf.r2_linear.median():.3f}  | Quad: {cdf.r2_quad.median():.3f}")


# ── Cell 4 ──────────────────────────────────────────────────────────────
FORECAST_YEARS = list(range(2021, 2030))
forecast_rows  = []

for country, grp in df.groupby('entity'):
    sub = grp.sort_values('year')[[TARGET,'year']].dropna()
    if len(sub) < 8:
        continue

    t0     = sub['year'].values.min()
    t_norm = sub['year'].values - t0
    y      = sub[TARGET].values

    # Fit quadratic: y = a*t^2 + b*t + c
    quad = np.polyfit(t_norm, y, 2)
    a, b, c = quad

    # Forecast and clip to [0, 100]
    t_future = np.array(FORECAST_YEARS) - t0
    fcast    = np.clip(np.polyval(quad, t_future), 0, 100)

    last_val   = y[-1]  # 2020 observed value
    cum_change = fcast[-1] - last_val

    row = {
        'entity':              country,
        'share_2020':          round(last_val, 2),
        'a_coef':              round(a, 6),
        'b_coef':              round(b, 6),
        'c_coef':              round(c, 6),
        'in_sample_mae':       round(np.mean(np.abs(y - np.polyval(quad, t_norm))), 3),
        'change_2020_to_2029': round(cum_change, 3),
        'at_risk':             cum_change < 0,
    }
    for yr, val in zip(FORECAST_YEARS, fcast):
        row[f'forecast_{yr}'] = round(val, 2)

    forecast_rows.append(row)

fc = pd.DataFrame(forecast_rows)
print(f"Forecasted {len(fc)} countries")
print(f"% with negative trajectory: {fc.at_risk.mean():.1%}")
print(f"Median cumulative change 2020→2029: {fc.change_2020_to_2029.median():.2f} pp")
fc.head(3)


# ── Cell 5 ──────────────────────────────────────────────────────────────
def classify_risk(row):
    if not row['at_risk']:
        return 'LOW RISK – Growth'
    elif row['change_2020_to_2029'] <= -5:
        return 'HIGH RISK – Decline'
    else:
        return 'MEDIUM RISK – Stagnation'

fc['risk_category'] = fc.apply(classify_risk, axis=1)
fc['risk_score']    = np.clip(-fc['change_2020_to_2029'] * 3, 0, 100).round(1)

# GDP group
gdp_mean = df.groupby('entity')['gdp_per_capita'].mean().reset_index()
gdp_mean.columns = ['entity','gdp_per_capita_mean']
gdp_mean['gdp_group'] = pd.qcut(gdp_mean['gdp_per_capita_mean'], 4,
                                  labels=['Low','Mid-Low','Mid-High','High'])
fc = fc.merge(gdp_mean, on='entity', how='left')

print("Risk distribution:")
print(fc['risk_category'].value_counts())


# ── Cell 6 ──────────────────────────────────────────────────────────────
risk_counts = fc['risk_category'].value_counts().reindex(
    ['HIGH RISK – Decline','MEDIUM RISK – Stagnation','LOW RISK – Growth'])
colors_risk = [RED, AMBER, GREEN]

fig, axes = plt.subplots(1, 2, figsize=(13,5))

ax = axes[0]
bars = ax.barh(risk_counts.index, risk_counts.values, color=colors_risk, edgecolor='white', height=0.55)
for bar, v in zip(bars, risk_counts.values):
    ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, str(v), va='center', fontweight='bold')
ax.set_xlabel('Number of Countries')
ax.set_title('Transition Risk Classification\n(Quadratic Forecast 2021–2029)', fontweight='bold')

ax = axes[1]
by_gdp = fc.groupby(['gdp_group','risk_category']).size().unstack(fill_value=0)
by_gdp = by_gdp.reindex(index=['Low','Mid-Low','Mid-High','High'],
    columns=['HIGH RISK – Decline','MEDIUM RISK – Stagnation','LOW RISK – Growth'], fill_value=0)
by_gdp.plot(kind='bar', ax=ax, color=colors_risk, edgecolor='white', stacked=True)
ax.set_title('Risk Category by GDP Group', fontweight='bold')
ax.set_xlabel('GDP Group'); ax.set_ylabel('Countries')
ax.legend(title='Risk', fontsize=7, bbox_to_anchor=(1,1))
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('fig1_risk_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Cell 7 ──────────────────────────────────────────────────────────────
at_risk_df  = fc[fc['at_risk']==True].sort_values('change_2020_to_2029').head(20)
bar_colors  = [RED if r=='HIGH RISK – Decline' else AMBER for r in at_risk_df['risk_category']]

fig, ax = plt.subplots(figsize=(11,7))
bars = ax.barh(at_risk_df['entity'], at_risk_df['risk_score'], color=bar_colors, edgecolor='white')
for bar, row in zip(bars, at_risk_df.itertuples()):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f'{row.change_2020_to_2029:+.1f}pp', va='center', fontsize=8)
ax.set_xlabel('Risk Score'); ax.invert_yaxis()
ax.set_title('Top 20 Countries at Transition Risk\n(Quadratic Model, 2020→2029)', fontweight='bold')
patch_r = mpatches.Patch(color=RED,   label='HIGH RISK – Decline (> −5 pp)')
patch_a = mpatches.Patch(color=AMBER, label='MEDIUM RISK – Stagnation (0 to −5 pp)')
ax.legend(handles=[patch_r, patch_a], loc='lower right')
plt.tight_layout()
plt.savefig('fig2_top20_at_risk.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Cell 8 ──────────────────────────────────────────────────────────────
top6 = fc[fc['risk_category']=='HIGH RISK – Decline'].sort_values('change_2020_to_2029').head(6)

fig, axes = plt.subplots(2, 3, figsize=(15,8))
axes = axes.flatten()

for i, (_, row) in enumerate(top6.iterrows()):
    ax = axes[i]
    hist = df[df['entity']==row['entity']].sort_values('year')[['year',TARGET]].dropna()
    t0   = hist['year'].min()

    # Historical points
    ax.plot(hist['year'], hist[TARGET], color=BLUE, lw=2, marker='o', ms=3, label='Historical')

    # Quadratic fit line over training range (smooth curve)
    t_fit = np.linspace(0, 20, 200)
    y_fit = np.clip(np.polyval([row['a_coef'], row['b_coef'], row['c_coef']], t_fit), 0, 100)
    ax.plot(t_fit + t0, y_fit, color=GREY, lw=1.5, label='Quadratic fit')

    # Forecast extension (curved — NOT linear)
    t_fcast = np.linspace(20, 29, 150)
    y_fcast = np.clip(np.polyval([row['a_coef'], row['b_coef'], row['c_coef']], t_fcast), 0, 100)
    ax.plot(t_fcast + t0, y_fcast, color=RED, lw=2.5, ls='--', label='Forecast (quadratic)')
    ax.axvspan(2020, 2029, alpha=0.07, color=RED)

    ax.set_title(f"{row['entity']}\nΔ2020→2029: {row['change_2020_to_2029']:+.1f}pp",
                 fontsize=9, fontweight='bold')
    ax.set_ylabel('Renewable Share (%)'); ax.set_xlabel('Year')
    ax.legend(fontsize=7); ax.set_xlim(1999, 2030)

plt.suptitle('Quadratic Forecast Trajectories – Top 6 High-Risk Countries',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig3_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Cell 9 ──────────────────────────────────────────────────────────────
cmap = {'HIGH RISK – Decline': RED, 'MEDIUM RISK – Stagnation': AMBER, 'LOW RISK – Growth': GREEN}

fig, axes = plt.subplots(1, 2, figsize=(13,5))

ax = axes[0]
ax.hist(fc['a_coef'], bins=40, color=BLUE, edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', lw=1.5, ls='--', label='a=0 (linear)')
ax.axvline(fc['a_coef'].median(), color='black', lw=1.5, label=f'Median={fc["a_coef"].median():.4f}')
ax.set_xlabel('Quadratic Coefficient (a)')
ax.set_title('Distribution of Quadratic Coefficient\n(a>0: accelerating | a<0: decelerating)', fontweight='bold')
ax.legend()

ax = axes[1]
for cat, grp in fc.groupby('risk_category'):
    ax.scatter(grp['a_coef'], grp['change_2020_to_2029'],
               c=cmap[cat], label=cat, alpha=0.7, s=40, edgecolors='white', lw=0.4)
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.axvline(0, color='black', lw=0.8, ls='--')
ax.set_xlabel('Quadratic Coefficient (a)')
ax.set_ylabel('Cumulative Change 2020→2029 (pp)')
ax.set_title('Curve Direction vs. Forecast Outcome\nby Risk Category', fontweight='bold')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('fig4_quadratic_coef.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Cell 10 ──────────────────────────────────────────────────────────────
display_cols = ['entity','share_2020','forecast_2024','forecast_2027','forecast_2029',
                'change_2020_to_2029','risk_score','risk_category','a_coef','in_sample_mae','gdp_group']

risk_register = fc[display_cols].copy()
risk_register.columns = ['Country','Share_2020','Forecast_2024','Forecast_2027','Forecast_2029',
                          'Change_2020_2029_pp','Risk_Score','Risk_Category','Quad_Coef_a',
                          'InSample_MAE','GDP_Group']
risk_register = risk_register.sort_values('Risk_Score', ascending=False)
risk_register.to_csv('transition_risk_register_quadratic.csv', index=False)
print("Saved: transition_risk_register_quadratic.csv")
print()
print("=== HIGH RISK countries ===")
print(risk_register[risk_register['Risk_Category']=='HIGH RISK – Decline'].to_string(index=False))


