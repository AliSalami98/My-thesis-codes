import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------
# User controls
# ------------------------------
sweep_key   = 'Theater'   # choose: 'pcharged', 'omega', or 'Theater'
base_dir    = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\Exergy\first patch'
csv_path    = os.path.join(base_dir, f"Results_{sweep_key}.csv")
save_dir    = os.path.dirname(csv_path)
save_format = 'eps'

# Optionally restrict which levels to plot (e.g., [30,50,70] for pcharged; None = plot all)
include_levels = None  # e.g., [30, 50, 70] or [100, 180, 240] or [600, 700, 800]

# ------------------------------
# Load
# ------------------------------
df = pd.read_csv(csv_path, sep=';')

# Checks
needed = ['Pr', 'Ex_eff [%]']
for c in needed:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# If the sweep column doesn't exist, fall back to contiguous 5-row blocks with typical labels
if sweep_key not in df.columns:
    n_per_seg = 5
    labels_map = {
        'omega':    [100, 180, 240],
        'pcharged': [30, 50, 70],
        'Theater':  [600, 700, 800],
    }
    values = labels_map.get(sweep_key, None)
    if values is None:
        raise ValueError(f"No column '{sweep_key}' in CSV and no fallback labels defined for it.")
    # Build synthetic sweep column
    sweep_col = []
    for i, v in enumerate(values):
        start, end = i*n_per_seg, (i+1)*n_per_seg
        sweep_col.extend([v] * len(df.iloc[start:end]))
    if len(sweep_col) != len(df):
        raise ValueError("Fallback segmentation length does not match dataframe length.")
    df[sweep_key] = sweep_col

# ------------------------------
# Pick which levels to plot
# ------------------------------
try:
    all_levels = sorted(df[sweep_key].unique(), key=lambda x: float(x))
except Exception:
    all_levels = sorted(df[sweep_key].unique())

levels = all_levels if include_levels is None else [lv for lv in all_levels if lv in include_levels]
if not levels:
    raise ValueError("No levels to plot (check include_levels).")

# ------------------------------
# Plot Exergy Efficiency vs Pr for each level
# ------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

for lv in levels:
    seg = df[df[sweep_key] == lv].copy()
    if seg.empty:
        continue
    seg = seg.sort_values('Pr')
    ax.plot(seg['Pr'].values, seg['Ex_eff [%]'].values, marker='o', label=f'{sweep_key}={lv}')

ax.set_xlabel(r'Pressure Ratio $p_\mathrm{r}$ [-]', fontsize=12)
ax.set_ylabel('Exergy Efficiency [%]', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(title=sweep_key, fontsize=10, title_fontsize=10, frameon=False)
ax.set_title(f'Exergy Efficiency vs Pressure Ratio — grouped by {sweep_key}', fontsize=13)
plt.tight_layout()

# Save
out_name = f"Exergy_eff_vs_Pr_by_{sweep_key}.{save_format}"
save_path = os.path.join(save_dir, out_name)
plt.savefig(save_path, format=save_format, bbox_inches='tight')
plt.show()

print(f"Saved: {save_path}")
