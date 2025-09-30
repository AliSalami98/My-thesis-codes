import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------
# User controls
# ------------------------------
sweep_key = 'Theater'   # <- choose: 'omega', 'pcharged', or 'Theater'
base_dir  = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\Exergy\first patch'
csv_path  = os.path.join(base_dir, f"Results_{sweep_key}.csv")
save_dir  = os.path.dirname(csv_path)

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv(csv_path, sep=';')

# ------------------------------
# Plot config (components)
# ------------------------------
components = ['Edest_cold', 'Edest_hot', 'Edest_reg', 'Edest_cylinder']
component_labels = ['Cold side', 'Hot side', 'Regenerator', 'Cylinder']
component_colors = ['b', 'r', 'g', 'orange']

# ------------------------------
# Build segments
# ------------------------------
segments = {}

if sweep_key in df.columns:
    # group by the chosen column if it exists
    try:
        keys = sorted(df[sweep_key].unique(), key=lambda x: float(x))
    except Exception:
        keys = sorted(df[sweep_key].unique())

    for k in keys:
        seg = df[df[sweep_key] == k].copy().reset_index(drop=True)
        title = f'{sweep_key} {k}'
        segments[title] = seg
else:
    # fallback: contiguous slicing
    step = 5
    labels_map = {
        'omega':     ['100', '180', '240'],
        'pcharged':  ['30', '50', '70'],
        'Theater':   ['600', '700', '800']
    }
    values = labels_map.get(sweep_key, [f'{i}' for i in range(len(df)//step)])

    for i, val in enumerate(values):
        start, end = i * step, (i + 1) * step
        seg = df.iloc[start:end].reset_index(drop=True)
        title = f'{sweep_key} {val}'
        segments[title] = seg

# ------------------------------
# Plot each segment
# ------------------------------
for title, segment_df in segments.items():
    segment_df = segment_df.sort_values('Pr').reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    x = np.arange(len(segment_df))
    labels = [f"{pr:.1f}" for pr in segment_df['Pr']]
    bottoms = np.zeros(len(segment_df))

    # stacked bars
    for comp, lab, col in zip(components, component_labels, component_colors):
        ax1.bar(x, segment_df[comp].values, 0.4, bottom=bottoms, label=lab, color=col)
        bottoms += segment_df[comp].values

    # primary axis
    ax1.set_xlabel(r'Pressure ratio $p_\mathrm{r}$ [-]', fontsize=14)
    ax1.set_ylabel(r'Exergy destruction $E_\mathrm{dest}$ [kW]', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    if len(segment_df) > 0:
        ax1.set_ylim(top=bottoms.max() * 1.15)

    # secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, segment_df['Ex_eff [%]'].values, 'k--o',
             label='Exergy efficiency [%]', markersize=6)
    ax2.set_ylabel('Exergy efficiency [%]', fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)

    # legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper center', ncol=3, fontsize=11, frameon=False)

    plt.title(f'Exergy Breakdown & Efficiency — {title}', fontsize=15)
    plt.tight_layout()

    # save
    safe_title = title.replace(' ', '_').replace(':', '_')
    out_name = f"Exergy_efficiency_{safe_title}.eps"
    save_path = os.path.join(save_dir, out_name)
    plt.savefig(save_path, format='eps')
    plt.show()

    print(f"Saved: {save_path}")
