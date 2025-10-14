import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# 1) Point to your installed file (from your screenshot)
path = r"C:\Users\ali.salame\AppData\Local\Microsoft\Windows\Fonts\CHARTERBT-ROMAN.OTF"
# (add the Bold/Italic too if you use them)
# fm.fontManager.addfont(r"...\CHARTERBT-BOLD.OTF")
# fm.fontManager.addfont(r"...\CHARTERBT-ITALIC.OTF")

# 2) Register and use the exact internal name
fm.fontManager.addfont(path)
prop = fm.FontProperties(fname=path)
mpl.rcParams["font.family"] = prop.get_name()   # e.g., "Bitstream Charter"
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10

# ------------------------------
# User controls
# ------------------------------
sweep_key = 'pcharged'   # choose: 'omega', 'pcharged', 'Theater'
base_dir  = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\Exergy\first patch'
csv_path  = os.path.join(base_dir, f"Results_{sweep_key}.csv")
save_dir  = os.path.dirname(csv_path)

target_pr = 1.375   # desired pressure ratio
save_format = 'eps'

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv(csv_path, sep=';')

components       = ['Edest_cold', 'Edest_hot', 'Edest_reg', 'Edest_cylinder']
component_labels = ['Cold side', 'Hot side', 'Regenerator', 'Cylinder']
component_colors = ['b', 'r', 'g', 'orange']

# check
missing = [c for c in components + ['Pr'] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# ------------------------------
# Pick row closest to target_pr
# ------------------------------
row = df.loc[(df['Pr'] - target_pr).abs().idxmin()]

values = np.array([float(row[c]) for c in components])
values = np.clip(values, 0, None)
total  = values.sum()
if total <= 0:
    raise ValueError("Total destroyed exergy is zero or negative.")

# ------------------------------
# Plot
# ------------------------------
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.pie(
    values,
    labels=component_labels,
    colors=component_colors,
    autopct=lambda p: f'{p:.1f}%',
    startangle=90
)
ax.axis('equal')
# ax.set_title(f'{sweep_key} — Exergy destruction breakdown\nat Pr ≈ {target_pr} (picked {row["Pr"]:.3f})', fontsize=12)

# save
out_name = f"Exergy_pie_{sweep_key}_Pr{row['Pr']:.3f}.{save_format}"
save_path = os.path.join(save_dir, out_name)
plt.savefig(save_path, format=save_format, bbox_inches='tight')
plt.show()

print(f"Saved: {save_path}")
