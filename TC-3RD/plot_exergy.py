import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
csv_path = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\Exergy\Results_Theater.csv'
df = pd.read_csv(csv_path, sep=',')
save_dir = os.path.dirname(csv_path)

# Components
# components = ['Edest_c [kW]', 'Edest_k [kW]', 'Edest_kr [kW]', 'Edest_r [kW]',
#               'Edest_hr [kW]', 'Edest_h [kW]', 'Edest_e [kW]']
# component_labels = ['Compressor', 'Cooler', 'Regenerator (kr)', 'Regenerator',
#                     'Regenerator (hr)', 'Heater', 'Expansion']

components = ['Edest_c [kW]', 'Edest_k [kW]', 'Edest_r [kW]',
             'Edest_h [kW]', 'Edest_e [kW]']
component_labels = ['Compressor', 'Cooler', 'Regenerator',
                     'Heater', 'Expansion']

component_colors = ['b', 'cyan', 'g', 'orange', 'r']  # Matches your line plot


# Split data
segments = {
    "omega 100": df.iloc[0:5].reset_index(drop=True),
    "omega 180": df.iloc[5:10].reset_index(drop=True),
    "omega 240": df.iloc[10:15].reset_index(drop=True)
}

# Plot each segment
for title, segment_df in segments.items():
    fig, ax1 = plt.subplots(figsize=(8, 6))  # Set a reasonable figure size

    # --- Bar settings ---
    bar_width = 0.4
    x = np.arange(len(segment_df))
    labels = [f"{pr:.1f}" for pr in segment_df['Pr']]  # Clean x-tick labels

    # --- Plot stacked bars ---
    bottoms = np.zeros(len(segment_df))
    bars = []
    for comp, label, color in zip(components, component_labels, component_colors):
        bar = ax1.bar(x, segment_df[comp], bar_width, bottom=bottoms, label=label, color = color)
        bottoms += segment_df[comp]
        bars.append(bar)

    # --- Primary y-axis settings ---
    ax1.set_xlabel(r'Pressure Ratio $p_\text{r}$ [-]', fontsize=14)
    ax1.set_ylabel(r'Exergy Destruction $E_\text{dest}$ [kW]', fontsize=14)
    ax1.set_xticks(x)
    # Add margin above tallest bar
    ymax = bottoms.max()
    ax1.set_ylim(top=ymax * 1.5)  # 5% headroom
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    # --- Secondary y-axis for Exergy Efficiency ---
    ax2 = ax1.twinx()
    ax2.plot(x, segment_df['Ex_eff [%]'], 'k--o', label='Exergy Efficiency [%]', markersize=6)
    ax2.set_ylabel('Exergy Efficiency [%]', fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)

    # --- Combined Legend ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2, labels1 + labels2,
        loc='upper center',
        ncol=3, fontsize=11, frameon=False
    )

    # --- Title & Layout ---
    # plt.title(f'Exergy Breakdown and Efficiency – {title}', fontsize=15)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.88)  # To make room for the legend below

    # --- Save & Show ---
    save_path = os.path.join(save_dir, f"stacked_exergy_efficiency_{title.replace(' ', '_')}.eps")
    plt.savefig(save_path, format='eps')
    plt.show()
