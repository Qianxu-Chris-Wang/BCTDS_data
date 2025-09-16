import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from datetime import datetime
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# ---------------------- Preamble: Matplotlib Font Settings ----------------------
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['text.usetex'] = True

# General publication-quality settings
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['figure.titlesize'] = 16

# grab and save data from current folder
script_dir = os.path.dirname(os.path.abspath(__file__))
fig_save_dir = os.path.join(script_dir, 'analysis_plots')
csv_data_dir = os.path.join(script_dir, 'S_param_csv_data')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

# ---------------------- Part 1: Generate S-parameter figure and save ----------------------
sample_type = 'sapphire'
color2 = 'orange'

S11_measured_path = os.path.join(csv_data_dir, f"SJ_waveguide_room_{sample_type}_S11.csv")
S11_sim_path = os.path.join(csv_data_dir, f"sim_{sample_type}_S11.csv")
S21_measured_path = os.path.join(csv_data_dir, f"SJ_waveguide_room_{sample_type}_S21.csv")
S21_sim_path = os.path.join(csv_data_dir, f"sim_{sample_type}_S21.csv")

plt.figure(figsize=(4.1, 2.5))

# S11 sim
if os.path.exists(S11_sim_path):
    df = pd.read_csv(S11_sim_path)
    freq_ghz = df.iloc[:, 0]
    s11_db = df.iloc[:, 1]
    plt.plot(freq_ghz, s11_db, label=r"$S_{11}$ Simulation", color=color2, linestyle='--')

# S11 measured
if os.path.exists(S11_measured_path):
    df_measured = pd.read_csv(S11_measured_path)
    freq_measured_ghz = df_measured.iloc[:, 0] / 1e9
    s11_measured_db = df_measured.iloc[:, 1]
    plt.plot(freq_measured_ghz, s11_measured_db, linewidth=2, label=r"$S_{11}$ Measurement", color=color2)

# S21 sim
if os.path.exists(S21_sim_path):
    df = pd.read_csv(S21_sim_path)
    freq_ghz = df.iloc[:, 0]
    s21_db = df.iloc[:, 1]
    plt.plot(freq_ghz, s21_db, label=r"$S_{21}$ Simulation", color='cornflowerblue', linestyle='--')

# S21 measured
if os.path.exists(S21_measured_path):
    df_measured = pd.read_csv(S21_measured_path)
    freq_measured_ghz = df_measured.iloc[:, 0] / 1e9
    s21_measured_db = df_measured.iloc[:, 1]
    plt.plot(freq_measured_ghz, s21_measured_db, linewidth=2, label=r"$S_{21}$ Measurement", color='cornflowerblue')

# Labels and title
plt.xlabel(r"Frequency (GHz)")
plt.ylabel("Magnitude (dB)")

# Plot formatting
plt.legend()
plt.xlim(2, 6)
plt.ylim(-70, 5)

sparam_png = os.path.join(fig_save_dir, f'{sample_type}_S11_and_S21.png')
plt.savefig(sparam_png, bbox_inches='tight', pad_inches=0.01, dpi=600)
# plt.show()

save_fig = True

fig = plt.figure()  # adjust to your desired figure size

img_path = os.path.join(fig_save_dir, "Fig_waveguide_design_unlabled.png")
img = mpimg.imread(img_path)

ax = fig.add_axes([0, 0, 1, 1])   # use full figure space
ax.imshow(img, aspect='equal')     # preserve original aspect ratio
ax.axis('off')

# Label (a) on the main axes
ax.text(0.055, 0.99, r'(a)', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='right', color='black')

# ---------------- Add second figure overlay ----------------
img_path_2 = os.path.join(fig_save_dir, f"{sample_type}_S11_and_S21.png")
img2 = mpimg.imread(img_path_2)

# Place new axes covering the bottom portion of the figure
ax2 = fig.add_axes([0, 0, 1, 0.45])  # [left, bottom, width, height]
ax2.imshow(img2, aspect='equal')
ax2.axis('off')

# Label (b) on the overlay axes so it stays visible
ax2.text(0.035, 1.05, r'(b)', transform=ax2.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='right', color='black', zorder=100)

# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, "Fig_waveguide_design.png")
if save_fig:
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Figure saved to: {file_path}")

plt.close()
