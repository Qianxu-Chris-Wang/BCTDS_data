import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import time
import matplotlib
# matplotlib.use('Agg')  # Use the 'Agg' backend, which does not require a display environment.
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.image as mpimg

# #### Preamble
# # Set the font globally to Helvetica
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

##############################################################################
# 1. General matplotlib settings
##############################################################################
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.titlesize'] = 16

script_dir = os.path.dirname(os.path.abspath(__file__))

fig_save_dir = os.path.join(script_dir, 'analysis_plots')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

save_fig = True

fig = plt.figure()  # adjust to your desired figure size

img_path = os.path.join(fig_save_dir, "Fig_overview_no_lable.png")
img = mpimg.imread(img_path)

ax = fig.add_axes([0, 0, 1, 1])   # use full figure space
ax.imshow(img, aspect='equal')   # preserve original aspect ratio
ax.axis('off')

# Add label relative to axes (which still map [0,1] x [0,1] here)

ab_x = 0.05
ac_y = 0.98
i_ii_y = 0.945
abc_size = 30
i_ii_size = 27

ax.text(ab_x, ac_y, r'\textbf{a}', transform=ax.transAxes, fontsize=abc_size, fontweight='bold', va='top', ha='right', color='black')
ax.text(ab_x, 0.49, r'\textbf{b}', transform=ax.transAxes, fontsize=abc_size, fontweight='bold', va='top', ha='right', color='black')
ax.text(0.52, ac_y, r'\textbf{c}', transform=ax.transAxes, fontsize=abc_size, fontweight='bold', va='top', ha='right', color='black')

ax.text(0.74, i_ii_y, r'\textbf{i}', transform=ax.transAxes, fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')
ax.text(0.96, i_ii_y, r'\textbf{ii}', transform=ax.transAxes, fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"Fig_overview.png")
if save_fig:
    plt.savefig(file_path, dpi=350, bbox_inches='tight', pad_inches=0)
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()
