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
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.titlesize'] = 20

def compute_fft_GHz(trace, time_spacing_us):
    time_spacing = time_spacing_us * 1e-6

    # FFT and frequencies
    fft_vals = np.fft.fft(trace)
    freqs = np.fft.fftfreq(len(trace), d=time_spacing)
    
    # Convert frequency to GHz
    freqs_GHz = freqs / 1e9
    fft_magnitude = np.abs(np.fft.fftshift(fft_vals))
    freqs_GHz_shifted = np.fft.fftshift(freqs_GHz)
    return freqs_GHz_shifted, fft_magnitude

def find_FWHM_and_plot(pulse_width=308, save_plot=False):
    pulse_width_us = pulse_width / 9830.4 # conversion using QICK synthesis clock frequency
    time_array = np.linspace(0, 1, 512)
    pulse = np.zeros_like(time_array)
    start_idx = int((0.5 - pulse_width_us / 2) * 512)
    end_idx = int((0.5 + pulse_width_us / 2) * 512)
    pulse[start_idx:end_idx] = 20
    
    freqs_GHz, fft_mag = compute_fft_GHz(pulse, 1/512)
    abs_diff = np.abs(fft_mag - np.max(fft_mag) / 2)
    FWHM_idx_0, FWHM_idx_1 = np.argsort(abs_diff)[:2]

    if save_plot:
        fig, ax = plt.subplots(figsize=(2, 3))
        ax.plot(fft_mag, freqs_GHz, c='indigo')
        
        plt.vlines(x=fft_mag[FWHM_idx_0], ymin=freqs_GHz[FWHM_idx_0], ymax=freqs_GHz[FWHM_idx_1], colors='indigo')

        x_pos = fft_mag[FWHM_idx_0]
        y_start = freqs_GHz[FWHM_idx_0]+ 0.003
        y_end = freqs_GHz[FWHM_idx_1] - 0.003

        # Plot arrow-like vertical line with variable thickness (linewidth) and arrowheads
        plt.annotate(
            '', 
            xy=(x_pos, y_end), 
            xytext=(x_pos, y_start),
            arrowprops=dict(
                arrowstyle='<->',      # double-headed arrow
                color='indigo',
                linewidth=2            # thickness of the arrow
            )
        )

        ax.set_xlabel("Magnitude [arb.]")
        ax.set_ylim([-0.1, 0.1])
        ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
        ax.set_ylabel("Frequency [GHz]")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_label_coords(0.6, 1.2)

        fig_save_dir = os.path.join(script_dir, 'analysis_plots')
        os.makedirs(fig_save_dir, exist_ok=True)
        file_path = os.path.join(fig_save_dir, f"bandwidth_only_full.png")
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {file_path}")

    return np.abs(freqs_GHz[FWHM_idx_1]- freqs_GHz[FWHM_idx_0])

script_dir = os.path.dirname(os.path.abspath(__file__))

fig_save_dir = os.path.join(script_dir, 'analysis_plots')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

matrix_data_dir = os.path.join(script_dir, 'matrix_npy_save_folder')

save_fig = True

fig = plt.figure(figsize=(6, 12))

# Settings
row_names = ['row1', 'row2', 'row3', 'row4']
row_heights = [0.3, 0.15, 0.15, 0.15]     # Custom height per row
v_gaps = [-0.16, 0.06, 0.06]               # Gaps *between* rows (len = num_rows - 1)

# Compute bottom positions from top to bottom
bottoms = []
current_bottom = 1 - row_heights[0] - 0.05  # Initial offset from top
bottoms.append(current_bottom)

for i in range(1, len(row_names)):
    current_bottom -= (row_heights[i - 1] + v_gaps[i - 1])
    bottoms.append(current_bottom)

# Unchanged left/width config
positions = {
    'row1': [0, 1],
    'row2': [0.15, 0.7],
    'row3': [0.15, 0.875],
    'row4': [0.15, 0.875],
}

# Create axes
axes = {}
for i, row in enumerate(row_names):
    left, width = positions[row]
    height = row_heights[i]
    bottom = bottoms[i]
    axes[row] = fig.add_axes([left, bottom, width, height])

# # Create axes dictionary
# axes = {}
# for i, key in enumerate(positions):
#     left, width = positions[key]
#     axes[key] = fig.add_axes([left, bottoms[i], width, row_height])
#     # axes[key].set_title(f"{key.upper()} Plot")

# Example usage
pulse_start_idx = 11
pulse_end_idx = 28
line_plot_line_width = 0.8
pulse_mark_line_width = 1.5
pulse_mark_line_color = 'black'
pulse_mark_line_style = '--'
display_freq_range = [3, 5]
frequency_ticks = [3.0, 3.5, 4.0, 4.5, 5.0]
text_box_x = 0.96
text_box_y = 0.09
text_box_font_size = 20
bcd_x = -0.13
bcd_y = 1.15

# ---------------- FIRST PLOT ----------------
img_path = os.path.join(fig_save_dir, "Calib_HFSS_only.png")
img = mpimg.imread(img_path)
ax = axes['row1']
ax.imshow(img)
# ax.imshow(img, aspect='auto', extent=[0, 1, 0, 1])
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
ax.axis('off')
ax.text(0.055, 1.10, r'\textbf{a}', transform=ax.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='right', color='black')

# ---------------- THIRD PLOT ----------------

# ---------------- Process Data ----------------
npy_name_prefix = "2nm_AlOx_no_calib"
display_offset = 25
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, rf"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:]

time_axis = np.arange(mag_avg_log_matrix.shape[1])/552.96  # Data points index
freq_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length

# ---------------- Right subplot: Color map ----------------
ax_right = axes['row3']

im = ax_right.imshow(
    mag_avg_log_matrix.T,
    aspect='auto',
    extent=[freq_axis[0], freq_axis[-1], time_axis[0], time_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3.5,
)
ax_right.set_ylim([0, 0.8])
ax_right.set_xlim(display_freq_range)
ax_right.set_xticks(frequency_ticks)
ax_right.set_xlabel("Frequency [GHz]")
ax_right.set_ylabel(r"Time [$\mu$s]", labelpad=12.5)
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

ax_right.text(
    text_box_x, text_box_y, "Uncalibrated",  # position in axes coordinates
    transform=ax_right.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

center = 4.78
band_width = find_FWHM_and_plot(pulse_width=308, save_plot=False)
pick_time=0.7
ax_right.hlines(y=pick_time, xmin=center - band_width/2, xmax=center + band_width/2, color='white', linewidth=1.5)

ax_right.text(bcd_x-0.009, bcd_y, r'\textbf{c}', transform=ax_right.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='right', color='black')


# ---------------- FORTH PLOT ----------------

# ---------------- Process Data ----------------
npy_name_prefix = "2nm_AlOx_HFSS_calib"
display_offset = 25
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()
IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, rf"{npy_name_prefix}_IQ_avg_matrix.npy"))

I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix_2 = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix_2 = np.log10(mag_avg_matrix_2 + 0.01)
mag_avg_log_matrix_2 = mag_avg_log_matrix_2[:, display_offset:]

time_axis = np.arange(mag_avg_log_matrix_2.shape[1])/552.96  # Data points index
freq_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix_2.shape[0]])/1e3  # Ensure correct length

# ---------------- Right subplot: Color map ----------------
ax_right = axes['row4']

im = ax_right.imshow(
    mag_avg_log_matrix_2.T,
    aspect='auto',
    extent=[freq_axis[0], freq_axis[-1], time_axis[0], time_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3.5,
)
ax_right.set_ylim([0, 0.8])
ax_right.set_xlim(display_freq_range)
ax_right.set_xticks(frequency_ticks)
ax_right.set_xlabel("Frequency [GHz]")
ax_right.set_ylabel(r"Time [$\mu$s]", labelpad=12.5)
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

ax_right.text(
    text_box_x, text_box_y, "Calibrated",  # position in axes coordinates
    transform=ax_right.transAxes,
    fontsize=text_box_font_size,
    ha='right', va='bottom',
    color='black',
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        boxstyle='round,pad=0.3',
        edgecolor='none'
    )
)

center = 4.78
band_width = find_FWHM_and_plot(pulse_width=308, save_plot=False)
pick_time=0.7
ax_right.hlines(y=pick_time, xmin=center - band_width/2, xmax=center + band_width/2, color='white', linewidth=1.5)

ax_right.text(bcd_x, bcd_y, r'\textbf{d}', transform=ax_right.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='right', color='black')

# ---------------- SECOND PLOT hfss data ----------------

# ---------------- Process Data ----------------
avg_E_at_freq_list = np.array([
    45.70362920445715, 84.01254413738913, 183.7138124332682, 655.8765832590868,
    1427.4670680256363, 575.6066950073034, 491.49252785286774, 562.5895410121663,
    723.9646496106842, 1137.146632533915, 1936.7408607073496, 1569.0531240754958,
    1043.547602667487, 824.4294167348312, 726.0535689978666, 702.1931998694984,
    631.8258663088808
])

HFSS_freq_list = np.arange(2, 6.001, 0.25)


# ---------------- Right subplot: Color map ----------------
ax_right = axes['row2']

# Left y-axis: Original data
ax_right.plot(HFSS_freq_list, avg_E_at_freq_list, label='Uncalibrated Field',
              color='tab:blue', linestyle='--', marker='D',
              markerfacecolor='tab:blue', 
              markersize=6)

ax_right.set_xlabel("Frequency [GHz]")
ax_right.set_ylabel("E-field [V/m]", color='tab:blue')
ax_right.tick_params(axis='y', labelcolor='tab:blue')
ax_right.set_xlim(3, 5)
ax_right.set_yticks([0, 500, 1000, 1500, 2000])
ax_right.set_ylim(0, 2000)

# Right y-axis: Inverse of data
ax_right_twin = ax_right.twinx()
inverse_E = 491.49252785286774 / avg_E_at_freq_list
interpolator = interp1d(HFSS_freq_list, inverse_E, kind='linear')
interp_freq_points = np.array(pulse_frequency_list)/1000
interp_inverse_E = [interpolator(pulse_frequency) for pulse_frequency in interp_freq_points]
ax_right_twin.plot(interp_freq_points, interp_inverse_E, label='Scaling Factor', color='tab:red')
ax_right_twin.set_ylabel("Scaling Factor", color='tab:red', labelpad=12.5)
ax_right_twin.tick_params(axis='y', labelcolor='tab:red')
ax_right_twin.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax_right_twin.set_ylim(0, 1)
ax_right.set_xticks(frequency_ticks)

# Calibrated field = inverse * uncalibrated
product = inverse_E * avg_E_at_freq_list
ax_right.plot(HFSS_freq_list, product, label='Calibrated Field', color='tab:blue')

# Combine legends from both y-axes
lines1, labels1 = ax_right.get_legend_handles_labels()
lines2, labels2 = ax_right_twin.get_legend_handles_labels()
ax_right.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax_right.text(bcd_x+0.0005, 1.33, r'\textbf{b}', transform=ax_right.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='right', color='black')



# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"Fig_proper_calibration.png")
if save_fig:
    plt.savefig(file_path, dpi=350, bbox_inches = 'tight')
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()
