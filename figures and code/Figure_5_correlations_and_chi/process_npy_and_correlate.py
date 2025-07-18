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


# #### Preamble
# # Set the font globally to Helvetica
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

##############################################################################
# 1. General matplotlib settings
##############################################################################
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.titlesize'] = 14

script_dir = os.path.dirname(os.path.abspath(__file__))

fig_save_dir = os.path.join(script_dir, 'analysis_plots')
if not os.path.isdir(fig_save_dir):
    os.mkdir(fig_save_dir)

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

def fft_custom(trace, time_spacing_us):
    N = len(trace)   # Number of points in FFT
    t = np.arange(N) * time_spacing_us  # Time vector
    fft_result = np.fft.fft(trace, N)
    frequencies = np.fft.fftfreq(N, time_spacing_us)  # Frequency axis
    magnitude = np.abs(fft_result)[:N // 2]  # One-sided spectrum
    frequencies_MHz = frequencies[:N // 2]  # Positive frequencies only
    return frequencies_MHz, magnitude

def g2_correlation(intensity, time_spacing_us, max_tau_us=None):
    intensity = np.array(intensity)
    n = len(intensity)
    max_lag = n if max_tau_us is None else min(n, int(max_tau_us / time_spacing_us))
    corr = np.correlate(intensity, intensity, mode='full')
    corr = corr[len(corr)//2:]
    normalization = np.mean(intensity)**2 * np.arange(n, n - max_lag, -1)
    g2 = corr[:max_lag] / normalization
    tau_values = np.arange(max_lag) * time_spacing_us
    return tau_values, g2

def compute_chi2_matrix(g2_arr, mag_data, time_spacing_us):
    num_freqs, num_tau = g2_arr.shape
    omega = np.fft.fftshift(np.fft.fftfreq(num_tau, d=time_spacing_us))
    pos_idx = omega >= 0
    omega = omega[pos_idx]
    chi2_matrix = np.zeros((num_freqs, len(omega)))
    for i in range(num_freqs):
        S_omega = np.fft.fftshift(np.fft.fft(g2_arr[i]))
        S_omega = S_omega[pos_idx]
        chi2_matrix[i] = -1*np.imag(S_omega)
    return omega, chi2_matrix

matrix_data_dir = os.path.join(script_dir, 'matrix_npy_save_folder')

save_fig = True

fig = plt.figure(figsize=(8, 10))  # Adjust width to accommodate both plots
gs = gridspec.GridSpec(
    4, 2,
    width_ratios=[1,3],
    wspace=0.2,
    hspace=0.4  # Increase or decrease for more/less vertical spacing
)

pulse_start_idx = 11
# pulse_end_idx = 28
pulse_end_idx = 85 #75
line_plot_line_width = 1.2
slicing_line_width = 1.5
slicing_line_color = 'white'
slicing_line_style = '--'
slicing_line_alpha = 0.5
slicing_line_dashes = (6,10)
display_freq_range = [3, 5]
display_tau_range = [0,0.5]
frequency_ticks = [3.0, 3.5, 4.0, 4.5, 5.0]
tau_color = plt.cm.inferno(215)
left_plot_color = 'cornflowerblue'
text_box_x = 0.96
text_box_y = 0.09
text_box_font_size = 16
max_tau = 0.5
slice_idx = 3322 - 3000 # actual freq - 3000
i_x = 0.95
i_y = 0.95
ii_x = 0.98
ii_y = 0.95
i_ii_size = 20 #27
abcd_x = -0.30 # -0.47
abcd_y = 1.17
abcd_size = 23 #30



# ---------------- Process Data ----------------
npy_name_prefix = "shipley"
display_offset = 25
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

mag_avg_log_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_mag_avg_log_matrix.npy"))/10
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:display_offset+600]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

mag_avg_log_matrix_pulse = mag_avg_log_matrix[:, pulse_start_idx:pulse_end_idx]
mag_avg_log_matrix_pulse_avg = np.mean(mag_avg_log_matrix_pulse, axis=1)

### these are the mag and log mag matrixes without pulse 
mag_avg_log_matrix_transient_region = mag_avg_log_matrix[:,pulse_end_idx:]
mag_avg_matrix_transient_region = 10**(mag_avg_log_matrix_transient_region) - 0.01

# initialize the fft and g2 matricies
fft_avg_list = []
g2_list = []

# process fft and g2
for idx, pulse_frequency in enumerate(pulse_frequency_list):

    mag_avg = mag_avg_matrix_transient_region[idx]
    intensity = mag_avg**2

    tau_values, g2 = g2_correlation(intensity, time_spacing_us=1/552.96, max_tau_us=max_tau)
    g2_list.append(g2)

    mag_avg_log = np.log10(mag_avg + 0.01)
    fft_freq_MHz, fft_of_truncated_trace = fft_custom(mag_avg_log, 1/552.96)
    fft_avg_list.append(fft_of_truncated_trace)

g2_matrix = np.array(g2_list)
g2_matrix_log = np.log10(g2_matrix + 1e-2)
tau_values = np.array(tau_values)
fft_matrix = np.array(fft_avg_list)
fft_matrix_log = np.log10(fft_matrix + 1e-2)
fft_freq_MHz = np.array(fft_freq_MHz)

omega, chi2 = compute_chi2_matrix(g2_matrix, mag_avg_matrix_transient_region, 1/552.96)
# chi2_log = 10 * np.log10(chi2 + 1e-2)
time_axis = np.arange(mag_avg_log_matrix_transient_region.shape[1]) / 552.96  # Data points index
freq_axis = np.array(pulse_frequency_list)/1e3  # Ensure correct length

# print(np.shape(omega))
# print(np.shape(chi2))
# print('waiting..')
# time.sleep(10000000)


# ---------------- 1st plot magnitude ----------------

# ---------------- Left subplot: Line plot ----------------
ax_left = fig.add_subplot(gs[0, 0])

ax_left.plot(mag_avg_log_matrix_transient_region[slice_idx], time_axis, '-', c=left_plot_color, linewidth=line_plot_line_width)
# ax_left.set_xlabel("Log Magnitude [arb.]")
ax_left.set_ylabel(r"Time [$\mu$s]")
ax_left.set_xlabel(r"Log$_{10}$(A) [arb.]")
ax_left.set_ylim([0, 0.8])
ax_left.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax_left.set_xlim([-2, 3])
ax_left.set_xticks([-2, 0, 2])
ax_left.text(abcd_x, abcd_y, r'\textbf{a}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[0, 1])

im = ax_right.imshow(
    mag_avg_log_matrix_transient_region.T,
    aspect='auto',
    extent=[freq_axis[0], freq_axis[-1], time_axis[0], time_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)
ax_right.set_ylim([0, 0.8])
ax_right.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax_right.axvline(x=freq_axis[slice_idx], color=slicing_line_color, linestyle=slicing_line_style, linewidth=slicing_line_width, alpha=slicing_line_alpha, dashes=slicing_line_dashes)
ax_right.set_xlabel('Frequency [GHz]')
ax_right.set_xticks(frequency_ticks)
# cbar = fig.colorbar(im, ax=ax_right, label="Log Magnitude [arb.]")
cbar = fig.colorbar(im, ax=ax_right)
cbar.set_label(r"Log$_{10}$(A) [arb.]", labelpad=7)
ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

center = 4.78
band_wdith = 0.038
pick_time=0.7
ax_right.hlines(y=pick_time, xmin=center - band_wdith/2, xmax=center + band_wdith/2, color='white', linewidth=1.5)




# ---------------- 2nd plot fft ----------------

# ---------------- Left subplot: Line plot ----------------
ax_left = fig.add_subplot(gs[1, 0])

ax_left.plot(fft_matrix[slice_idx], fft_freq_MHz, '-', c=left_plot_color, linewidth=line_plot_line_width)
# ax_left.set_xlabel("Log Magnitude [arb.]")
ax_left.set_ylabel(r"Frequency [MHz]")
ax_left.set_xlabel(r"FFT(Log$_{10}$(A)) [arb.]")
ax_left.set_xlim([0, 20])
ax_left.set_ylim([0, 200])
ax_left.set_xticks([0, 5, 10, 15, 20])
ax_left.text(abcd_x, abcd_y, r'\textbf{b}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')


# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[1, 1])

im = ax_right.imshow(
    fft_matrix.T,
    aspect='auto',
    extent=[freq_axis[0], freq_axis[-1], fft_freq_MHz[0], fft_freq_MHz[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=0,
    vmax=20,
)
# X, Y = np.meshgrid(freq_axis, fft_freq_MHz)
# contour = ax_right.contour(
#     X, Y, fft_matrix.T,
#     levels=[140],
#     colors='cyan',
#     linewidths=0.5
# )

ax_right.set_ylim([0, 200])
ax_right.axvline(x=freq_axis[slice_idx], color=slicing_line_color, linestyle=slicing_line_style, linewidth=slicing_line_width, alpha=slicing_line_alpha, dashes=slicing_line_dashes)
ax_right.set_xticks(frequency_ticks)
ax_right.set_xlabel('Frequency [GHz]')
cbar = fig.colorbar(im, ax=ax_right)
cbar.set_label(r"FFT(Log$_{10}$(A)) [arb.]", labelpad=10)
ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

center = 4.78
band_wdith = 0.038
pick_time=200*7/8
ax_right.hlines(y=pick_time, xmin=center - band_wdith/2, xmax=center + band_wdith/2, color='white', linewidth=1.5)

# ---------------- 3rd plot g2 ----------------

# ---------------- Left subplot: Line plot ----------------
ax_left = fig.add_subplot(gs[2, 0])

ax_left.plot(g2_matrix_log[slice_idx], tau_values, '-', c=left_plot_color, linewidth=line_plot_line_width)
# ax_left.set_xlabel("Log Magnitude [arb.]")
ax_left.set_ylabel(r"$\tau^{\prime}$ [$\mu$s]")
ax_left.set_xlabel(r"Log$_{10}$(g$^{(2)}$) [arb.]")
ax_left.set_xlim([-2, 1.5])
ax_left.set_xticks([-2, -1, 0, 1])
ax_left.set_yticks([0, 0.2, 0.4])
ax_left.set_ylim([0, max_tau])
ax_left.text(abcd_x, abcd_y, r'\textbf{c}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[2, 1])

im = ax_right.imshow(
    g2_matrix_log.T,
    aspect='auto',
    extent=[freq_axis[0], freq_axis[-1], tau_values[0], tau_values[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2.0,
    vmax=1.5,
)

# ax_right.set_ylim([0, 0.8])
ax_right.axvline(x=freq_axis[slice_idx], color=slicing_line_color, linestyle=slicing_line_style, linewidth=slicing_line_width, alpha=slicing_line_alpha, dashes=slicing_line_dashes)
ax_right.set_xticks(frequency_ticks)
ax_right.set_xlabel('Frequency [GHz]')
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(g$^{(2)}$) [arb.]")
ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

center = 4.78
band_wdith = 0.038
pick_time=0.5*7/8
ax_right.hlines(y=pick_time, xmin=center - band_wdith/2, xmax=center + band_wdith/2, color='white', linewidth=1.5)

# ---------------- 4th plot chi_2 ----------------

# ---------------- Left subplot: Line plot ----------------
ax_left = fig.add_subplot(gs[3, 0])

ax_left.plot(chi2[slice_idx], omega, '-', c=left_plot_color, linewidth=line_plot_line_width)
# ax_left.set_xlabel("Log Magnitude [arb.]")
ax_left.set_ylabel(r"Frequency [MHz]")
ax_left.set_xlabel(r"$\chi^{\prime\prime}$ [arb.]")
ax_left.set_xlim([-10, 150])
ax_left.set_ylim([0, 200])
ax_left.set_xticks([ 0, 50, 100, 150])
ax_left.axvline(x=0, color='cyan', linewidth=0.5)
ax_left.text(abcd_x, abcd_y, r'\textbf{d}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[3, 1])

im = ax_right.imshow(
    # chi2_log.T,
    chi2.T,
    aspect='auto',
    extent=[freq_axis[0], freq_axis[-1], omega[0], omega[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-10,
    vmax=150,
)
X, Y = np.meshgrid(freq_axis, omega)
contour = ax_right.contour(
    X, Y, chi2.T,
    levels=[0],
    colors='cyan',
    linewidths=0.5
)
ax_right.set_ylim([0, 200])
ax_right.axvline(x=freq_axis[slice_idx], color=slicing_line_color, linestyle=slicing_line_style, linewidth=slicing_line_width, alpha=slicing_line_alpha, dashes=slicing_line_dashes)
ax_right.set_xticks(frequency_ticks)
ax_right.set_xlabel('Frequency [GHz]')
cbar = fig.colorbar(im, ax=ax_right, label=r"$\chi^{\prime\prime}$ [arb.]")
# Add horizontal cyan line at 0 on the colorbar
cbar.ax.hlines(y=0, xmin=-0.05, xmax=1.05, colors='cyan', linewidth=0.8, transform=cbar.ax.transData, clip_on=False)
ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')


center = 4.78
band_wdith = 0.038
pick_time=200*7/8
ax_right.hlines(y=pick_time, xmin=center - band_wdith/2, xmax=center + band_wdith/2, color='white', linewidth=1.5)

# # add (a)bcd...
# plt.rcParams['font.sans-serif'] = ['Helvetica']
# plt.rcParams['text.usetex'] = True

# abcd_x = 0.05
# i_x  = 0.25
# ii_x = 0.78
# a_y = 0.87
# label_font_size = 25

# fig.text(abcd_x, a_y, r'\textbf{a}', fontsize=label_font_size)
# fig.text(abcd_x, 0.77, r'\textbf{b}', fontsize=label_font_size)
# fig.text(abcd_x, 0.5, r'\textbf{c}', fontsize=label_font_size)
# fig.text(abcd_x, 0.26, r'\textbf{d}', fontsize=label_font_size)
# fig.text(i_x, a_y, r'\textbf{i}', fontsize=label_font_size)
# fig.text(ii_x, a_y, r'\textbf{ii}', fontsize=label_font_size)

# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"shipley_correlations_neg_{slice_idx+3000}_end_{pulse_end_idx}.png")
if save_fig:
    plt.savefig(file_path, dpi=350, bbox_inches = 'tight')
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()
