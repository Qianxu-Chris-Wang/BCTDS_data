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
from matplotlib.widgets import Slider
import matplotlib.image as mpimg

def fit_piecewise_slider(trace_list, time_spacing_us, show_slider=True):
    N = len(trace_list)
    t = np.arange(len(trace_list[0])) * time_spacing_us
    tau_array = np.full(N, np.nan)
    fit_curves = []

    # --- Piecewise model: continuous linear -> flat ---
    def piecewise_model_continuous(t, x_c, k, b):
        return np.where(t < x_c, k * t + b, k * x_c + b)

    # --- Fit all traces ---
    for trace in trace_list:
        if not isinstance(trace, np.ndarray):
            fit_curves.append(None)
            continue
        try:
            x_c0 = t[len(t)//2]
            k0 = (trace[5] - trace[0]) / (t[5] - t[0])
            b0 = trace[0]
            p0 = [x_c0, k0, b0]
            bounds = ([t[0], -np.inf, -np.inf], [t[-1], np.inf, np.inf])
            popt, _ = curve_fit(piecewise_model_continuous, t, trace, p0=p0, bounds=bounds)
            tau_array[len(fit_curves)] = -np.log10(np.e)/popt[1] * 1000 # Save x_c as effective decay time
            fit_curves.append((popt, piecewise_model_continuous(t, *popt)))
        except Exception:
            fit_curves.append((None, None))

    if not show_slider:
        return tau_array

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.3)

    first_trace = trace_list[0]
    trace_line, = ax.plot(t, first_trace, label="Data", color="black")
    first_fit = fit_curves[0][1] if fit_curves[0][1] is not None else np.full_like(t, np.nan)
    fit_line, = ax.plot(t, first_fit, label="Fit", color="red")

    title = ax.set_title(f"Trace at {3000:.1f} MHz")
    equation_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=10, verticalalignment='top')

    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    # --- Slider setup ---
    slider_ax = plt.axes([0.2, 0.15, 0.6, 0.03])
    idx_slider = Slider(
        ax=slider_ax,
        label='Index',
        valmin=0,
        valmax=N - 1,
        valinit=0,
        valstep=1,
    )

    # --- Update function ---

    def update(val):
        idx = int(idx_slider.val)
        trace = trace_list[idx]
        trace_line.set_ydata(trace)

        popt, fit_vals = fit_curves[idx]
        if popt is not None:
            fit_line.set_ydata(fit_vals)
            fit_line.set_visible(True)

            # Refit to get A, tau for annotation
            try:
                x_c, k, b = popt
                C = k * x_c + b
                eq_str = (
                    rf"$y(t) = {k:.2f} \cdot t + {b:.2f}, \quad t < {x_c:.2f}$" + "\n" +
                    rf"$y(t) = {C:.2f}, \quad t \ge {x_c:.2f}$"
                )
                equation_text.set_text(eq_str)
            except Exception:
                equation_text.set_text('')
        else:
            fit_line.set_visible(False)
            equation_text.set_text('')

        title.set_text(f"Trace at {3000 + idx:.1f} MHz")
        fig.canvas.draw_idle()

    idx_slider.on_changed(update)

    # Non-blocking display; resume after closing
    plt.show(block=False)
    input("Press Enter to continue...")
    plt.close(fig)
    plt.pause(0.001)

    return tau_array

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
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['figure.titlesize'] = 20

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

        ax.text(x=190, y=-0.003, s=r"$\Delta$f", fontsize=22, color="indigo", va='center')

        fig_save_dir = os.path.join(script_dir, 'analysis_plots')
        os.makedirs(fig_save_dir, exist_ok=True)
        file_path = os.path.join(fig_save_dir, f"bandwidth_only_full.png")
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {file_path}")
        plt.close()

    return np.abs(freqs_GHz[FWHM_idx_1]- freqs_GHz[FWHM_idx_0])

band_width = find_FWHM_and_plot(pulse_width=308, save_plot=True)


matplotlib.rcParams['axes.labelsize'] = 25
matrix_data_dir = os.path.join(script_dir, 'matrix_npy_save_folder')


save_fig = True

fig = plt.figure(figsize=(12, 10.5))  # optionally increase height for 3 rows

# Column layout
width_ratios = [1, 4, 1, 5]
gap_ratios   = [0.15, 0.7, 0.15]

unit = 1 / (sum(width_ratios) + sum(gap_ratios))
norm_widths = [w * unit for w in width_ratios]
norm_gaps   = [g * unit for g in gap_ratios]

lefts = [
    0,
    norm_widths[0] + norm_gaps[0],
    norm_widths[0] + norm_gaps[0] + norm_widths[1] + norm_gaps[1],
    norm_widths[0] + norm_gaps[0] + norm_widths[1] + norm_gaps[1] + norm_widths[2] + norm_gaps[2]
]

# Row layout: three rows
row_heights = [0.25, 0.25, 0.25]  # adjust if needed
row_gap = 0.07

# Compute bottoms for three rows
bottoms = [
    1 - row_heights[0],  # top row
    1 - row_heights[0] - row_gap - row_heights[1],  # middle row
    1 - row_heights[0] - row_gap - row_heights[1] - row_gap - row_heights[2],  # bottom row
]

pulse_start_idx = 11
pulse_end_idx = 28
line_plot_line_width = 0.8
pulse_mark_line_width = 1.5
pulse_mark_line_color = 'black'
pulse_mark_line_style = '--'
display_freq_range = [3, 5]
display_tau_range = [0,200]
display_tau_ticks = [40, 160]
frequency_ticks = [3.0, 3.5, 4.0, 4.5, 5.0]
# tau_color = plt.cm.inferno(215)
# tau_color = '#E65A2F'
tau_color = '#F57C1F'
transmission_color = 'indigo'
text_box_x = 0.96
text_box_y = 0.09
text_box_font_size = 20
show_fit_slices = False
i_x = 0.9
i_y = 0.95
ii_x = 0.97
ii_y = 0.95
i_ii_size = 30
abcd_x = -0.44
abcd_y = 1.1
abcd_size = 30

# ---------------- FIRST ROW LEFT PLOTS ----------------

# ---------------- Process Data ----------------

# npy_name_prefix = "SJ_empty_room_temp"
npy_name_prefix = "SJ_empty_room_fine"
display_offset = 20
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

# IQ_avg_matrix = np.load(r"C:\Users\chris\Desktop\TLS paper\Experiment data for Figures\Figure_3_version_2_six_sample_compare\matrix_npy_save_folder\SJ_empty_room_temp_IQ_avg_matrix.npy")
IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, rf"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[1000:, display_offset:display_offset+465]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

mag_avg_log_matrix_pulse = mag_avg_log_matrix[:, pulse_start_idx:pulse_end_idx]
mag_avg_log_matrix_pulse_avg = np.mean(mag_avg_log_matrix_pulse, axis=1)
# mag_avg_log_matrix_pulse_avg = mag_avg_log_matrix_pulse[:, 8]
mag_avg_log_matrix_transient_region = mag_avg_log_matrix[:,pulse_end_idx:]
mag_avg_matrix_transient_region = 10**(mag_avg_log_matrix_transient_region) - 0.01
# tau_us_array = fit_exponential_decay_list(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, plot_index=None) #2150
tau_us_array = fit_piecewise_slider(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, show_slider=show_fit_slices)
x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96


# ---------------- Left subplot: Line plot of pulse_avg ----------------
ax_left = fig.add_axes([lefts[0], bottoms[0], norm_widths[0], row_heights[0]])

ax_left.plot(tau_us_array, y_axis, color=tau_color, label=r'$\tau$ [$\mu$s]', linewidth=line_plot_line_width)
# ax_left.set_xlabel(r"Lifetime $\tau$ [ns]", color=tau_color)
ax_left.set_ylabel("Frequency [GHz]")
ax_left.set_ylim(display_freq_range)
ax_left.set_xlim(display_tau_range)
ax_left.set_xticks(display_tau_ticks)
ax_left.set_yticks(frequency_ticks)

ax_left.text(abcd_x, abcd_y, r'\textbf{a}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# Color bottom x-axis
ax_left.tick_params(axis='x', colors=tau_color)
ax_left.spines['bottom'].set_color(tau_color)

# Top x-axis (log magnitude)
ax_top = ax_left.twiny()
ax_top.plot(mag_avg_log_matrix_pulse_avg, y_axis, color=transmission_color, label='Avg. Log Mag', linewidth=line_plot_line_width)
ax_top.set_xlabel(r"Log$_{10}$(A) [arb.]", color=transmission_color, labelpad=10)
ax_top.set_xlim([2.0, 3.0])
ax_top.set_xticks([2.2, 2.8]) 

# Color top x-axis
ax_top.tick_params(axis='x', colors=transmission_color)
ax_top.spines['top'].set_color(transmission_color)


# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_axes([lefts[1], bottoms[0], norm_widths[1], row_heights[0]])

im = ax_right.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim(display_freq_range)
# ax_right.set_yticks(frequency_ticks)
ax_right.set_yticks([])
ax_right.set_ylabel('')
# ax_right.set_xlabel(r"Time [$\mu$s]")
# cbar = fig.colorbar(im, ax=ax_right, label="Log Magnitude [arb.]")

center = 3.532
pick_time=0.33
ax_right.vlines(x=pick_time, ymin=center - band_width/2, ymax=center + band_width/2, color='white', linewidth=1.5)

ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

ax_right.text(
    text_box_x, text_box_y, r"Empty, 300 K",  # position in axes coordinates
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




# ---------------- FIRST ROW RIGHT PLOTS ----------------

# ---------------- Process Data ----------------

npy_name_prefix = "SJ_sapphire_room_fine"
display_offset = 20
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

# IQ_avg_matrix = np.load(r"C:\Users\chris\Desktop\TLS paper\Experiment data for Figures\Figure_3_version_2_six_sample_compare\matrix_npy_save_folder\SJ_sapphire_room_fine_IQ_avg_matrix.npy")
IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, rf"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[1000:, display_offset:display_offset+465]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

mag_avg_log_matrix_pulse = mag_avg_log_matrix[:, pulse_start_idx:pulse_end_idx]
mag_avg_log_matrix_pulse_avg = np.mean(mag_avg_log_matrix_pulse, axis=1)
# mag_avg_log_matrix_pulse_avg = mag_avg_log_matrix_pulse[:, 8]
mag_avg_log_matrix_transient_region = mag_avg_log_matrix[:,pulse_end_idx:]
mag_avg_matrix_transient_region = 10**(mag_avg_log_matrix_transient_region) - 0.01
# tau_us_array = fit_exponential_decay_list(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, plot_index=None) #2150
tau_us_array = fit_piecewise_slider(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, show_slider=show_fit_slices)
x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96


# ---------------- Left subplot: Line plot of pulse_avg ----------------
ax_left = fig.add_axes([lefts[2], bottoms[0], norm_widths[2], row_heights[0]])

ax_left.text(abcd_x, abcd_y, r'\textbf{b}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')


ax_left.plot(tau_us_array, y_axis, color=tau_color, label=r'$\tau$ [$\mu$s]', linewidth=line_plot_line_width)
# ax_left.set_xlabel(r"Ringdown Lifetime $\tau$ [$\mu$s]", color=tau_color)
# ax_left.set_ylabel("Frequency [GHz]")
ax_left.set_ylim(display_freq_range)
ax_left.set_xlim(display_tau_range)
ax_left.set_xticks(display_tau_ticks)
ax_left.set_yticks(frequency_ticks)

# Color bottom x-axis
ax_left.tick_params(axis='x', colors=tau_color)
ax_left.spines['bottom'].set_color(tau_color)

# Top x-axis (log magnitude)
ax_top = ax_left.twiny()
ax_top.plot(mag_avg_log_matrix_pulse_avg, y_axis, color=transmission_color, label='Avg. Log Mag', linewidth=line_plot_line_width)
ax_top.set_xlabel(r"Log$_{10}$(A) [arb.]", color=transmission_color, labelpad=10)
ax_top.set_xlim([2.0, 3.0])
ax_top.set_xticks([2.2, 2.8]) 

# Color top x-axis
ax_top.tick_params(axis='x', colors=transmission_color)
ax_top.spines['top'].set_color(transmission_color)

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_axes([lefts[3], bottoms[0], norm_widths[3], row_heights[0]])

im = ax_right.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim(display_freq_range)
# ax_right.set_yticks(frequency_ticks)
ax_right.set_yticks([])
ax_right.set_ylabel('')
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

center = 3.532
pick_time=0.33
ax_right.vlines(x=pick_time, ymin=center - band_width/2, ymax=center + band_width/2, color='white', linewidth=1.5)

# plt.rcParams['font.sans-serif'] = ['Helvetica']
# plt.rcParams['text.usetex'] = True
ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

ax_right.text(
    text_box_x, text_box_y, "Sapphire, 300 K",  # position in axes coordinates
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

# ---------------- SECOND ROW LEFT PLOTS ----------------

# ---------------- Process Data ----------------
npy_name_prefix = "SJ_empty_cryo"
display_offset = 32
pulse_frequency_list = np.arange(3000, 5000, 10).tolist()

# IQ_avg_matrix = np.load(r"C:\Users\chris\Desktop\TLS paper\Experiment data for Figures\Figure_3_version_2_six_sample_compare\matrix_npy_save_folder\SJ_empty_cryo_IQ_avg_matrix.npy")
IQ_avg_matrix = np.load(os.path.join(matrix_data_dir, rf"{npy_name_prefix}_IQ_avg_matrix.npy"))
I_avg_matrix = IQ_avg_matrix[0]
Q_avg_matrix = IQ_avg_matrix[1]
mag_avg_matrix = np.abs(I_avg_matrix + 1j *Q_avg_matrix)
mag_avg_log_matrix = np.log10(mag_avg_matrix + 0.01)
mag_avg_log_matrix = mag_avg_log_matrix[100:, display_offset:display_offset+465]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

mag_avg_log_matrix_pulse = mag_avg_log_matrix[:, pulse_start_idx:pulse_end_idx]
mag_avg_log_matrix_pulse_avg = np.mean(mag_avg_log_matrix_pulse, axis=1)
# mag_avg_log_matrix_pulse_avg = mag_avg_log_matrix_pulse[:, 8]
mag_avg_log_matrix_transient_region = mag_avg_log_matrix[:,pulse_end_idx:]
mag_avg_matrix_transient_region = 10**(mag_avg_log_matrix_transient_region) - 0.01
# tau_us_array = fit_exponential_decay_list(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, plot_index=None) #2150
tau_us_array = fit_piecewise_slider(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, show_slider=show_fit_slices)
x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96


# ---------------- Left subplot: Line plot of pulse_avg ----------------
ax_left = fig.add_axes([lefts[0], bottoms[1], norm_widths[0], row_heights[1]])

ax_left.plot(tau_us_array, y_axis, color=tau_color, label=r'$\tau$ [$\mu$s]', linewidth=line_plot_line_width)
# ax_left.set_xlabel(r"Lifetime $\tau$ [ns]", color=tau_color)
ax_left.set_ylabel("Frequency [GHz]")
ax_left.set_ylim(display_freq_range)
ax_left.set_xlim(display_tau_range)
ax_left.set_xticks(display_tau_ticks)
ax_left.set_yticks(frequency_ticks)

ax_left.text(abcd_x, abcd_y, r'\textbf{c}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# Color bottom x-axis
ax_left.tick_params(axis='x', colors=tau_color)
ax_left.spines['bottom'].set_color(tau_color)

# Top x-axis (log magnitude)
ax_top = ax_left.twiny()
ax_top.plot(mag_avg_log_matrix_pulse_avg, y_axis, color=transmission_color, label='Avg. Log Mag', linewidth=line_plot_line_width)
# ax_top.set_xlabel("Pulse Log Magnitude [arb.]", color=transmission_color)
ax_top.set_xlim([3.0, 3.5]) 
ax_top.set_xticks([3.1, 3.4]) 

# Color top x-axis
ax_top.tick_params(axis='x', colors=transmission_color)
ax_top.spines['top'].set_color(transmission_color)

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_axes([lefts[1], bottoms[1], norm_widths[1], row_heights[1]])

# pad the top slice data with NaN values, since we used np.arange(3000, 5000, 10)
nan_row = np.full((1, np.shape(mag_avg_log_matrix)[1]), np.nan) 
mag_avg_log_matrix = np.concatenate([mag_avg_log_matrix, nan_row], axis=0)
mag_avg_log_matrix = np.ma.masked_invalid(mag_avg_log_matrix)
y_axis = np.arange(3000, 5010, 10)/1e3
cmap = plt.cm.inferno.copy()
cmap.set_bad(color='black')

im = ax_right.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap=cmap,
    interpolation='none',
    vmin=-2,
    vmax=3,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim(display_freq_range)
# ax_right.set_yticks(frequency_ticks)
ax_right.set_yticks([])
ax_right.set_ylabel('')
# ax_right.set_xlabel(r"Time [$\mu$s]")
# cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

center = 3.532
pick_time=0.33
ax_right.vlines(x=pick_time, ymin=center - band_width/2, ymax=center + band_width/2, color='white', linewidth=1.5)

ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

ax_right.text(
    text_box_x, text_box_y, r"Empty, 10 mK",  # position in axes coordinates
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

# ---------------- SECOND ROW RIGHT PLOTS ----------------

# ---------------- Process Data ----------------

npy_name_prefix = "truncated_cold_clean_sapphire_fine"
display_offset = 32
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

mag_avg_log_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_mag_avg_log_matrix.npy"))/10
mag_avg_log_matrix = mag_avg_log_matrix[500:, display_offset:display_offset+465]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

mag_avg_log_matrix_pulse = mag_avg_log_matrix[:, pulse_start_idx:pulse_end_idx]
mag_avg_log_matrix_pulse_avg = np.mean(mag_avg_log_matrix_pulse, axis=1)
# mag_avg_log_matrix_pulse_avg = mag_avg_log_matrix_pulse[:, 8]
mag_avg_log_matrix_transient_region = mag_avg_log_matrix[:,pulse_end_idx:]
mag_avg_matrix_transient_region = 10**(mag_avg_log_matrix_transient_region) - 0.01
# tau_us_array = fit_exponential_decay_list(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, plot_index=None) #2150
tau_us_array = fit_piecewise_slider(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, show_slider=show_fit_slices)
x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96


# ---------------- Left subplot: Line plot of pulse_avg ----------------
ax_left = fig.add_axes([lefts[2], bottoms[1], norm_widths[2], row_heights[1]])
ax_left.plot(tau_us_array, y_axis, color=tau_color, label=r'$\tau$ [$\mu$s]', linewidth=line_plot_line_width)
# ax_left.set_xlabel(r"Ringdown Lifetime $\tau$ [$\mu$s]", color=tau_color)
# ax_left.set_ylabel("Frequency [GHz]")
ax_left.set_ylabel('')
ax_left.set_ylim(display_freq_range)
ax_left.set_xlim(display_tau_range)
ax_left.set_xticks(display_tau_ticks)
ax_left.set_yticks(frequency_ticks)

ax_left.text(abcd_x, abcd_y, r'\textbf{d}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# Color bottom x-axis
ax_left.tick_params(axis='x', colors=tau_color)
ax_left.spines['bottom'].set_color(tau_color)

# Top x-axis (log magnitude)
ax_top = ax_left.twiny()
ax_top.plot(mag_avg_log_matrix_pulse_avg, y_axis, color=transmission_color, label='Avg. Log Mag', linewidth=line_plot_line_width)
# ax_top.set_xlabel(r"Log$_{10}$(A) [arb.]", color=transmission_color)
ax_top.set_xlim([3.0, 3.5])  # adjust as needed
ax_top.set_xticks([3.1, 3.4]) 

# Color top x-axis
ax_top.tick_params(axis='x', colors=transmission_color)
ax_top.spines['top'].set_color(transmission_color)

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_axes([lefts[3], bottoms[1], norm_widths[3], row_heights[1]])

im = ax_right.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim(display_freq_range)
# ax_right.set_yticks(frequency_ticks)
ax_right.set_yticks([])
ax_right.set_ylabel('')
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

center = 3.532
pick_time=0.33
ax_right.vlines(x=pick_time, ymin=center - band_width/2, ymax=center + band_width/2, color='white', linewidth=1.5)

# img_path = r"C:\Users\chris\Desktop\TLS paper\Experiment data for Figures\Figure_3_version_2_six_sample_compare\analysis_plots\bandwidth_only_full.png"
# img = mpimg.imread(img_path)
# ax_right.imshow(img, extent=(0.5, 0.95,18, 0.95))

# img_path = r"C:\Users\chris\Desktop\TLS paper\Experiment data for Figures\Figure_3_version_2_six_sample_compare\analysis_plots\bandwidth_only_full.png"
img_path = os.path.join(fig_save_dir, rf"bandwidth_only_full.png")
inset_ax = ax_right.inset_axes([0.30, 0.22, 0.93, 0.74])  # [x0, y0, width, height] in axis coords
inset_img = mpimg.imread(img_path)
inset_ax.imshow(inset_img)
inset_ax.axis('off')

ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black', zorder=10)

ax_right.plot([0.335, 0.55], [3.512, 3.96], color=tau_color, linewidth=1.5, linestyle='--', dashes=(5, 3), zorder=100)
ax_right.plot([0.335, 0.55], [3.552, 4.175], color=tau_color, linewidth=1.5, linestyle='--', dashes=(5, 3), zorder=100)

ax_right.text(
    text_box_x, text_box_y, "Sapphire, 10 mK",  # position in axes coordinates
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

# plt.show()

# ---------------- THIRD ROW LEFT PLOTS ----------------

# ---------------- Process Data ----------------

npy_name_prefix = "2nm_Al2O3_fine"
display_offset = 33
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

mag_avg_log_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_mag_avg_log_matrix.npy"))/10
mag_avg_log_matrix = mag_avg_log_matrix[1000:, display_offset:display_offset+465]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

mag_avg_log_matrix_pulse = mag_avg_log_matrix[:, pulse_start_idx:pulse_end_idx]
mag_avg_log_matrix_pulse_avg = np.mean(mag_avg_log_matrix_pulse, axis=1)
# mag_avg_log_matrix_pulse_avg = mag_avg_log_matrix_pulse[:, 8]
mag_avg_log_matrix_transient_region = mag_avg_log_matrix[:,pulse_end_idx:]
mag_avg_matrix_transient_region = 10**(mag_avg_log_matrix_transient_region) - 0.01
# tau_us_array = fit_exponential_decay_list(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, plot_index=None) #2150
tau_us_array = fit_piecewise_slider(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, show_slider=show_fit_slices)
x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96


# ---------------- Left subplot: Line plot of pulse_avg ----------------
ax_left = fig.add_axes([lefts[0], bottoms[2], norm_widths[0], row_heights[2]])
ax_left.plot(tau_us_array, y_axis, color=tau_color, label=r'$\tau$ [$\mu$s]', linewidth=line_plot_line_width)
ax_left.set_xlabel(r"Lifetime $\tau$ [ns]", color=tau_color)
ax_left.set_ylabel("Frequency [GHz]")
ax_left.set_ylim(display_freq_range)
ax_left.set_xlim(display_tau_range)
ax_left.set_xticks(display_tau_ticks)
ax_left.set_yticks(frequency_ticks)

ax_left.text(abcd_x, abcd_y, r'\textbf{e}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# Color bottom x-axis
ax_left.tick_params(axis='x', colors=tau_color)
ax_left.spines['bottom'].set_color(tau_color)

# Top x-axis (log magnitude)
ax_top = ax_left.twiny()
ax_top.plot(mag_avg_log_matrix_pulse_avg, y_axis, color=transmission_color, label='Avg. Log Mag', linewidth=line_plot_line_width)
# ax_top.set_xlabel("Pulse Log Magnitude [arb.]", color=transmission_color)
ax_top.set_xlim([3.0, 3.5])  # adjust as needed
ax_top.set_xticks([3.1, 3.4]) 

# Color top x-axis
ax_top.tick_params(axis='x', colors=transmission_color)
ax_top.spines['top'].set_color(transmission_color)

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_axes([lefts[1], bottoms[2], norm_widths[1], row_heights[2]])

im = ax_right.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim(display_freq_range)
# ax_right.set_yticks(frequency_ticks)
ax_right.set_yticks([])
ax_right.set_ylabel('')
ax_right.set_xlabel(r"Time [$\mu$s]")
# cbar = fig.colorbar(im, ax=ax_right, label="Log Magnitude [arb.]")

center = 3.532
pick_time=0.33
ax_right.vlines(x=pick_time, ymin=center - band_width/2, ymax=center + band_width/2, color='white', linewidth=1.5)

ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

ax_right.text(
    text_box_x, text_box_y, r"2nm AlO\textsubscript{x}, 10 mK",  # position in axes coordinates
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

# ---------------- THIRD ROW RIGHT PLOTS ----------------

# ---------------- Process Data ----------------
npy_name_prefix = "shipley"
display_offset = 25
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

mag_avg_log_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_mag_avg_log_matrix.npy"))/10
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:display_offset+465]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

mag_avg_log_matrix_pulse = mag_avg_log_matrix[:, pulse_start_idx:pulse_end_idx]
mag_avg_log_matrix_pulse_avg = np.mean(mag_avg_log_matrix_pulse, axis=1)
# mag_avg_log_matrix_pulse_avg = mag_avg_log_matrix_pulse[:, 8]
mag_avg_log_matrix_transient_region = mag_avg_log_matrix[:,pulse_end_idx:]
mag_avg_matrix_transient_region = 10**(mag_avg_log_matrix_transient_region) - 0.01
# tau_us_array = fit_exponential_decay_list(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, plot_index=None) #2150
tau_us_array = fit_piecewise_slider(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, show_slider=show_fit_slices)
x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96


# ---------------- Left subplot: Line plot of pulse_avg ----------------
ax_left = fig.add_axes([lefts[2], bottoms[2], norm_widths[2], row_heights[2]])
ax_left.plot(tau_us_array, y_axis, color=tau_color, label=r'$\tau$ [$\mu$s]', linewidth=line_plot_line_width)
ax_left.set_xlabel(r"Lifetime $\tau$ [ns]", color=tau_color)
ax_left.set_ylabel("Frequency [GHz]")
ax_left.set_ylabel('')
ax_left.set_ylim(display_freq_range)
ax_left.set_xlim(display_tau_range)
ax_left.set_xticks(display_tau_ticks)
ax_left.set_yticks(frequency_ticks)

ax_left.text(abcd_x, abcd_y, r'\textbf{f}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# Color bottom x-axis
ax_left.tick_params(axis='x', colors=tau_color)
ax_left.spines['bottom'].set_color(tau_color)

# Top x-axis (log magnitude)
ax_top = ax_left.twiny()
ax_top.plot(mag_avg_log_matrix_pulse_avg, y_axis, color=transmission_color, label='Avg. Log Mag', linewidth=line_plot_line_width)
# ax_top.set_xlabel("Pulse Log Magnitude [arb.]", color=transmission_color)
ax_top.set_xlim([3.0, 3.5]) 
ax_top.set_xticks([3.1, 3.4]) 

# Color top x-axis
ax_top.tick_params(axis='x', colors=transmission_color)
ax_top.spines['top'].set_color(transmission_color)

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_axes([lefts[3], bottoms[2], norm_widths[3], row_heights[2]])

im = ax_right.imshow(
    mag_avg_log_matrix,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='inferno',
    interpolation='none',
    vmin=-2,
    vmax=3,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim(display_freq_range)
# ax_right.set_yticks(frequency_ticks)
ax_right.set_yticks([])
ax_right.set_ylabel('')
ax_right.set_xlabel(r"Time [$\mu$s]")
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

center = 3.532
pick_time=0.33
ax_right.vlines(x=pick_time, ymin=center - band_width/2, ymax=center + band_width/2, color='white', linewidth=1.5)

ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')

ax_right.text(
    text_box_x, text_box_y, r"Photoresist, 10 mK",  # position in axes coordinates
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



# # add (a)bcd...
# plt.rcParams['font.sans-serif'] = ['Helvetica']
# plt.rcParams['text.usetex'] = True
# fig.text(-0.02, 0.95, r'\textbf{a}', fontsize=40)
# fig.text(0.49, 0.95, r'\textbf{b}', fontsize=40)
# fig.text(-0.02, 0.45, r'\textbf{c}', fontsize=40)
# fig.text(0.49, 0.45, r'\textbf{d}', fontsize=40)


# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"Fig_six_sample_compare.png")
if save_fig:
    plt.savefig(file_path, dpi=350, bbox_inches = 'tight')
    print(f"Figure saved to: {file_path}")
plt.show()
plt.close()
