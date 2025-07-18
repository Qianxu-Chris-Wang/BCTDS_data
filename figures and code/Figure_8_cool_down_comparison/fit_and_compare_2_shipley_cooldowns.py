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
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
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

        fig_save_dir = os.path.join(script_dir, 'analysis_plots')
        os.makedirs(fig_save_dir, exist_ok=True)
        file_path = os.path.join(fig_save_dir, f"bandwidth_only_full.png")
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {file_path}")

    return np.abs(freqs_GHz[FWHM_idx_1]- freqs_GHz[FWHM_idx_0])


matrix_data_dir = os.path.join(script_dir, 'matrix_npy_save_folder')

# pulse_width_list = [308]

## samples from comparing room sapphire and cold clean sapphrie

# npy_name_prefix = "room_temp_fine"
# display_offset = 0

# npy_name_prefix = "truncated_cold_clean_sapphire_fine"
# display_offset = 32

# npy_name_prefix = "2nm_Al2O3_fine"
# display_offset = 33

# npy_name_prefix = "shipley"
# display_offset = 25

# npy_name_prefix = "shipley_2nd"
# display_offset = 25

# npy_name_prefix = "true_silicon"
# display_offset = 25

save_fig = True

fig = plt.figure(figsize=(7, 9))

gs = gridspec.GridSpec(
    nrows=3,
    ncols=2,
    wspace=0.04,
    hspace=0.3,
    width_ratios=[1, 5]
)

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
i_ii_size = 27
abcd_x = -0.47
abcd_y = 1.15
abcd_size = 30


# ---------------- FIRST ROW PLOTS ----------------

# ---------------- Process Data ----------------

npy_name_prefix = "shipley"
display_offset = 25
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

mag_avg_log_matrix = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_mag_avg_log_matrix.npy"))/10
mag_avg_log_matrix = mag_avg_log_matrix[:, display_offset:display_offset+465]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix)}')

mag_avg_log_matrix_pulse = mag_avg_log_matrix[:, pulse_start_idx:pulse_end_idx]
mag_avg_log_matrix_pulse_avg = np.mean(mag_avg_log_matrix_pulse, axis=1)
mag_avg_log_matrix_transient_region = mag_avg_log_matrix[:,pulse_end_idx:]
mag_avg_matrix_transient_region = 10**(mag_avg_log_matrix_transient_region) - 0.01
# tau_us_array = fit_exponential_decay_list(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, plot_index=None) #2150
tau_us_array = fit_piecewise_slider(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, show_slider=show_fit_slices)
x_axis = np.arange(mag_avg_log_matrix.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96


# ---------------- Left subplot: Line plot of pulse_avg ----------------
ax_left = fig.add_subplot(gs[0, 0])
ax_left.plot(tau_us_array, y_axis, color=tau_color, label=r'$\tau$ [$\mu$s]', linewidth=line_plot_line_width)
# ax_left.set_xlabel(r"Ringdown Lifetime $\tau$ [$\mu$s]", color=tau_color)
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
ax_top.set_xlim([3.0, 3.5])  # adjust as needed
ax_top.set_xticks([3.1, 3.4]) 

# Color top x-axis
ax_top.tick_params(axis='x', colors=transmission_color)
ax_top.spines['top'].set_color(transmission_color)

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[0, 1])

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
ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

ax_right.text(
    text_box_x, text_box_y, "Cooldown 1",  # position in axes coordinates
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
ax_right.vlines(x=pick_time, ymin=center - band_width/2, ymax=center + band_width/2, color='white', linewidth=1.5)

# ---------------- SECOND ROW PLOTS ----------------

# ---------------- Process Data ----------------

npy_name_prefix = "shipley_2nd"
display_offset = 25
pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

mag_avg_log_matrix_2 = np.load(os.path.join(matrix_data_dir, f"{npy_name_prefix}_mag_avg_log_matrix.npy"))/10
mag_avg_log_matrix_2 = mag_avg_log_matrix_2[1000:, display_offset:display_offset+465]
print(f'{npy_name_prefix}: {np.shape(mag_avg_log_matrix_2)}')

mag_avg_log_matrix_pulse_2 = mag_avg_log_matrix_2[:, pulse_start_idx:pulse_end_idx]
mag_avg_log_matrix_pulse_avg_2 = np.mean(mag_avg_log_matrix_pulse_2, axis=1)
mag_avg_log_matrix_transient_region_2 = mag_avg_log_matrix_2[:,pulse_end_idx:]
mag_avg_matrix_transient_region_2 = 10**(mag_avg_log_matrix_transient_region_2) - 0.01
# tau_us_array = fit_exponential_decay_list(mag_avg_log_matrix_transient_region, time_spacing_us=1/552.96, plot_index=None) #2150
tau_us_array_2 = fit_piecewise_slider(mag_avg_log_matrix_transient_region_2, time_spacing_us=1/552.96, show_slider=show_fit_slices)
x_axis = np.arange(mag_avg_log_matrix_2.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix_2.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96


# ---------------- Left subplot: Line plot of pulse_avg ----------------
ax_left = fig.add_subplot(gs[1, 0])
ax_left.plot(tau_us_array_2, y_axis, color=tau_color, label=r'$\tau$ [$\mu$s]', linewidth=line_plot_line_width)
# ax_left.set_xlabel(r"Ringdown Lifetime $\tau$ [$\mu$s]", color=tau_color)
ax_left.set_ylabel("Frequency [GHz]")
ax_left.set_ylim(display_freq_range)
ax_left.set_xlim(display_tau_range)
ax_left.set_xticks(display_tau_ticks)
ax_left.set_yticks(frequency_ticks)

# Color bottom x-axis
ax_left.tick_params(axis='x', colors=tau_color)
ax_left.spines['bottom'].set_color(tau_color)
ax_left.text(abcd_x, abcd_y, r'\textbf{b}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# Top x-axis (log magnitude)
ax_top = ax_left.twiny()
ax_top.plot(mag_avg_log_matrix_pulse_avg_2, y_axis, color=transmission_color, label='Avg. Log Mag', linewidth=line_plot_line_width)
# ax_top.set_xlabel("Log Magnitude [arb.]", color=transmission_color)
ax_top.set_xlim([3.0, 3.5])  # adjust as needed
ax_top.set_xticks([3.1, 3.4]) 

# Color top x-axis
ax_top.tick_params(axis='x', colors=transmission_color)
ax_top.spines['top'].set_color(transmission_color)

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[1, 1])

im = ax_right.imshow(
    mag_avg_log_matrix_2,
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
ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

ax_right.text(
    text_box_x, text_box_y, "Cooldown 2",  # position in axes coordinates
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
ax_right.vlines(x=pick_time, ymin=center - band_width/2, ymax=center + band_width/2, color='white', linewidth=1.5)

# ---------------- THIRD ROW PLOTS ----------------

# ---------------- Process Data ----------------

pulse_frequency_list = np.arange(3000, 5000, 1).tolist()

mag_avg_log_matrix_diff = mag_avg_log_matrix_2 - mag_avg_log_matrix

print(f'diff: {np.shape(mag_avg_log_matrix_diff)}')

x_axis = np.arange(mag_avg_log_matrix_2.shape[1])  # Data points index
y_axis = np.array(pulse_frequency_list[: mag_avg_log_matrix_2.shape[0]])/1e3  # Ensure correct length
x_min, x_max = x_axis[0] / 552.96, x_axis[-1] / 552.96

mag_avg_log_matrix_pulse_avg_diff = mag_avg_log_matrix_pulse_avg_2 - mag_avg_log_matrix_pulse_avg
tau_us_array_diff = tau_us_array_2 - tau_us_array


# ---------------- Left subplot: Line plot of pulse_avg ----------------
ax_left = fig.add_subplot(gs[2, 0])
ax_left.plot(tau_us_array_diff, y_axis, color=tau_color, label=r'$\tau$ [$\mu$s]', linewidth=line_plot_line_width)
ax_left.set_xlabel(r"Lifetime $\tau$ [ns]", color=tau_color)
ax_left.set_ylabel("Frequency [GHz]")
ax_left.set_ylim(display_freq_range)
ax_left.set_xlim([-150,150])
ax_left.set_xticks([-90,90])
ax_left.set_yticks(frequency_ticks)

# Color bottom x-axis
ax_left.tick_params(axis='x', colors=tau_color)
ax_left.spines['bottom'].set_color(tau_color)
ax_left.text(abcd_x, abcd_y, r'\textbf{c}', transform=ax_left.transAxes,
             fontsize=abcd_size, fontweight='bold', va='top', ha='right', color='black')

ax_left.text(i_x, i_y, r'\textbf{i}', transform=ax_left.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='black')

# Top x-axis (log magnitude)
ax_top = ax_left.twiny()
ax_top.plot(mag_avg_log_matrix_pulse_avg_diff, y_axis, color=transmission_color, label='Avg. Log Mag', linewidth=line_plot_line_width)
# ax_top.set_xlabel("Pulse Log Magnitude [arb.]", color=transmission_color)
ax_top.set_xlim([-0.01, 0.04])  # adjust as needed
ax_top.set_xticks([0, 0.03]) 

# Color top x-axis
ax_top.tick_params(axis='x', colors=transmission_color)
ax_top.spines['top'].set_color(transmission_color)

# ---------------- Right subplot: Color map ----------------
ax_right = fig.add_subplot(gs[2, 1])

im = ax_right.imshow(
    mag_avg_log_matrix_diff,
    aspect='auto',
    extent=[x_min, x_max, y_axis[0], y_axis[-1]],
    origin='lower',
    cmap='coolwarm',
    interpolation='none',
    vmin=-1.2,
    vmax=1.2,
)
ax_right.axvline(x=pulse_start_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.axvline(x=pulse_end_idx/552.96, color=pulse_mark_line_color, linestyle=pulse_mark_line_style, linewidth=pulse_mark_line_width)
ax_right.set_xlim([0, 0.8])
ax_right.set_ylim(display_freq_range)
# ax_right.set_yticks(frequency_ticks)
ax_right.set_yticks([])
ax_right.set_ylabel('')
ax_right.text(ii_x, ii_y, r'\textbf{ii}', transform=ax_right.transAxes,
             fontsize=i_ii_size, fontweight='bold', va='top', ha='right', color='white')
ax_right.set_xlabel(r"Time [$\mu$s]")
cbar = fig.colorbar(im, ax=ax_right, label=r"Log$_{10}$(A) [arb.]")

ax_right.text(
    text_box_x, text_box_y, r"Difference",  # position in axes coordinates
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
ax_right.vlines(x=pick_time, ymin=center - band_width/2, ymax=center + band_width/2, color='white', linewidth=1.5)

# ---------------- Figure save and show ----------------
file_path = os.path.join(fig_save_dir, f"Fig_cooldown_compare.png")
if save_fig:
    plt.savefig(file_path, dpi=350, bbox_inches = 'tight')
    print(f"Figure saved to: {file_path}")
# plt.show()
plt.close()
