import os, numpy as np, multiprocessing as mp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qutip import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

plt.rcParams.update({
    "font.size": 40,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ───────────────────────── parameters ──────────────────────────────────
omega1, omega2        = 3.0, 4.0          # TLS splittings [GHz]
j_coupling            = 0.05              # coupling (σˣ₁σˣ₂)    [GHz]
drive_amp             = 0.10              # drive amplitude       [arb.]
t_drive_ns, t_total_ns = 100.0, 400.0     # ns
dt_ns                 = 0.01              # time step             [ns]
omega_d_vals          = np.linspace(2.0, 5.0, 300)   # drive sweep [GHz]
gamma_collective      = 0.002             # collective decay      [ns⁻¹]

# ───────────────────────── operators / static H ───────────────────────
sx1 = tensor(sigmax(), qeye(2));  sx2 = tensor(qeye(2), sigmax())
sz1 = tensor(sigmaz(), qeye(2));  sz2 = tensor(qeye(2), sigmaz())
sp1 = tensor(sigmap(),  qeye(2)); sp2 = tensor(qeye(2), sigmap())
sm1 = tensor(sigmam(),  qeye(2)); sm2 = tensor(qeye(2), sigmam())

sp_tot, sm_tot = sp1 + sp2, sm1 + sm2
pop_op = sp_tot * sm_tot                                    # ⟨σ⁺σ⁻⟩

h_static = 0.5*omega1*sz1 + 0.5*omega2*sz2 + j_coupling*sx1*sx2
psi0     = h_static.eigenstates()[1][0]

c_ops = [np.sqrt(gamma_collective) * sm_tot]
solver_opts = {"nsteps": 5000}

# time grids
t_ns = np.arange(0.0, t_total_ns, dt_ns)
t_us = t_ns / 1e3                                   # μs for plotting

# ───────────────────────── parallel worker ────────────────────────────
def run_single(task):
    idx, omega_d = task

    def drive_coeff(t, args):
        """time-dependent drive: simple square pulse with cosine"""
        if t <= t_drive_ns:
            return drive_amp * np.cos(omega_d * t)
        return 0.0

    h_full = [h_static, [sx1 + sx2, drive_coeff]]

    pop = mesolve(h_full, psi0, t_ns, c_ops, [pop_op],
                  options=solver_opts).expect[0].real

    try:
        u = propagator(h_full, t_drive_ns, c_ops=[], options=solver_opts)
        phases = np.angle(np.linalg.eigvals(u.full()))
        folded = ((phases + np.pi) % (2*np.pi) - np.pi) / t_drive_ns
        quasi  = np.sort(folded)
    except Exception:
        quasi = np.full(4, np.nan)

    return idx, pop, quasi

# ───────────────────────── simulate sweep ─────────────────────────────
n_freq = len(omega_d_vals)
heat   = np.zeros((len(t_ns), n_freq))
quasi  = np.zeros((n_freq, 4))

with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
    futures = [pool.submit(run_single, (i, w)) for i, w in enumerate(omega_d_vals)]
    for fut in tqdm(as_completed(futures), total=n_freq, desc="drive sweep"):
        idx, pop_row, q_row = fut.result()
        heat[:, idx]  = pop_row
        quasi[idx]    = q_row

# ───────────────────────── plotting ───────────────────────────────────
fig = plt.figure(figsize=(20, 18))
gs  = fig.add_gridspec(2, 2, width_ratios=[30, 0.8],
                       height_ratios=[2.0, 1.3], hspace=0.2, wspace=0.05)

ax_heat = fig.add_subplot(gs[0, 0])
ax_quasi= fig.add_subplot(gs[1, 0])
cax     = fig.add_subplot(gs[0, 1])          # colour-bar axis

# panel (a) heat-map
im = ax_heat.imshow(heat,
                    extent=[omega_d_vals[0], omega_d_vals[-1],
                            t_us[0], t_us[-1]],
                    origin="lower", aspect="auto", cmap="inferno")
ax_heat.axvline(omega1, ls="--", color="white", lw=2, alpha=0.5)  # marker at ω₁
ax_heat.axvline(omega2, ls="--", color="white", lw=2, alpha=0.5)  # marker at ω₁
ax_heat.axhline(t_drive_ns/1e3, ls='--', color='white', lw=2)
ax_heat.set_ylabel(r"Time [$\mu$s]", labelpad=20)
ax_heat.tick_params(axis='both', pad=12)

cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$\langle \sigma^{+}\sigma^{-} \rangle$", labelpad=12)

# panel (b) quasi-energies
for k in range(quasi.shape[1]):
    ax_quasi.plot(omega_d_vals, quasi[:, k]*1e3, lw=4.0, label=fr"$Q_{{{k}}}$")
ax_quasi.set_xlabel(r"Drive Frequency [GHz]")
ax_quasi.set_ylabel("Quasi-Energy [MHz]")
ax_quasi.set_xlim(omega_d_vals[0], omega_d_vals[-1])
ax_quasi.legend(fontsize=28, ncol=2, loc="upper left", frameon=False)

# panel labels
ax_heat.text(-0.12, 1.05, r"\textbf{a}", transform=ax_heat.transAxes,
             fontsize=70, fontweight="bold")
ax_quasi.text(-0.12, 1.05, r"\textbf{b}", transform=ax_quasi.transAxes,
              fontsize=70, fontweight="bold")

fig.subplots_adjust(left=0.12, right=0.92, top=0.93, bottom=0.07)
os.makedirs("../plots", exist_ok=True)
fig.savefig("../plots/floquet_quasienergies_column.png", dpi=300)
plt.close(fig)
print("✓ Figure saved → ../plots/floquet_quasienergies_column.png")
