#!/usr/bin/env python3
"""
two_tls_floquet_parallel.py
───────────────────────────
• Panel a: heat-map of ⟨(σ₁⁺+σ₂⁺)(σ₁⁻+σ₂⁻)⟩ vs time (µs) and ω_d  
• Panel b: Floquet quasi-energies vs ω_d  
Both panels share exactly the same y-axis limits *and* y-tick locations.
"""

# ───────────────────── imports & style ────────────────────────────────
import os, numpy as np, matplotlib, multiprocessing as mp
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qutip import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

plt.rcParams.update({
    "font.size": 40,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ───────────────── system parameters ───────────────────────────────────
omega_tls_1, omega_tls_2 = 3.0, 4.0  # GHz
J, Omega_amp             = 0.05, 0.1
T_drive, T_total         = 100.0, 400     # ns
dt                       = 0.01            # ns
omega_d_vals             = np.linspace(2.0, 5.0, 300)
gamma_collective         = 0.002           # ns⁻¹

# ───────────────── operators / Hamiltonian ─────────────────────────────
sx1 = tensor(sigmax(), qeye(2));  sx2 = tensor(qeye(2), sigmax())
sz1 = tensor(sigmaz(), qeye(2));  sz2 = tensor(qeye(2), sigmaz())
sp1 = tensor(sigmap(),  qeye(2)); sp2 = tensor(qeye(2), sigmap())
sm1 = tensor(sigmam(),  qeye(2)); sm2 = tensor(qeye(2), sigmam())
sp_tot, sm_tot = sp1 + sp2, sm1 + sm2
collective_excitation = sp_tot * sm_tot

c_ops = [np.sqrt(gamma_collective) * sm_tot]

H_static = 0.5*omega_tls_1*sz1 + 0.5*omega_tls_2*sz2 + J*sx1*sx2
psi0     = H_static.eigenstates()[1][0]      # ground state

# time grid (ns) → convert to µs for plotting
tlist_ns = np.arange(0.0, T_total, dt)
tlist_us = tlist_ns / 1e3
opts = {"nsteps": 5000}

# ───────────────── worker executed in parallel ─────────────────────────
def run_one(idx_and_w):
    idx, omega_d = idx_and_w

    T_ramp = 1.0  # ns
    def drive_coeff(t, args):
        if t < T_ramp:
            return 0.5*Omega_amp*(1-np.cos(np.pi*t/T_ramp))*np.cos(omega_d*t)
        elif t < T_drive:
            return Omega_amp*np.cos(omega_d*t)
        return 0.0

    H_full = [H_static, [sx1+sx2, drive_coeff]]

    sig = mesolve(H_full, psi0, tlist_ns, c_ops, [collective_excitation],
                  options=opts).expect[0].real

    try:
        U = propagator(H_full, T_drive, args=None, c_ops=[], options=opts)
        phases = np.angle(np.linalg.eigvals(U.full()))
        folded = (phases + np.pi) % (2*np.pi) - np.pi
        quasi  = np.sort(folded / T_drive)
    except Exception:
        quasi = np.full(4, np.nan)

    return idx, sig, quasi

# ───────────────── parallel sweep over ω_d ─────────────────────────────
num_ω = len(omega_d_vals)
sigmat  = np.zeros((num_ω, len(tlist_ns)))
quasien = np.zeros((num_ω, 4))

with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
    futures = [pool.submit(run_one, (i, w)) for i, w in enumerate(omega_d_vals)]
    for fut in tqdm(as_completed(futures), total=num_ω, desc="Drive sweep"):
        idx, sig, q = fut.result()
        sigmat[idx] = sig
        quasien[idx] = q

# ───────────────── plotting ────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(25, 10),
                        gridspec_kw={"width_ratios": [2.4, 1.5]})

# shared y-axis limits & ticks
y_min, y_max = omega_d_vals[0], omega_d_vals[-1]
y_ticks = np.linspace(y_min, y_max, 3)     # 6 evenly spaced ticks

# panel (a): heat-map
im = axs[0].imshow(sigmat,
                   extent=[tlist_us[0], tlist_us[-1], y_min, y_max],
                   origin="lower", aspect="auto", cmap="inferno")
axs[0].axvline(T_drive/1e3, ls="--", color="white", lw=2.0)
axs[0].set_xlabel(r"Time [$\mu$s]")
axs[0].set_ylabel(r"Drive Frequency [GHz]")
axs[0].set_yticks(y_ticks)
axs[0].tick_params(axis='both', which='major', pad=12)   # add extra padding between tick labels and axes

cbar = fig.colorbar(im, ax=axs[0], fraction=0.05, pad=0.02)
cbar.set_label(r"$\langle \sigma^+ \sigma^- \rangle$", labelpad=14)

# panel (b): Floquet quasi-energies
for col in range(quasien.shape[1]):
    axs[1].plot(quasien[:, col], omega_d_vals, lw=3.0, label=rf"$Q_{{{col}}}$")
axs[1].set_xlabel("Quasi-Energy [GHz]")
axs[1].set_xlim(np.nanmin(quasien), np.nanmax(quasien))
axs[1].set_ylim(y_min, y_max)
axs[1].set_yticks(y_ticks)
axs[1].legend(fontsize=30, loc="center left", bbox_to_anchor=(1.02, 0.5))

# panel labels
axs[0].text(-0.15, 1.1, r"\textbf{a}", transform=axs[0].transAxes,
            fontsize=48, fontweight="bold")
axs[1].text(-0.15, 1.1, r"\textbf{b}", transform=axs[1].transAxes,
            fontsize=48, fontweight="bold")

plt.tight_layout()
os.makedirs("../plots", exist_ok=True)
outfile = "../plots/floquet_quasienergies_parallel.png"
plt.savefig(outfile, dpi=180)
plt.close()
print("✓ Figure saved →", outfile)
