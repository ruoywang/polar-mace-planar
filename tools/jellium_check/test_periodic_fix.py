"""Validate the periodic compensation-plane features (fix/periodic-plane-features).

Run with PYTHONPATH=$WORK/repos/polar-mace-planar so the repo mace shadows the venv.
"""

import math
import sys

import numpy as np
import torch
from ase.io import read

import mace.modules.solvent_charge_layer as scl
from mace.modules.extensions import (
    _slab_compensation_gaussian_features,
    _slab_compensation_gaussian_potential_field_nodes,
)
from mace.modules.solvent_charge_layer import (
    periodic_gaussian_layer_potential_field_nodes,
)
from graph_longrange.gto_utils import DisplacedGTOExternalFieldBlock
from graph_longrange.utils import FIELD_CONSTANT

torch.set_default_dtype(torch.float64)
print("mace from:", scl.__file__)
assert "repos/polar-mace-planar" in scl.__file__, "repo mace not shadowing venv!"

TRAIN_XYZ = "/scratch/08384/tg876840/tmp/9-final/2-NiN/5-fermi_diff/train.xyz"
SIGMA_G = 0.85
atoms = read(TRAIN_XYZ, index=0)
cell_np = np.array(atoms.get_cell())
Q = float(atoms.info["total_charge"])
area = float(np.linalg.norm(np.cross(cell_np[0], cell_np[1])))
L = float(np.linalg.norm(cell_np[2]))
V = area * L
z0 = float(atoms.get_positions()[:, 2].max()) + 0.7

cell = torch.from_numpy(cell_np).view(1, 3, 3)
pbc = torch.tensor([[True, True, False]])
tc = torch.tensor([Q])
center = torch.tensor([z0])

N = 4096
z_line = np.linspace(0.0, L, N, endpoint=False)
pos = torch.zeros((N, 3))
pos[:, 2] = torch.from_numpy(z_line)
batch = torch.zeros(N, dtype=torch.long)


def call_periodic(receiver_sigma=0.0, positions=pos, batch_=batch, tc_=tc, pbc_=pbc):
    return periodic_gaussian_layer_potential_field_nodes(
        total_charge=tc_, center=center, cell=cell, pbc=pbc_, batch=batch_,
        positions=positions, sigma_g=SIGMA_G, receiver_sigma=receiver_sigma,
        axis=2, num_grid=4096,
    )


results = {}

# --- A: match against an independent numpy FFT reference ---------------------
phi_new, field_new = call_periodic()
q_plane = -Q
a_, b_ = (0 - z0) / SIGMA_G, (L - z0) / SIGMA_G
norm = 0.5 * (math.erf(b_ / math.sqrt(2)) - math.erf(a_ / math.sqrt(2)))
gauss = np.exp(-0.5 * ((z_line - z0) / SIGMA_G) ** 2) / (
    math.sqrt(2 * math.pi) * SIGMA_G * norm
)
raw = (q_plane / area) * gauss * V
mode = np.fft.fftfreq(N, d=1.0) * N
gabs = np.abs(2 * math.pi * mode / L)
coeff = np.fft.fft(raw) / N
phi_hat = np.zeros_like(coeff)
m = gabs > 1e-14
phi_hat[m] = coeff[m] * FIELD_CONSTANT / (gabs[m] ** 2 * V)
phi_ref = np.real(np.fft.ifft(phi_hat * N))
phi_ref -= phi_ref.mean()
results["A_max_abs_diff_vs_ref_eV"] = float(
    np.abs(phi_new.numpy() - phi_ref).max()
)

# --- B: Poisson equation phi'' = -k (rho_layer - q/V) -------------------------
dz = z_line[1] - z_line[0]
phi_np = phi_new.numpy()
lap = (np.roll(phi_np, -1) - 2 * phi_np + np.roll(phi_np, 1)) / dz**2
rho_neutralized = (q_plane / area) * gauss - q_plane / V
poisson_resid = lap + FIELD_CONSTANT * rho_neutralized
results["B_poisson_resid_max_over_scale"] = float(
    np.abs(poisson_resid).max() / (FIELD_CONSTANT * np.abs(rho_neutralized).max())
)

# --- C: receiver convolution equals real-space Gaussian convolution ----------
sr = 1.3
phi_conv, _ = call_periodic(receiver_sigma=sr)
kern_z = (np.arange(N) - N // 2) * dz
kern = np.exp(-0.5 * (kern_z / sr) ** 2) / (math.sqrt(2 * math.pi) * sr) * dz
conv_ref = np.real(np.fft.ifft(np.fft.fft(phi_np) * np.fft.fft(np.fft.ifftshift(kern))))
results["C_conv_max_abs_diff_eV"] = float(np.abs(phi_conv.numpy() - conv_ref).max())

# --- D: autograd through node positions matches the returned field -----------
pos_g = pos[::64].clone().requires_grad_(True)
batch_g = torch.zeros(pos_g.shape[0], dtype=torch.long)
phi_g, field_g = call_periodic(positions=pos_g, batch_=batch_g)
(grad,) = torch.autograd.grad(phi_g.sum(), pos_g)
results["D_autograd_vs_field_max_diff"] = float(
    (grad[:, 2] - field_g[:, 2].detach()).abs().max()
)
results["D_grad_xy_zero"] = float(grad[:, :2].abs().max())

# --- E: spurious parabola is gone (curvature of phi_new - phi_ref_iso trick) --
# old isolated phi for comparison
phi_iso, _ = _slab_compensation_gaussian_potential_field_nodes(
    total_charge=tc, center=center, cell=cell, pbc=pbc, batch=batch,
    positions=pos, sigma_g=SIGMA_G, axis=2,
)
c2_old = np.polyfit(z_line, phi_iso.numpy() - phi_ref, 2)[0] * 2
c2_new = np.polyfit(z_line, phi_np - phi_ref, 2)[0] * 2
results["E_curvature_isolated_eV_A2"] = float(c2_old)
results["E_curvature_periodic_eV_A2"] = float(c2_new)

# --- F: neutral and non-slab systems give zero --------------------------------
phi0, f0 = call_periodic(tc_=torch.tensor([0.0]))
results["F_neutral_max_abs"] = float(max(phi0.abs().max(), f0.abs().max()))
phi_ns, f_ns = call_periodic(pbc_=torch.tensor([[True, True, True]]))
results["F_nonslab_max_abs"] = float(max(phi_ns.abs().max(), f_ns.abs().max()))

# --- G: feature wiring — isolated path unchanged, periodic path consistent ---
block = DisplacedGTOExternalFieldBlock(1, [1.0, 2.0], "receiver")
feats_iso, phi_i, _ = _slab_compensation_gaussian_features(
    external_field_block=block, total_charge=tc, center=center, cell=cell,
    pbc=pbc, batch=batch[:64], positions=pos[:64], sigma_g=SIGMA_G,
    feature_sigmas=[1.0, 2.0], axis=2, convention="isolated",
)
# manual re-derivation of the legacy isolated feature for sigma pair
sigma_eff = math.sqrt(SIGMA_G**2 + 1.0**2)
phi_manual, _ = _slab_compensation_gaussian_potential_field_nodes(
    total_charge=tc, center=center, cell=cell, pbc=pbc, batch=batch[:64],
    positions=pos[:64], sigma_g=sigma_eff, axis=2,
)
results["G_isolated_phi_matches_legacy"] = float((phi_i - phi_manual).abs().max())
feats_per, _, _ = _slab_compensation_gaussian_features(
    external_field_block=block, total_charge=tc, center=center, cell=cell,
    pbc=pbc, batch=batch[:64], positions=pos[:64], sigma_g=SIGMA_G,
    feature_sigmas=[1.0, 2.0], axis=2, convention="periodic",
)
results["G_feature_shapes"] = f"{tuple(feats_iso.shape)} vs {tuple(feats_per.shape)}"
results["G_features_finite"] = bool(torch.isfinite(feats_per).all())

for k, v in results.items():
    print(f"{k}: {v}")

# A tolerance: graph_longrange.FIELD_CONSTANT (1/5.526349406e-3) and the local
# POTENTIAL_FROM_DIPOLE_SCALE (4*pi*Ha/bohr) differ by 8e-9 relative; the code
# base already mixes the two families (loss.py vs energy), so the algorithmic
# agreement is checked at 1e-6 eV.
ok = (
    results["A_max_abs_diff_vs_ref_eV"] < 1e-6
    and results["B_poisson_resid_max_over_scale"] < 1e-4
    and results["C_conv_max_abs_diff_eV"] < 1e-8
    and results["D_autograd_vs_field_max_diff"] < 1e-10
    and results["D_grad_xy_zero"] == 0.0
    and abs(results["E_curvature_periodic_eV_A2"]) < 1e-8
    and results["F_neutral_max_abs"] == 0.0
    and results["F_nonslab_max_abs"] == 0.0
    and results["G_isolated_phi_matches_legacy"] == 0.0
    and results["G_features_finite"]
)
print("ALL CHECKS PASS" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
