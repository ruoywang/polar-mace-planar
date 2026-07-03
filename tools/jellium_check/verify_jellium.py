"""Verify the feature-level jellium inconsistency of the planar solvent layer.

Compares, for a real charged sample (2-NiN train set):
  phi_code(z): isolated erf potential of the compensation plane, exactly as
               produced by mace.modules.extensions._slab_compensation_gaussian_potential_field_nodes
  phi_ref(z):  periodic G!=0 (1D FFT) potential of the same plane density,
               i.e. the convention used for the explicit density features.
The slab dipole-correction feature (k*mu_plane*z/V) is identical in both
conventions and cancels in the difference, so

  d(z) = phi_code(z) - phi_ref(z)

should be a parabola with curvature -k*q_plane/V (k = FIELD_CONSTANT,
q_plane = -total_charge), plus an arbitrary linear+constant gauge.
"""

import json
import math
import sys

import numpy as np
import torch
from ase.io import read

from graph_longrange.utils import FIELD_CONSTANT
from mace.modules.extensions import _slab_compensation_gaussian_potential_field_nodes

torch.set_default_dtype(torch.float64)

TRAIN_XYZ = "/scratch/08384/tg876840/tmp/9-final/2-NiN/5-fermi_diff/train.xyz"
SIGMA_G = 0.85
NGRID = 8192

atoms = read(TRAIN_XYZ, index=0)
cell = np.array(atoms.get_cell())
Q_total = float(atoms.info["total_charge"])
q_plane = -Q_total
z_atoms = atoms.get_positions()[:, 2]
area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
L = float(np.linalg.norm(cell[2]))
V = area * L

# plane position: top of the explicit system + the default mean shift
z0 = float(z_atoms.max()) + 0.7

# ---- phi_code: the actual model function on a dense z line ------------------
z_line = np.linspace(0.0, L, NGRID, endpoint=False)
pos = torch.zeros((NGRID, 3))
pos[:, 2] = torch.from_numpy(z_line)
phi_code_t, field_code_t = _slab_compensation_gaussian_potential_field_nodes(
    total_charge=torch.tensor([Q_total]),
    center=torch.tensor([z0]),
    cell=torch.from_numpy(cell).view(1, 3, 3),
    pbc=torch.tensor([[True, True, False]]),
    batch=torch.zeros(NGRID, dtype=torch.long),
    positions=pos,
    sigma_g=SIGMA_G,
    axis=2,
)
phi_code = phi_code_t.numpy()

# ---- phi_ref: periodic G!=0 solution of the same (truncated) layer ----------
# truncated normalized gaussian, same construction as
# _slab_compensation_periodic_1d_energy_radial
a = (0.0 - z0) / SIGMA_G
b = (L - z0) / SIGMA_G
norm = 0.5 * (math.erf(b / math.sqrt(2)) - math.erf(a / math.sqrt(2)))
gauss = np.exp(-0.5 * ((z_line - z0) / SIGMA_G) ** 2) / (
    math.sqrt(2 * math.pi) * SIGMA_G * norm
)
rho = (q_plane / area) * gauss            # e/A^3, integrates to q_plane/area per area
raw = rho * V                             # VASP-style raw profile (mean -> total charge)

mode = np.fft.fftfreq(NGRID, d=1.0) * NGRID
gabs = np.abs(2.0 * math.pi * mode / L)
coeff = np.fft.fft(raw) / NGRID
phi_hat = np.zeros_like(coeff)
mask = gabs > 1e-14
phi_hat[mask] = coeff[mask] * FIELD_CONSTANT / (gabs[mask] ** 2 * V)
phi_ref = np.real(np.fft.ifft(phi_hat * NGRID))
phi_ref -= phi_ref.mean()

# ---- difference and parabola fit --------------------------------------------
d = phi_code - phi_ref
c2, c1, c0 = np.polyfit(z_line, d, 2)
curv_fit = 2.0 * c2
curv_pred = -FIELD_CONSTANT * q_plane / V
resid = d - np.polyval([c2, c1, c0], z_line)

# physically meaningful distortion: remove only the linear gauge, look at the
# spread across the explicit atoms
lin = np.polyfit(z_line, d, 1)
d_nolin = d - np.polyval(lin, z_line)
at_lo, at_hi = z_atoms.min(), z_atoms.max()
sel = (z_line >= at_lo) & (z_line <= at_hi)
spread_atoms = float(d_nolin[sel].max() - d_nolin[sel].min())
spread_cell = float(d_nolin.max() - d_nolin.min())

out = {
    "sample": {"Q_total_e": Q_total, "area_A2": area, "L_A": L, "V_A3": V,
               "z_atoms_min": float(at_lo), "z_atoms_max": float(at_hi),
               "z_plane": z0, "n_atoms": len(atoms)},
    "curvature_fit_eV_per_A2": float(curv_fit),
    "curvature_predicted_kQ_over_V": float(curv_pred),
    "curvature_rel_err": float(abs(curv_fit - curv_pred) / abs(curv_pred)),
    "parabola_fit_residual_rms_eV": float(np.sqrt((resid ** 2).mean())),
    "spurious_spread_over_atoms_eV": spread_atoms,
    "spurious_spread_over_cell_eV": spread_cell,
    "field_constant": float(FIELD_CONSTANT),
}
print(json.dumps(out, indent=2))

np.savez(
    "jellium_check_data.npz",
    z=z_line, d=d, d_nolin=d_nolin, phi_code=phi_code, phi_ref=phi_ref,
    z_atoms=z_atoms, z_plane=z0, curv_fit=curv_fit, curv_pred=curv_pred,
    Q_total=Q_total, V=V,
)
