"""Dump the closure parity pack on cal_18: the 3-D input fields plus numpy
reference closure quantities (A_scr, prior, S_ion_z, plane variances,
solute z-dipole)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/work/08384/tg876840/ls6/repos/cep-dip-python-pb")
DATA = Path("/scratch/08384/tg876840/tmp/b-1D_PB/data/case_cal18")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "pb_check"))

from tools.vasp_volumetric import read_vasp_volumetric
from pure_python.config import load_config
from pure_python.dipole_correction import valence_ion_dipole_cart
from pure_python.grid import EDEPS, Grid, normalized_gaussian_kernel_g
from pure_python.pb import create_cavity, derived_params, local_field_factor
from pure_python.potcar import read_potcar
from pure_python.solute_potential import solute_potential_g
from solve_1d_frozen_response import langevin_g, plane_avg, response_coefficient_3d

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "exp_prior_profiles" / "closure_parity_pack.npz"

cfg = load_config(str(REPO / "pure_python/configs/cal18.json"))
chg = read_vasp_volumetric(str(DATA / "CHGCAR"))
grid3 = Grid(chg.cell, chg.grid)
valence_values = chg.values.reshape(chg.grid, order="F")
entries = read_potcar(str(DATA / "POTCAR"))
positions = np.asarray(cfg["positions_direct"], dtype=float)
counts = list(cfg["counts"])
zvals = [e.zval for e in entries]
cvhar_g, dencor = solute_potential_g(grid3, valence_values, entries, counts, positions, None)
cvhar3 = grid3.ifft_real_full(cvhar_g)
n_e_density = (valence_values + dencor) / grid3.volume
params = derived_params(cfg["solvation"])
s_ion3, s_diel3, _ = create_cavity(np.clip(n_e_density, 0.0, None), grid3, params, None)

a3_zero, _ = response_coefficient_3d(np.zeros_like(cvhar3), s_diel3, grid3, params)
eps3 = 1.0 + EDEPS * a3_zero
sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
w_b3 = normalized_gaussian_kernel_g(grid3, sigma_b)
_, _, ez_v, emag_v = grid3.grad_from_recip(-np.conj(w_b3) * grid3.fft(cvhar3))
emag_scr = emag_v / eps3
ez_scr = ez_v / eps3
f_loc = local_field_factor(emag_scr, params)
y = float(params["PBETA"]) * emag_scr * f_loc
g = langevin_g(y)
poe = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
a3_scr = f_loc * float(params["N_MOL"]) * s_diel3 * poe

def pvar(a):
    m = a.mean(axis=(0, 1))
    return (a * a).mean(axis=(0, 1)) - m * m

A_scr = plane_avg(a3_scr).ravel()
S_ion_z = plane_avg(s_ion3).ravel()
prior = (plane_avg(a3_scr * ez_scr) - plane_avg(a3_scr) * plane_avg(ez_scr)).ravel()
var_chi = pvar(a3_scr)
var_e = pvar(ez_scr)

# per-atom zvals list for the dipole reference (counts = [1]*natoms form)
zv_atoms = []
for zv, c in zip(zvals, counts):
    zv_atoms += [zv] * c
vid = valence_ion_dipole_cart(
    valence_values + dencor, positions, zv_atoms, [1] * len(zv_atoms), chg.cell)

np.savez(
    OUT,
    n_e_density=n_e_density.astype(np.float64),
    cvhar3=cvhar3.astype(np.float64),
    cell=chg.cell,
    A_scr=A_scr, S_ion_z=S_ion_z, prior=prior,
    var_chi=var_chi, var_e=var_e,
    val_ion_dipole_z=float(vid[2]),
    positions_direct=positions,
    zv_atoms=np.asarray(zv_atoms, dtype=float),
)
print(f"wrote {OUT} (fields {n_e_density.shape})")
