"""Dump a solver parity pack on cal_18: all inputs the torch Solver1D needs,
plus numpy solve_frozen reference outputs for two P_off choices (exact
anchor, screened-vacuum prior). Also measures C_unit (the exact linear
coefficient ef_z = C_unit * d_mix of the Ewald dipole correction) and
verifies its linearity.
"""
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
from pure_python.dipole_correction import (
    EwaldDipoleMixer,
    cdipol_indmin_from_center,
    cdipol_potential_1d,
    solvent_moments,
    valence_ion_dipole_cart,
)
from pure_python.grid import EDEPS, Grid, normalized_gaussian_kernel_g
from pure_python.pb import create_cavity, derived_params, local_field_factor
from pure_python.potcar import read_potcar
from pure_python.solute_potential import solute_potential_g
from solve_1d_frozen_response import (
    langevin_g,
    plane_avg,
    response_coefficient_3d,
    solve_frozen,
)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "exp_prior_profiles" / "solver_parity_pack.npz"

cfg = load_config(str(REPO / "pure_python/configs/cal18.json"))
chg = read_vasp_volumetric(str(DATA / "CHGCAR"))
phi_ref = read_vasp_volumetric(str(DATA / "PHI"))
grid3 = Grid(chg.cell, chg.grid)
valence_values = chg.values.reshape(chg.grid, order="F")
entries = read_potcar(str(DATA / "POTCAR"))
positions = np.asarray(cfg["positions_direct"], dtype=float)
counts = list(cfg["counts"])
zvals = [e.zval for e in entries]
cvhar_g, dencor = solute_potential_g(grid3, valence_values, entries, counts, positions, None)
cvhar3 = grid3.ifft_real_full(cvhar_g)
n_e = (valence_values + dencor) / grid3.volume
params = derived_params(cfg["solvation"])
s_ion3, s_diel3, _ = create_cavity(n_e, grid3, params, None)
del n_e, dencor

nz = grid3.shape[2]
phi_ref3 = phi_ref.values.reshape(phi_ref.grid, order="F")
phi_ref_z = phi_ref3.mean(axis=(0, 1))
a3_f, ez3_f = response_coefficient_3d(phi_ref3, s_diel3, grid3, params)
del phi_ref3
pz_ref, ez_ref = plane_avg(a3_f * ez3_f), plane_avg(ez3_f)
del a3_f, ez3_f
a3_z, _ = response_coefficient_3d(np.zeros_like(cvhar3), s_diel3, grid3, params)
eps3 = 1.0 + EDEPS * a3_z
del a3_z
sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
w_b3 = normalized_gaussian_kernel_g(grid3, sigma_b)
_, _, ez_v, emag_v = grid3.grad_from_recip(-np.conj(w_b3) * grid3.fft(cvhar3))
del w_b3
emag_scr = emag_v / eps3
ez_scr3 = ez_v / eps3
del emag_v, ez_v, eps3
f_loc = local_field_factor(emag_scr, params)
y = float(params["PBETA"]) * emag_scr * f_loc
g = langevin_g(y)
poe = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
a3_scr = f_loc * float(params["N_MOL"]) * s_diel3 * poe
del emag_scr, f_loc, y, g, poe
A_scr = plane_avg(a3_scr)
p_prior = plane_avg(a3_scr * ez_scr3) - A_scr * plane_avg(ez_scr3)
del a3_scr, ez_scr3
p_exact = pz_ref - A_scr * ez_ref

val_ion_dipole = valence_ion_dipole_cart(valence_values, positions, zvals, counts, chg.cell)
val_ion_dipole[0:2] = 0.0
del valence_values
center_abs = 0.5 * chg.cell[0] + 0.5 * chg.cell[1] + 0.5 * chg.cell[2]
length_z = float(np.linalg.norm(chg.cell[2]))
q_sol = float(cfg["q_sol"])
grid1 = Grid(chg.cell, (1, 1, nz))
s_ion1 = plane_avg(s_ion3)
cvhar1 = plane_avg(cvhar3)
del s_ion3, s_diel3, cvhar3, grid3

# ---- C_unit: linear coefficient of the dipole-correction field ----
m = EwaldDipoleMixer.fresh()
m.dipolc_tmp[2] = 1.0
_, ef1 = m.ewald_dipol(np.array([0.0, 0.0, 1.0]), chg.cell, 3)
m2 = EwaldDipoleMixer.fresh()
m2.dipolc_tmp[2] = 2.0
_, ef2 = m2.ewald_dipol(np.array([0.0, 0.0, 2.0]), chg.cell, 3)
c_unit = float(ef1[2])
lin_err = abs(ef2[2] - 2.0 * ef1[2]) / abs(ef2[2])
print(f"C_unit = {c_unit:.12e}, linearity rel err {lin_err:.2e}")
assert lin_err < 1.0e-12

# ---- numpy reference solves (exact / prior) with the standard protocol ----
def np_solve(p_off):
    indmin_z = cdipol_indmin_from_center(nz, 0.5)
    mixer = EwaldDipoleMixer.fresh()
    qsol_cache = 0.0
    dsol_cache = np.zeros(3)
    phi_total = np.zeros(grid1.shape)
    for _ in range(5):
        dip = val_ion_dipole.copy()
        dip[2] += dsol_cache[2] - qsol_cache * center_abs[2]
        _, ef = mixer.ewald_dipol(dip, chg.cell, 3)
        cvdip_z = cdipol_potential_1d(nz, length_z, ef[2], indmin_z)
        phi_sol = cvhar1 + cvdip_z[None, None, :]
        phi_total, n_b, n_ion, _h = solve_frozen(
            phi_total, phi_sol, s_ion1, A_scr, grid1, params, q_sol,
            1.0e-3, 12, 40, p_off)
        qsol_cache, dsol_cache = solvent_moments(n_b + n_ion, chg.cell)
    return phi_total.ravel(), n_b.ravel() / grid1.volume, n_ion.ravel() / grid1.volume

phi_ex, nb_ex, nion_ex = np_solve(p_exact)
phi_pr, nb_pr, nion_pr = np_solve(p_prior)
print("numpy solves done")

param_keys = ["ZBETA", "theta_b", "n_max", "invBETA", "R_B", "A_K"]
np.savez(
    OUT,
    nz=nz, cell=chg.cell, length_z=length_z, volume=grid1.volume,
    cvhar_z=cvhar1.ravel(), s_ion_z=s_ion1.ravel(),
    A_scr=A_scr.ravel(), p_exact=p_exact.ravel(), p_prior=p_prior.ravel(),
    q_sol=q_sol, val_ion_dipole_z=float(val_ion_dipole[2]),
    center_z=float(center_abs[2]), c_unit=c_unit,
    indmin=cdipol_indmin_from_center(nz, 0.5),
    LNLION=int(bool(params["LNLION"])), LION=int(bool(params["LION"])),
    **{k: float(params[k]) for k in param_keys},
    ref_phi_exact=phi_ex, ref_nb_exact=nb_ex, ref_nion_exact=nion_ex,
    ref_phi_prior=phi_pr, ref_nb_prior=nb_pr, ref_nion_prior=nion_pr,
    phi_ref_z=phi_ref_z,
)
print(f"wrote {OUT}")
