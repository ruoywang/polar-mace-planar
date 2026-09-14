"""Offline diagnostic (user plan step 5, 2026-09-13): after the energy_weight-1000
run, if the paired charging residual is still mostly a bias, fit ONE
coefficient on the TRAINING frames' total energies,

    E_diag = E_model + a * dN_e,

with mace's per-atom normalisation and unit sample weights (the same
objective as the training energy term, i.e. minimise sum_i ((E_model,i + a
dN_e,i - E_dft,i) / N_i)^2 over training frames), then report what it does
to the absolute energies per state and to the paired charging energy on
train AND val (val never enters the fit). Two definitions of dN_e are
reported because the data carry no system-specified fixed electron count:
  per-frame  : dN_e = -total_charge of the frame (what the model sees)
  per-system : dN_e = the mean of -total_charge over that system's charged
               frames (NiN44, NiN88), the same fixed number for every charged
               frame of a system; 0 for neutral frames
Inputs: the per-frame npz written by ab_eval.py (KIT_STRIDE=1 for the train
set). No model is touched. The fitted a shifts dE/dN by a, i.e. the
chemical potential the model implies; whether that is consistent with the
Fermi-level supervision is discussed in the ledger, not decided here.
"""
import sys
import numpy as np
from ase.io import read
npz = sys.argv[1]                       # ab_eval_<tag>.npz with train stride 1 and val complete
RUN = sys.argv[2] if len(sys.argv) > 2 else "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/ab_head_nn_e1000"
z = np.load(npz, allow_pickle=True); rows = z["rows"]
R = {(r[0], int(r[1])): r for r in rows}        # (split, sid) -> (split, sid, state, natoms, E_model, E_dft, F_rmse, F_max)
q = {}
for sp in ("train", "val"):
    for a in read(f"{RUN}/data/{sp}.xyz", ":"):
        q[int(a.info["sample_id"])] = float(a.info.get("total_charge", 0.0))
def dn_per_frame(sid): return -q[sid]
sys_of = {sid: ("NiN88" if R[k][3] == 339 else "NiN44") for k in R for sid in [k[1]]}
mean_dn = {s: np.mean([-q[k[1]] for k in R if k[0] == "train" and sys_of[k[1]] == s and abs(q[k[1]]) > 1e-6]) for s in ("NiN44", "NiN88")}
def dn_per_system(sid): return mean_dn[sys_of[sid]] if abs(q[sid]) > 1e-6 else 0.0
print(f"frames: train {sum(1 for k in R if k[0]=='train')}, val {sum(1 for k in R if k[0]=='val')};  per-system dN_e (mean over charged train frames): {mean_dn}")
def fit(dn):
    tr = [k for k in R if k[0] == "train"]
    x = np.array([dn(k[1]) / R[k][3] for k in tr]); y = np.array([(R[k][5] - R[k][4]) / R[k][3] for k in tr])   # minimise sum ((E_m + a dN - E_dft)/N)^2
    a = float((x * y).sum() / (x * x).sum()); return a
def report(label, a, dn):
    print(f"\n=== {label}: a = {a:+.4f} eV per electron (fitted on train energies only) ===")
    for sp in ("train", "val"):
        print(f"  {sp}: per-state E error (meV/atom) rmse / bias, before -> after")
        for st in ("charged NiN44", "charged NiN88", "neutral NiN44"):
            ks = [k for k in R if k[0] == sp and R[k][2] == st]
            e0 = np.array([1e3 * (R[k][4] - R[k][5]) / R[k][3] for k in ks]); e1 = np.array([1e3 * (R[k][4] + a * dn(k[1]) - R[k][5]) / R[k][3] for k in ks])
            print(f"    {st:>14} ({len(ks):3d}): {np.sqrt((e0**2).mean()):6.2f} / {e0.mean():+6.2f}  ->  {np.sqrt((e1**2).mean()):6.2f} / {e1.mean():+6.2f}")
        pairs = sorted(k[1] for k in R if k[0] == sp and k[1] < 600 and (sp, k[1] + 600) in R)
        r0 = np.array([(R[(sp,k)][4] - R[(sp,k+600)][4]) - (R[(sp,k)][5] - R[(sp,k+600)][5]) for k in pairs])
        r1 = np.array([(R[(sp,k)][4] + a*dn(k) - R[(sp,k+600)][4] - a*dn(k+600)) - (R[(sp,k)][5] - R[(sp,k+600)][5]) for k in pairs])
        print(f"  {sp}: paired dE ({len(pairs)} pairs) rmse / bias / mean-removed: {np.sqrt((r0**2).mean()):.4f} / {r0.mean():+.4f} / {r0.std():.4f}  ->  {np.sqrt((r1**2).mean()):.4f} / {r1.mean():+.4f} / {r1.std():.4f}")
for label, dn in (("per-frame dN_e = -total_charge", dn_per_frame), ("per-system fixed dN_e", dn_per_system)):
    report(label, fit(dn), dn)
print("\nNOTE: the correction does not touch forces; it changes dE/dN_e by a (the implied chemical potential),")
print("so its consistency with the Fermi-level supervision and the energy reference has to be argued separately.")
