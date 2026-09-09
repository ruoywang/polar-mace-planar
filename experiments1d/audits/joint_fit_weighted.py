"""Joint OPTIMISATION (not just joint measurement) of the residual basis.

Correction to the previous screen (user review 2026-09-09): that run minimised
the POINT error subject to one linear cross-energy constraint, and merely
measured the self-energy and amplitude afterwards. It therefore showed only
that the selected solution was bad — not that the span contains no acceptable
solution. This run puts the energy-carrying region into the OBJECTIVE.

The cross energy is int n*phi dV, so the region that carries it is where |phi|
is large. Fitting with per-point weights w(p) makes the objective care about
exactly that region:
    w = 1        plain L2 (uniform in volume; the previous baseline)
    w = |phi|    weights each point by the potential it sits in
    w = phi^2    stronger emphasis on the deep-potential region
For every (basis, weighting) the weighted ridge solution is reported, and also
its cross-energy-constrained variant along the weighted H^-1 g direction, so a
weighting that overshoots or undershoots the reference cross energy can still
be brought onto it. All four scores are reported for every solution:
point rms over all points, point rms over the deepest 10% of |phi| points,
cross energy, self-energy, charge amplitude.

Bases (each strictly contains the previous one's span, so its optimum can
never be worse in the same objective):
    A = env_b + s_ion  |  A + s_diel  |  A + env_b^0.5(smooth)
Target: the DFT lateral solvent charge, plane mean removed (the 3-D part owns
the lateral shortfall only).

Decision rule: the span contains an acceptable solution if some (basis,
weighting) reaches cross/ref, self/ref and |q|/ref all near 1 without a large
degradation of the point rms. If none does, that is much stronger evidence for
a capacity limit than the previous run gave — still not a proof of
non-existence.  cwd = gate_bl.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")
sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import (
    normalized_gradient_envelope, poisson_phi_periodic, _interp3_periodic)

RIDGE = 1.0e-7
EPS_SQ = 1.0e-3
GCE44 = "/scratch/08384/tg876840/tmp/2-NiN_single/1-44_GCE"
GCE88 = "/scratch/08384/tg876840/tmp/2-NiN_single/2-88_GCE"
NEU = "/scratch/08384/tg876840/tmp/2-NiN_single/5-44_neutral_withsolv"
FRAMES = [(1, f"{GCE44}/cal_1", "NiN44 q=-1.00", 80000),
          (601, f"{NEU}/cal_1", "neutral", 80000),
          (201, f"{GCE88}/cal_1", "NiN88 q=-1.00", 45000)]

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(f="checkpoints/s3d_gate_bl2_run-123.model", map_location=device).to(device)
ck = torch.load("checkpoints/s3d_gate_bl2_run-123_epoch-33.pt", map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True

cap = {}
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["cvhar3"] = cv.detach().clone(); cap["grid"] = grid
    return _clo(n_e, cv, grid, params, tp_)
PB.closure_from_fields = wrap_clo

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
want = {sid: None for sid, _, _, _ in FRAMES}
for split in ("train", "val"):
    for a in read(f"data/{split}.xyz", ":"):
        sid = int(a.info.get("sample_id", -1))
        if sid in want and want[sid] is None:
            want[sid] = a
    if all(v is not None for v in want.values()):
        break


def read_grid(path):
    f = open(path); f.readline(); s = float(f.readline())
    lat = np.array([[float(x) for x in f.readline().split()] for _ in range(3)]) * s
    line = f.readline().split()
    try:
        counts = [int(x) for x in line]
    except ValueError:
        counts = [int(x) for x in f.readline().split()]
    nat = sum(counts); f.readline()
    for _ in range(nat):
        f.readline()
    f.readline()
    nx, ny, nz = [int(x) for x in f.readline().split()]
    need = nx * ny * nz; vals = []
    while len(vals) < need:
        vals.extend(float(x) for x in f.readline().split())
    return lat, torch.tensor(np.ascontiguousarray(
        np.array(vals[:need]).reshape(nz, ny, nx).transpose(2, 1, 0)))


def to_shape(t, shape):
    if tuple(t.shape) == tuple(shape):
        return t
    return torch.nn.functional.interpolate(
        t[None, None], size=tuple(shape), mode="trilinear",
        align_corners=False)[0, 0].contiguous()


sigmas = [float(x) for x in (model.solvent3d_sigmas.tolist()
                             if torch.is_tensor(model.solvent3d_sigmas)
                             else model.solvent3d_sigmas)]
BASES = [("A", [0, 1]), ("A+s_diel", [0, 1, 2]), ("A+env^0.5", [0, 1, 3])]
WTS = ["1", "|phi|", "phi^2"]
rows = []
for sid, dftdir, tag, NPTS in FRAMES:
    a = want[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        model(b.to_dict(), compute_force=False, training=False)
    grid = cap["grid"]
    cell64 = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    shape = tuple(shape_ := grid.shape); nx, ny, nz = shape
    V = grid.volume; dV = V / float(np.prod(shape))
    phi = -(cap["cvhar3"] - cap["cvhar3"].mean())
    cavd = grid._solv3d_cavity
    env_b = normalized_gradient_envelope(cavd[1], cell64)
    s_ion = torch.clamp(cavd[0], 0.0, 1.0)
    s_diel = torch.clamp(cavd[1], 0.0, 1.0)
    env_sq = env_b / torch.sqrt(env_b + EPS_SQ)
    env_sq = env_sq / torch.clamp(env_sq.max(), min=1e-30)
    CH = [env_b, s_ion, s_diel, env_sq]
    CHm = [torch.clamp(w.mean(dim=(0, 1)), min=1e-12) for w in CH]

    lat, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(lat)))
    n_solv = to_shape(-(rb + ri) / Vd, shape).to(device)
    n_lat = n_solv - n_solv.mean(dim=(0, 1), keepdim=True)
    del rb, ri, n_solv
    cross_ref = float((n_lat * phi).sum() * dV)
    self_ref = 0.5 * float((n_lat * poisson_phi_periodic(n_lat, cell64)).sum() * dV)
    amp_ref = float(n_lat.abs().sum() * dV)
    print(f"\n== {tag} ==  reference lateral: cross {cross_ref:+.3f}  "
          f"self {self_ref:+.3f}  |q| {amp_ref:.3f} e", flush=True)

    g = torch.Generator(device="cpu").manual_seed(11)
    idx = torch.randint(0, int(np.prod(shape)), (NPTS,), generator=g).to(device)
    iz = idx % nz; iy = (idx // nz) % ny; ix = idx // (nz * ny)
    frac = torch.stack([(ix.to(torch.float64) + 0.5) / nx,
                        (iy.to(torch.float64) + 0.5) / ny,
                        (iz.to(torch.float64) + 0.5) / nz], dim=1)
    y = _interp3_periodic(n_lat, frac)
    phi_p = _interp3_periodic(phi, frac)
    deep = phi_p.abs() >= torch.quantile(phi_p.abs(), 0.90)
    rms_all_ref = math.sqrt(float((y ** 2).mean()))
    rms_deep_ref = math.sqrt(float((y[deep] ** 2).mean()))

    nat = len(a); K1 = nat * len(sigmas) * 9
    A = torch.zeros(NPTS, 4 * K1, dtype=torch.float32, device=device)
    gvec = torch.zeros(4 * K1, dtype=torch.float64, device=device)
    pos_frac = torch.remainder(
        torch.as_tensor(a.get_positions(), device=device, dtype=torch.float64)
        @ torch.linalg.inv(cell64), 1.0)
    beobj = model._pb1d_backend
    cvec = torch.zeros(nat, len(sigmas), 9, dtype=torch.float64, device=device)
    col = 0
    for ia in range(nat):
        for isig in range(len(sigmas)):
            for lm in range(9):
                cvec.zero_(); cvec[ia, isig, lm] = 1.0
                Gk = grid.ifft_real(beobj._gto_net_density_g(
                    grid, pos_frac, cvec, sigmas)) / V
                for ci, (w, wm) in enumerate(zip(CH, CHm)):
                    raw = w * Gk
                    r = raw.mean(dim=(0, 1)) / wm
                    prj = raw - r[None, None, :] * w
                    A[:, ci * K1 + col] = _interp3_periodic(prj, frac).to(torch.float32)
                    gvec[ci * K1 + col] = (prj * phi).sum() * dV
                col += 1
        if ia % 80 == 0:
            print(f"   [{tag}] build {ia}/{nat}", flush=True)

    sols = {}   # (basis, wt) -> (chans, c_sol, c_dir, mu)
    for bname, chans in BASES:
        sel = torch.cat([torch.arange(c * K1, (c + 1) * K1, device=device)
                         for c in chans])
        n = sel.numel()
        for wt in WTS:
            wp = (torch.ones_like(phi_p) if wt == "1"
                  else phi_p.abs() if wt == "|phi|" else phi_p ** 2)
            wp = wp / wp.mean()
            Hw = torch.zeros(n, n, dtype=torch.float64, device=device)
            Awy = torch.zeros(n, dtype=torch.float64, device=device)
            for i in range(0, NPTS, 4000):
                blk = A[i:i + 4000][:, sel].to(torch.float64)
                wb = wp[i:i + 4000]
                Hw += blk.T @ (blk * wb[:, None])
                Awy += blk.T @ (y[i:i + 4000] * wb)
                del blk
            Hw += RIDGE * float(torch.diagonal(Hw).mean()) * torch.eye(
                n, dtype=torch.float64, device=device)
            L = torch.linalg.cholesky(Hw)
            c_sol = torch.cholesky_solve(Awy.unsqueeze(1), L).squeeze(1)
            gs = gvec[sel]
            c_dir = torch.cholesky_solve(gs.unsqueeze(1), L).squeeze(1)
            gHg = float(gs @ c_dir)
            mu = (float(gs @ c_sol) - cross_ref) / gHg if gHg != 0 else 0.0
            sols[(bname, wt)] = (chans, c_sol, c_dir, mu)
            del Hw, Awy, L
            torch.cuda.empty_cache()
    # point-space diagnostics while A is alive
    pdiag = {}
    for key, (chans, c_sol, c_dir, mu) in sols.items():
        sel = torch.cat([torch.arange(c * K1, (c + 1) * K1, device=device)
                         for c in chans])
        for mode, cv in (("w", c_sol), ("w+con", c_sol - mu * c_dir)):
            pred = torch.zeros(NPTS, dtype=torch.float64, device=device)
            for i in range(0, NPTS, 4000):
                pred[i:i + 4000] = A[i:i + 4000][:, sel].to(torch.float64) @ cv
            res = pred - y
            pdiag[key + (mode,)] = (
                math.sqrt(float((res ** 2).mean())) / rms_all_ref,
                math.sqrt(float((res[deep] ** 2).mean())) / rms_deep_ref)
            del pred, res
    del A
    torch.cuda.empty_cache()

    # rebuild every solution's field in one synthesis pass
    flds = {}
    for key, (chans, c_sol, c_dir, mu) in sols.items():
        flds[key + ("sol",)] = torch.zeros(shape, dtype=torch.float64, device=device)
        flds[key + ("dir",)] = torch.zeros(shape, dtype=torch.float64, device=device)
    col = 0
    for ia in range(nat):
        for isig in range(len(sigmas)):
            for lm in range(9):
                todo = []
                for key, (chans, c_sol, c_dir, mu) in sols.items():
                    for j, ch in enumerate(chans):
                        vs = float(c_sol[j * K1 + col]); vd = float(c_dir[j * K1 + col])
                        if abs(vs) > 1e-14 or abs(vd) > 1e-14:
                            todo.append((key, ch, vs, vd))
                if todo:
                    cvec.zero_(); cvec[ia, isig, lm] = 1.0
                    Gk = grid.ifft_real(beobj._gto_net_density_g(
                        grid, pos_frac, cvec, sigmas)) / V
                    projs = {}
                    for key, ch, vs, vd in todo:
                        if ch not in projs:
                            w = CH[ch]; wm = CHm[ch]
                            raw = w * Gk
                            r = raw.mean(dim=(0, 1)) / wm
                            projs[ch] = raw - r[None, None, :] * w
                        if abs(vs) > 1e-14:
                            flds[key + ("sol",)] += vs * projs[ch]
                        if abs(vd) > 1e-14:
                            flds[key + ("dir",)] += vd * projs[ch]
                col += 1
        if ia % 80 == 0:
            print(f"   [{tag}] rebuild {ia}/{nat}", flush=True)

    print(f"  {'basis':>11} {'weight':>7} {'mode':>6} {'rms/ref':>8} "
          f"{'rms deep':>9} {'cross/ref':>10} {'self/ref':>9} {'|q|/ref':>8}")
    print(f"  {'reference':>11} {'—':>7} {'—':>6} {'—':>8} {'—':>9} "
          f"{1.0:10.3f} {1.0:9.3f} {1.0:8.3f}")
    for bname, _ in BASES:
        for wt in WTS:
            chans, c_sol, c_dir, mu = sols[(bname, wt)]
            fs = flds[(bname, wt, "sol")]; fd = flds[(bname, wt, "dir")]
            for mode, f in (("w", fs), ("w+con", fs - mu * fd)):
                cr = float((f * phi).sum() * dV)
                se = 0.5 * float((f * poisson_phi_periodic(f, cell64)).sum() * dV)
                am = float(f.abs().sum() * dV)
                r1, r2 = pdiag[(bname, wt, mode)]
                rows.append((tag, bname, wt, mode, r1, r2,
                             cr / cross_ref, se / self_ref, am / amp_ref))
                print(f"  {bname:>11} {wt:>7} {mode:>6} {r1:8.3f} {r2:9.3f} "
                      f"{cr/cross_ref:10.3f} {se/self_ref:9.3f} "
                      f"{am/amp_ref:8.3f}", flush=True)
    del flds, n_lat, phi, env_b, s_ion, s_diel, env_sq, CH
    import gc; gc.collect(); torch.cuda.empty_cache()

print("\n=== best solution per frame by max deviation from the reference ===")
for tg in dict.fromkeys(r[0] for r in rows):
    cand = [r for r in rows if r[0] == tg]
    cand.sort(key=lambda r: max(abs(r[6] - 1), abs(r[7] - 1), abs(r[8] - 1)))
    for r in cand[:3]:
        print(f"{r[0]:>14} {r[1]:>11} w={r[2]:>6} {r[3]:>6} | rms {r[4]:.3f} "
              f"deep {r[5]:.3f} | cross {r[6]:.3f} self {r[7]:.3f} |q| {r[8]:.3f}")
print("DONE  (an acceptable solution = cross/self/|q| all near 1 without a "
      "large point-rms penalty)")
