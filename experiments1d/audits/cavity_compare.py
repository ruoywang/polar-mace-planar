"""Where is the solvent allowed to be, and can the model represent it there?
(user directive 2026-09-09, revised after a conceptual correction)

CORRECTION being applied. The "lateral" charge is rho_solv - <rho_solv>_xy, so
even where the TRUE solvent charge is exactly zero the lateral residual is
-<rho_solv>(z), non-zero, purely to cancel the plane-averaged background. The
earlier reading "28.5% of the lateral charge sits where the cavity is closed,
therefore the cavity is wrong" was therefore invalid: that share can be pure
background cancellation. Raw and lateral charges are now counted SEPARATELY.

This also raises a distinct and more likely defect: the model broadcasts its
1-D background over the whole plane, including inside the solute, while the
envelope forbids the 3-D residual from entering that region — so the residual
CANNOT cancel the background there, however correct the cavity is. The
measurements below are arranged to tell the two apart:
  cavity wrong        -> significant RAW solvent charge where the model's
                         cavity is closed but the DFT cavity is open;
  representation wrong -> raw charge ~ 0 in the closed region while the model's
                         broadcast background is not, with the residual unable
                         to cancel it, and a matching spurious cross energy.
Note on VASPsol++: the bound charge is -div P with a non-local convolution, so
"s_diel small implies bound charge zero" is not a valid requirement — hence the
raw distributions are reported, not asserted.
"""
_OLD = """Where is the solvent actually allowed to be? (user directive 2026-09-09)

This takes priority over designing any near-shell charge channel, because it
can change what the fix is: if the true cavity is OPEN in the near-solute
region while the model's cavity is CLOSED there, then the thing to repair is
the cavity (hence the predicted density it is built from), not the charge basis.

Three questions, in order of how artifact-free they are:

Q1 (no interpolation at all, DFT native grid): where does the true lateral
   solvent charge live relative to the TRUE cavity, and where does its
   attraction energy come from? Bin |n_lat| and its cross-energy contribution
   by s_diel^DFT. Physically the bound charge is -div P with P proportional to
   s_diel, so charge at s_diel ~ 0 would be a red flag on the diagnostic, not
   on the model.
   The cavity recipe applied to the DFT density is the verified one: it
   reproduced VASPsol's own printed cavity area to 0.01 A^2 (433.82 = 433.82).

Q2 (model grid): do the two cavities agree? Classify every point as both-open,
   both-closed, model-closed-DFT-open, model-open-DFT-closed, and report the
   volume, the true lateral charge and the cross energy in each class. The
   third class is the one that would explain "the envelope cannot reach where
   the missing energy is".

Q3: if they disagree, is it the density or the recipe? Compare the model's own
   electron density (the field its cavity is built from) with the DFT density
   in each class.

Frames: one charged, one neutral.  cwd = gate_bl.
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
from pure_python import torch_pb as tp
from mace.modules.solvent3d import poisson_phi_periodic

GCE44 = "/scratch/08384/tg876840/tmp/2-NiN_single/1-44_GCE"
NEU = "/scratch/08384/tg876840/tmp/2-NiN_single/5-44_neutral_withsolv"
FRAMES = [(1, f"{GCE44}/cal_1", "NiN44 q=-1.00"),
          (601, f"{NEU}/cal_1", "neutral")]
SBINS = [0.0, 0.01, 0.1, 0.5, 0.9, 1.01]

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
_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    cap["rho_layer_z"] = out["rho_layer_z"].detach().clone()
    if out.get("s3d_obs") is not None:
        o = out["s3d_obs"]
        # delta_grid is the EXACT residual the energy integrals use, on the
        # energy grid (MACE_S3D_EXPORT_DELTA). d_sup_* would be wrong here:
        # they live on the upsampled supervision grid.
        cap["delta"] = o["delta_grid"].detach().clone()
    return out
PB.PB1DBackend.solve_graph = wrap_solve
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["n_e"] = n_e.detach().clone()      # the density the cavity is built from
    cap["cvhar3"] = cv.detach().clone()
    cap["grid"] = grid
    cap["params"] = params
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
want = {sid: None for sid, _, _ in FRAMES}
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
        t[None, None].to(torch.float64), size=tuple(shape), mode="trilinear",
        align_corners=False)[0, 0].contiguous()


for sid, dftdir, tag in FRAMES:
    a = want[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        model(b.to_dict(), compute_force=False, training=False)
    grid = cap["grid"]
    params = cap["params"]
    cellm = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    shm = tuple(grid.shape)
    Vm = grid.volume; dVm = Vm / float(np.prod(shm))
    s_diel_m = torch.clamp(grid._solv3d_cavity[1], 0.0, 1.0)
    ne_m = cap["n_e"].to(device)
    phi_m = -(cap["cvhar3"] - cap["cvhar3"].mean())
    print(f"\n=============== {tag} ===============", flush=True)

    # ---------- DFT native grid ----------
    latd, chg = read_grid(f"{dftdir}/CHGCAR")
    _, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    _, phif = read_grid(f"{dftdir}/PHI")
    Vd = float(abs(np.linalg.det(latd)))
    shd = tuple(chg.shape)
    dVd = Vd / float(np.prod(shd))
    celld = torch.tensor(latd, device=device)
    ne_d = torch.clamp(chg.to(device) / Vd, min=0.0)
    grid_d = tp.TorchGrid(latd, shd, device=str(device), dtype=torch.float64,
                          rspec=True)
    s_ion_d, s_diel_d, _ = tp.create_cavity_torch(ne_d, grid_d, params)
    s_diel_d = torch.clamp(s_diel_d, 0.0, 1.0)
    n_solv_d = -((rb + ri).to(device)) / Vd
    n_solv_d_keep = n_solv_d
    n_lat_d = n_solv_d - n_solv_d.mean(dim=(0, 1), keepdim=True)
    # DFT's own solute potential on its native grid (physical sign, zero mean)
    phi_d_tot = -phif.to(device)
    B = 2 * math.pi * torch.linalg.inv(celld).T
    fx = torch.fft.fftfreq(shd[0], device=device) * shd[0]
    fy = torch.fft.fftfreq(shd[1], device=device) * shd[1]
    fz = torch.arange(shd[2] // 2 + 1, dtype=torch.float64, device=device)
    G2 = ((fx[:, None, None, None] * B[0] + fy[None, :, None, None] * B[1]
           + fz[None, None, :, None] * B[2]) ** 2).sum(-1)
    from mace.modules.solvent3d import FIELD_CONSTANT_EVA
    n_tot_d = torch.fft.irfftn(torch.fft.rfftn(phi_d_tot) * G2
                               / (4 * math.pi * FIELD_CONSTANT_EVA), s=shd)
    phi_sol_d = poisson_phi_periodic(n_tot_d - n_solv_d, celld)
    del chg, phif, n_tot_d

    # RAW (not lateral) DFT solvent charges, kept separate on purpose
    rb_d = -(rb.to(device)) / Vd
    ri_d = -(ri.to(device)) / Vd
    s_ion_dc = torch.clamp(s_ion_d, 0.0, 1.0)
    for nm, fld, shp in (("RHOB vs s_diel", rb_d, s_diel_d),
                         ("RHOION vs s_ion", ri_d, s_ion_dc),
                         ("full solvent vs s_diel", n_solv_d, s_diel_d)):
        tot = float(fld.abs().sum() * dVd)
        net = float(fld.sum() * dVd)
        print(f"\n[Q1-raw] {nm}: int|rho| {tot:.3f} e, net {net:+.3f} e",
              flush=True)
        print(f"  {'shape bin':>16} {'volume%':>8} {'|rho|%':>8} {'net (e)':>9} "
              f"{'cross':>9} {'<|rho|>':>10}")
        for k in range(len(SBINS) - 1):
            m = (shp >= SBINS[k]) & (shp < SBINS[k + 1])
            if not bool(m.any()):
                continue
            print(f"  {SBINS[k]:6.2f}-{SBINS[k+1]:<8.2f} "
                  f"{100.0*float(m.sum())/float(np.prod(shd)):8.2f} "
                  f"{100.0*float((fld.abs()*m).sum()*dVd)/max(tot,1e-30):8.2f} "
                  f"{float((fld*m).sum()*dVd):+9.3f} "
                  f"{float((fld*phi_sol_d*m).sum()*dVd):+9.3f} "
                  f"{float(fld[m].abs().mean()):10.2e}", flush=True)
    amp_d = float(n_lat_d.abs().sum() * dVd)
    cross_d = float((n_lat_d * phi_sol_d).sum() * dVd)
    print(f"\n[Q1-lateral] DFT native grid {shd}: |n_lat| {amp_d:.3f} e, "
          f"cross with the DFT solute potential {cross_d:+.3f} eV", flush=True)
    print(f"  {'s_diel^DFT bin':>16} {'volume%':>8} {'|n_lat|%':>9} "
          f"{'cross':>9} {'cross%':>8} {'<|phi|>':>8}")
    for k in range(len(SBINS) - 1):
        m = (s_diel_d >= SBINS[k]) & (s_diel_d < SBINS[k + 1])
        if not bool(m.any()):
            continue
        vol = 100.0 * float(m.sum()) / float(np.prod(shd))
        aq = 100.0 * float((n_lat_d.abs() * m).sum() * dVd) / amp_d
        cx = float((n_lat_d * phi_sol_d * m).sum() * dVd)
        print(f"  {SBINS[k]:6.2f}-{SBINS[k+1]:<8.2f} {vol:8.2f} {aq:9.2f} "
              f"{cx:+9.3f} {100*cx/cross_d:8.2f} "
              f"{float(phi_sol_d[m].abs().mean()):8.3f}", flush=True)

    # ---------- model grid: do the two cavities agree? ----------
    s_diel_d_m = torch.clamp(to_shape(s_diel_d, shm), 0.0, 1.0)
    ne_d_m = to_shape(ne_d, shm)
    n_lat_dm = to_shape(n_lat_d, shm)
    amp_dm = float(n_lat_dm.abs().sum() * dVm)
    cross_dm = float((n_lat_dm * phi_m).sum() * dVm)
    openm = s_diel_m > 0.5
    opend = s_diel_d_m > 0.5
    classes = [("both open", openm & opend), ("both closed", ~openm & ~opend),
               ("model CLOSED, DFT open", ~openm & opend),
               ("model open, DFT CLOSED", openm & ~opend)]
    print(f"\n[Q2] model grid {shm}: |n_lat| {amp_dm:.3f} e, cross with the "
          f"MODEL solute potential {cross_dm:+.3f} eV", flush=True)
    print(f"  {'class':>24} {'volume%':>8} {'|n_lat|%':>9} {'cross':>9} "
          f"{'cross%':>8} {'<s_diel^m>':>11} {'<s_diel^D>':>11}")
    for name, m in classes:
        if not bool(m.any()):
            print(f"  {name:>24} {'empty':>8}"); continue
        vol = 100.0 * float(m.sum()) / float(np.prod(shm))
        aq = 100.0 * float((n_lat_dm.abs() * m).sum() * dVm) / amp_dm
        cx = float((n_lat_dm * phi_m * m).sum() * dVm)
        print(f"  {name:>24} {vol:8.2f} {aq:9.2f} {cx:+9.3f} "
              f"{100*cx/cross_dm:8.2f} {float(s_diel_m[m].mean()):11.4f} "
              f"{float(s_diel_d_m[m].mean()):11.4f}", flush=True)

    # ---------- Q3: density or recipe? ----------
    # the model's own full 3-D solvent charge: 1-D background + residual
    rl = cap["rho_layer_z"].to(device).to(torch.float64)
    nzm = shm[2]
    if rl.shape[0] != nzm:
        rl = torch.nn.functional.interpolate(
            rl[None, None], size=(nzm,), mode="linear",
            align_corners=False)[0, 0]
    bg = rl[None, None, :].expand(shm).contiguous()
    dl = cap["delta"].to(device)
    assert tuple(dl.shape) == shm, (dl.shape, shm)
    # sanity: the per-plane projection must make every plane of the residual
    # sum to zero, so this is the field the energy scored (expect ~1e-15)
    print(f"\n  [check] residual plane-sum max |mean_xy(delta)| "
          f"{float(dl.mean(dim=(0,1)).abs().max()):.2e} e/A^3; 1-D background "
          f"net {float(bg.sum()*dVm):+.3f} e (solver net "
          f"{float(cap['rho_layer_z'].sum())*Vm/float(cap['rho_layer_z'].shape[0]):+.3f} e)",
          flush=True)
    n_solv_dm = to_shape(n_solv_d_keep, shm)
    print(f"\n[Q2-raw] per class: TRUE raw solvent charge vs the model's "
          f"broadcast background and its residual (e, and eV of cross energy)",
          flush=True)
    print(f"  {'class':>24} {'vol%':>6} {'true net':>9} {'true |q|':>9} "
          f"{'bg net':>8} {'bg |q|':>8} {'delta |q|':>10} {'true cross':>11} "
          f"{'model cross':>12}")
    for name, m in classes:
        if not bool(m.any()):
            continue
        print(f"  {name:>24} {100.0*float(m.sum())/float(np.prod(shm)):6.2f} "
              f"{float((n_solv_dm*m).sum()*dVm):+9.3f} "
              f"{float((n_solv_dm.abs()*m).sum()*dVm):9.3f} "
              f"{float((bg*m).sum()*dVm):+8.3f} "
              f"{float((bg.abs()*m).sum()*dVm):8.3f} "
              f"{float((dl.abs()*m).sum()*dVm):10.3f} "
              f"{float((n_solv_dm*phi_m*m).sum()*dVm):+11.3f} "
              f"{float(((bg+dl)*phi_m*m).sum()*dVm):+12.3f}", flush=True)
    print(f"  (a class where the true net charge is ~0 while the background is "
          f"not, and |delta| is small, is the uncancelled-background defect;\n"
          f"   a class where the true charge is significant and the model's "
          f"cavity is closed is a cavity error)", flush=True)

    print(f"\n[Q3] electron density in each class (e/A^3), model vs DFT",
          flush=True)
    print(f"  {'class':>24} {'<ne model>':>11} {'<ne DFT>':>10} "
          f"{'<ne m - ne D>':>13}")
    for name, m in classes:
        if not bool(m.any()):
            continue
        print(f"  {name:>24} {float(ne_m[m].mean()):11.4f} "
              f"{float(ne_d_m[m].mean()):10.4f} "
              f"{float((ne_m - ne_d_m)[m].mean()):+13.4f}", flush=True)
    thr = float(params.get("NC_K", 0.0025)) if hasattr(params, "get") else 0.0025
    print(f"  (cavity switches around the density threshold NC_K = {thr:g} "
          f"e/A^3; s_diel = 0 means solvent excluded. CAVEAT: the DFT density "
          f"here is put through the MODEL's cavity parameters, so this "
          f"isolates the effect of the density INPUT only — it cannot by "
          f"itself rule out a recipe or parameter mismatch.)", flush=True)
    tot = float(np.prod(shm))
    print(f"  cavity mismatch volume: "
          f"{100*float((openm ^ opend).sum())/tot:.2f}% of the cell", flush=True)
    del s_diel_d, s_ion_d, ne_d, grid_d, n_solv_d, n_lat_d, phi_sol_d
    del rb, ri, rb_d, ri_d, s_ion_dc, n_solv_d_keep, bg, dl, n_solv_dm, rl
    del s_diel_d_m, ne_d_m, n_lat_dm, s_diel_m, ne_m, phi_m
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
