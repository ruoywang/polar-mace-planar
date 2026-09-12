"""Where inside the CAVITY term does the second-order (force-loss-gradient)
truncation sit? Staged AD-vs-FD along the cavity chain (2026-09-12).

floss_grad_by_term.py put 100% of the remaining v.dL_F gap for the density-map
parameters in the cavity term (sid 28: AD -5.883e-05 vs FD -5.434e-05, 8.3%;
sid 628: -3.660e-04 vs -3.624e-04, 1.0%), while the same term's FIRST-order
v.dE closes to 2e-6 and its force closes to 0.01 meV/A. So some operation in
the chain
    coefficients, positions -> GTO net density -> ne = clamp(neutral - net)
    -> s_vdw = shape(log clamp(ne/NC_K)) -> convolutions -> s_cav (* Stern box)
    -> spectral gradient -> sqrt(|grad|^2 + 1e-30) -> area -> TAU * area * dV
has a correct value and a correct first derivative but a wrong (or FD-
unresolvable) mixed second derivative. This script takes the live
intermediates exported by the backend (MACE_PB1D_CAV_EXPORT=1) and, for a
scalar S at each stage, compares
    AD_S = v . grad_theta [ g . F_S ],   F_S = -dS/dR (create_graph)
    FD_S = [ g . F_S(theta+eps v) - g . F_S(theta-eps v) ] / 2 eps
with the same frozen g = dL_F/dF and the same v as before (seed 7, first
group). Stage scalars: the cavity energy term itself (must reproduce the
floss number), the backend's area, the area recomputed with a larger floor
(MACE_PB1D_AREA_EPS is only read by the backend; here the floor is varied on
the exported gradients), sum w.|grad|^2 (no sqrt), sum w.s_cav, sum w.s_vdw,
sum w.ne with a fixed random +-1 grid field w. The first stage whose AD
departs from FD beyond the FD drift names the operation. Also printed: the
plateau population (|grad|^2 against the floor) and the clamp-active
fractions at theta, which decide between "floor" and "kink" readings.
"""
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
os.environ["MACE_PB1D_CAV_EXPORT"] = "1"
assert os.environ.get("MACE_PB1D_LIVE_POS") and not os.environ.get("MACE_PB1D_DFORCE")
assert os.environ.get("MACE_PB1D_GRAD_PASSES") == "0"
from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB
from pure_python import torch_pb as TP
RUN = "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le"; os.chdir(RUN)
GROUP = os.environ.get("KIT_GROUP", "field_dependent_charges_maps")
EPS_REL = [float(x) for x in os.environ.get("KIT_EPS", "3e-5,1e-5").split(",")]; SEED = 7
SIDS = [int(x) for x in os.environ.get("KIT_SIDS", "28,628").split(",")]
device = torch_tools.init_device("cuda"); torch_tools.set_default_dtype("float64")
args = tools.build_default_arg_parser().parse_args(["--config", "config_pb1d.yaml", "--name", "s3d_gate_le", "--seed", "123", "--work_dir", ".", "--device", "cuda"])
args.key_specification = KeySpecification(); update_keyspec_from_kwargs(args.key_specification, vars(args))
mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model"))) if "compiled" not in p][-1]
cp = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")), key=lambda p: int(p.rsplit("epoch-",1)[1].split(".")[0]))[-1]
model = torch.load(f=mp, map_location=device).to(device)
model.load_state_dict(torch.load(cp, map_location=device)["model"], strict=False)
model.eval(); model.solvent3d_energy = model.solvent_cavity_energy = model.solvent_baseline_coupling = True
model._pb1d_epoch = 39
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
_bd = {}; _bk = PB.PB1DBackend.solve_graph
def _wrap(self, *a, **k): _bd["b"] = self; return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap
def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict): d.clear()
        b.last_cav_export = None
    torch.cuda.empty_cache()
atoms_by_sid = {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if os.path.exists(f):
        for a in read(f, ":"):
            s = a.info.get("sample_id")
            if s is not None: atoms_by_sid[int(s)] = a
PARAMS = [(n, p) for n, p in model.named_parameters() if n.split(".")[0] == GROUP]
plist = [p for _, p in PARAMS]
gen = torch.Generator(device="cpu").manual_seed(SEED)
vs = [torch.randn(p.shape, generator=gen, dtype=torch.float64).to(device) for _, p in PARAMS]
nrm = torch.sqrt(sum((v**2).sum() for v in vs)); DIRS = [v / nrm for v in vs]
TN = float(torch.sqrt(sum((p.detach()**2).sum() for _, p in PARAMS)))
W = {}   # fixed random +-1 grid field per grid shape

def set_requires(flag):
    for p in model.parameters(): p.requires_grad_(False)
    if flag:
        for p in plist: p.requires_grad_(True)

def batch(sid):
    cfg = mace_data.config_from_atoms(atoms_by_sid[sid], key_specification=args.key_specification)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    b["positions"].requires_grad_(True); return b

def wfield(shape):
    key = tuple(shape)
    if key not in W:
        g = torch.Generator(device="cpu").manual_seed(11)
        W[key] = (torch.randint(0, 2, shape, generator=g).to(torch.float64) * 2 - 1).to(device)
    return W[key]

def stage_scalars(pred, ex):
    """All stage scalars from one forward's prediction + live export."""
    p = ex["p"]; ne = ex["ne_cav"]; w = wfield(ne.shape)
    x_vdw = torch.log(torch.clamp(ne / float(p["NC_K"]), min=float(p["N_MIN"])))
    s_vdw = TP._shape_func(x_vdw, float(p["SIGMA_K"]))
    g2 = ex["gx"] ** 2 + ex["gy"] ** 2 + ex["gz"] ** 2
    S = {"E_cav term": pred["cavity_energy_g"].sum(),
         "area (backend, floor 1e-30)": ex["area"],
         "area floor 1e-20": torch.sqrt(g2 + 1e-20).sum(),
         "area floor 1e-12": torch.sqrt(g2 + 1e-12).sum(),
         "sum w.|grad s_cav|^2": (w * g2).sum(),
         "sum w.s_cav": (w * ex["s_cav3e"]).sum(),
         "sum w.s_vdw": (w * s_vdw).sum(),
         "sum w.ne": (w * ne).sum()}
    return S, g2

def plateau_stats(ex, g2):
    ne = ex["ne_cav"]; p = ex["p"]
    raw = (ex["neutral_e"] - ex["grid"].ifft_real(ex["net_g2"])) / ex["volume"]
    n = g2.numel()
    out = {"grid points": n,
           "|grad|^2 < 1e-28": float((g2 < 1e-28).sum()) / n,
           "|grad|^2 < 1e-20": float((g2 < 1e-20).sum()) / n,
           "|grad|^2 < 1e-12": float((g2 < 1e-12).sum()) / n,
           "min |grad|^2": float(g2.min()), "max |grad|^2": float(g2.max()),
           "ne clamp(min=0) active": float((raw < 0).sum()) / n,
           "log floor active (ne/NC_K < N_MIN)": float((ne / float(p["NC_K"]) < float(p["N_MIN"])).sum()) / n}
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}

print(f"model {os.path.basename(mp)} @ {os.path.basename(cp)}  LIVE_POS=1  GRAD_PASSES=0  group {GROUP} ({sum(p.numel() for p in plist)} params)  eps_rel {EPS_REL}")
for sid in SIDS:
    q = float(atoms_by_sid[sid].info.get("total_charge", 0.0))
    f_dft = torch.as_tensor(np.asarray(atoms_by_sid[sid].get_array("forces")), dtype=torch.float64, device=device)
    print("\n" + "=" * 78); print(f"sid {sid} ({'charged' if q else 'neutral'}), training-cache path")
    # ---- reference: frozen g, plateau statistics, stage names
    set_requires(True); b = batch(sid)
    pred = model(b.to_dict(), compute_force=False, training=False); ex = _bd["b"].last_cav_export
    E = pred["energy"].sum(); F = -torch.autograd.grad(E, b["positions"], create_graph=True, retain_graph=True)[0]
    gvec = (2.0 * (F - f_dft) / f_dft.numel()).detach()
    S, g2 = stage_scalars(pred, ex)
    ecav = float(pred["cavity_energy_g"].sum()); area0 = float(ex["area"])
    print(f"  E {float(E):.9f}   E_cav {ecav:.6f} eV   area {area0:.4f}   E_cav/area (TAU*dV) {ecav/area0:.6e}")
    st = plateau_stats(ex, g2.detach())
    for k, v in st.items(): print(f"    {k:>36}: {v:.4e}" if isinstance(v, float) else f"    {k:>36}: {v}")
    names = list(S.keys()); del pred, ex, E, F, S, g2, b; _evict()
    # ---- AD per stage, one forward each
    AD = {}
    for nm in names:
        set_requires(True); b = batch(sid)
        pred = model(b.to_dict(), compute_force=False, training=False); ex = _bd["b"].last_cav_export
        S, _ = stage_scalars(pred, ex)
        Fs = torch.autograd.grad(S[nm], b["positions"], create_graph=True, allow_unused=True)[0]
        if Fs is None or not Fs.requires_grad:
            AD[nm] = 0.0
        else:
            Q = (gvec * (-Fs)).sum()
            gq = torch.autograd.grad(Q, plist, allow_unused=True)
            AD[nm] = sum(float((gi * v).sum()) for gi, v in zip(gq, DIRS) if gi is not None)
        del pred, ex, S, Fs, b; _evict()
    # ---- FD per stage, shared perturbed forwards
    set_requires(False); saved = [p.detach().clone() for p in plist]
    FD = {}
    for er in EPS_REL:
        eps = er * TN; Q = {}
        for sign in (+1.0, -1.0):
            with torch.no_grad():
                for p, v, s0 in zip(plist, DIRS, saved): p.copy_(s0 + sign * eps * v)
            b = batch(sid)
            with torch.enable_grad():
                pred = model(b.to_dict(), compute_force=False, training=False); ex = _bd["b"].last_cav_export
                S, _ = stage_scalars(pred, ex)
                Q[sign] = {}
                for nm in names:
                    Fs = torch.autograd.grad(S[nm], b["positions"], retain_graph=True, allow_unused=True)[0]
                    Q[sign][nm] = 0.0 if Fs is None else float((gvec * (-Fs.detach())).sum())
            del pred, ex, S, b; _evict()
        with torch.no_grad():
            for p, s0 in zip(plist, saved): p.copy_(s0)
        FD[er] = {nm: (Q[+1.0][nm] - Q[-1.0][nm]) / (2 * eps) for nm in names}
    er0, erL = EPS_REL[0], EPS_REL[-1]
    print(f"\n  {'stage':>30} {'AD':>14} {'FD eps'+format(er0,'g'):>16} {'FD eps'+format(erL,'g'):>16} {'AD-FD rel':>10} {'FD drift':>9}")
    for nm in names:
        fd0, fdL = FD[er0][nm], FD[erL][nm]
        rel = abs(AD[nm] - fdL) / max(abs(fdL), 1e-300); dr = abs(fd0 - fdL) / max(abs(fdL), 1e-300)
        print(f"  {nm:>30} {AD[nm]:+14.6e} {fd0:+16.6e} {fdL:+16.6e} {rel:10.2e} {dr:9.2e}")
print("\nREADING: the first stage (reading bottom-up: ne -> s_vdw -> s_cav -> |grad|^2 -> area) whose AD-FD")
print("relative difference stands above its FD drift is where the mixed second derivative goes wrong.")
print("If the two area floors differ from each other while |grad|^2 closes, it is the sqrt floor.")
print("DONE")
