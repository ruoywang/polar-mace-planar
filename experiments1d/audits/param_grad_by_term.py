"""Which ENERGY TERM carries the 25-29% error in the energy gradient w.r.t. the
density-coefficient parameters? (user-specified, 2026-09-12.)

The second-order check measured v.grad_theta E for theta = field_dependent_
charges_maps 25-29% below its finite difference, unchanged by the stage-1
reconnection (as expected: those maps act after the recursion). The user's
instruction is to split that error by energy term instead of scanning centres:
same parameter direction v, same two perturbed forwards, and for every term
E_k that enters total_energy

    AD_k = v . grad_theta E_k          FD_k = [E_k(theta+eps v) - E_k(theta-eps v)] / 2 eps

with the closures sum_k AD_k = v.grad E and sum_k FD_k = FD(E) checked, and
then D_k = AD_k - FD_k per term, sum_k D_k = D_total. The term with the
disagreement names the dependency; only then is anything modified.

Terms as in term_derivative_split.py (read from the assembly): e0,
interaction_energy, electron_energy, electrostatic_energy, the two
compensation terms, external_field . explicit_dipole, cavity, solvent3d,
baseline coupling. Training-cache path, both frames of the pair, LIVE_POS=1,
GRAD_PASSES=0.
"""
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
assert os.environ.get("MACE_PB1D_LIVE_POS") and not os.environ.get("MACE_PB1D_DFORCE")
from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB
RUN = "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le"; os.chdir(RUN)
GROUPS = os.environ.get("KIT_GROUPS", "field_dependent_charges_maps,products").split(",")
EPS_REL = [1e-4, 3e-5]; SEED = 7
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
TERMS = ["e0", "interaction_energy", "electron_energy", "electrostatic_energy",
         "compensation_periodic_1d_energy", "compensation_slab_correction_energy",
         "extfield_dipole", "cavity_energy_g", "solvent3d_energy_g", "baseline_coupling_energy_g"]
SHORT = {"e0":"e0","interaction_energy":"E head","electron_energy":"local_e","electrostatic_energy":"solute ES",
         "compensation_periodic_1d_energy":"comp 1D","compensation_slab_correction_energy":"slab dip",
         "extfield_dipole":"extfield.dip","cavity_energy_g":"cavity","solvent3d_energy_g":"solv3D",
         "baseline_coupling_energy_g":"E_bl"}
atoms_by_sid, split_of = {}, {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if os.path.exists(f):
        for a in read(f, ":"):
            s = a.info.get("sample_id")
            if s is not None: atoms_by_sid[int(s)] = a; split_of[int(s)] = sp
pair = [(k, k+600) for k in range(1, 401) if k in atoms_by_sid and k+600 in atoms_by_sid and split_of.get(k) == "val" and len(atoms_by_sid[k]) == 207][0]
PARAMS = {g: [(n, p) for n, p in model.named_parameters() if n.split(".")[0] == g] for g in GROUPS}
gen = torch.Generator(device="cpu").manual_seed(SEED)
DIRS = {}
for g in GROUPS:
    vs = [torch.randn(p.shape, generator=gen, dtype=torch.float64).to(device) for _, p in PARAMS[g]]
    nrm = torch.sqrt(sum((v**2).sum() for v in vs)); DIRS[g] = [v / nrm for v in vs]
TN = {g: float(torch.sqrt(sum((p.detach()**2).sum() for _, p in PARAMS[g]))) for g in GROUPS}

def batch(sid):
    cfg = mace_data.config_from_atoms(atoms_by_sid[sid], key_specification=args.key_specification)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    return next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)

def term_tensors(pred, b):
    out = {}
    nat = int(b["ptr"][1] - b["ptr"][0])
    heads = b["head"][b["batch"]] if "head" in b.to_dict() else torch.zeros_like(b["batch"])
    e0v = model.atomic_energies_fn(b["node_attrs"])
    out["e0"] = (e0v[torch.arange(nat, device=device), heads] if e0v.dim() > 1 else e0v).sum()
    for k in TERMS:
        if k in ("e0", "extfield_dipole"): continue
        v = pred.get(k); out[k] = v.sum() if v is not None else torch.zeros((), device=device)
    ef, ed = pred.get("external_field"), pred.get("explicit_dipole")
    out["extfield_dipole"] = (ef * ed).sum() if ef is not None and ed is not None else torch.zeros((), device=device)
    return out

def forward_terms(sid, grad):
    b = batch(sid)
    if grad:
        pred = model(b.to_dict(), compute_force=False, training=False)
        tt = term_tensors(pred, b); E = pred["energy"].sum()
        return E, tt
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
        tt = term_tensors(pred, b)
        r = ({k: float(v) for k, v in tt.items()}, float(pred["energy"]))
    _evict(); return r

print(f"model {os.path.basename(mp)} @ {os.path.basename(cp)}  LIVE_POS=1  GRAD_PASSES={os.environ.get('MACE_PB1D_GRAD_PASSES','1')}  training-cache path")
for sid in pair:
    q = float(atoms_by_sid[sid].info.get("total_charge", 0.0))
    for g in GROUPS:
        print("\n" + "=" * 78); print(f"sid {sid} ({'charged' if q else 'neutral'}), group {g} ({sum(p.numel() for _, p in PARAMS[g])} params)")
        for p in model.parameters(): p.requires_grad_(False)
        for _, p in PARAMS[g]: p.requires_grad_(True)
        E, tt = forward_terms(sid, True)
        plist = [p for _, p in PARAMS[g]]
        AD = {}
        for k in TERMS:
            gs = torch.autograd.grad(tt[k], plist, retain_graph=True, allow_unused=True)
            AD[k] = sum(float((gi * v).sum()) for gi, v in zip(gs, DIRS[g]) if gi is not None)
        gE = torch.autograd.grad(E, plist, allow_unused=True)
        AD_tot = sum(float((gi * v).sum()) for gi, v in zip(gE, DIRS[g]) if gi is not None)
        for p in model.parameters(): p.requires_grad_(False)
        _evict()
        saved = [p.detach().clone() for _, p in PARAMS[g]]
        FD = {}
        for er in EPS_REL:
            eps = er * TN[g]; vals = []
            for sign in (+1.0, -1.0):
                with torch.no_grad():
                    for (_, p), v, s0 in zip(PARAMS[g], DIRS[g], saved): p.copy_(s0 + sign * eps * v)
                vals.append(forward_terms(sid, False))
            with torch.no_grad():
                for (_, p), s0 in zip(PARAMS[g], saved): p.copy_(s0)
            FD[er] = ({k: (vals[0][0][k] - vals[1][0][k]) / (2*eps) for k in TERMS},
                      (vals[0][1] - vals[1][1]) / (2*eps))
        er = EPS_REL[0]; fdk, fdE = FD[er]; fdk2, fdE2 = FD[EPS_REL[1]]
        print(f"  CLOSURE AD: sum_k AD_k {sum(AD.values()):+.6e} vs v.grad E {AD_tot:+.6e}  |diff| {abs(sum(AD.values())-AD_tot):.2e}")
        print(f"  CLOSURE FD: sum_k FD_k {sum(fdk.values()):+.6e} vs FD(E)   {fdE:+.6e}  |diff| {abs(sum(fdk.values())-fdE):.2e}   (FD drift {abs(fdE-fdE2)/max(abs(fdE),1e-30):.2e})")
        print(f"  D_total = AD - FD = {AD_tot - fdE:+.6e}   ({100*(AD_tot-fdE)/max(abs(fdE),1e-30):+.1f}% of FD)")
        print(f"  {'term':>14} {'AD_k':>13} {'FD_k':>13} {'D_k':>13} {'share of D_total':>17} {'FD_k drift':>11}")
        Dsum = 0.0
        for k in sorted(TERMS, key=lambda k: -abs(AD[k]-fdk[k])):
            Dk = AD[k] - fdk[k]; Dsum += Dk
            dr = abs(fdk[k]-fdk2[k])/max(abs(fdk[k]),1e-30)
            print(f"  {SHORT[k]:>14} {AD[k]:+13.5e} {fdk[k]:+13.5e} {Dk:+13.5e} {100*Dk/max(abs(AD_tot-fdE),1e-30):+16.1f}% {dr:11.2e}")
        print(f"  {'sum D_k':>14} {'':>13} {'':>13} {Dsum:+13.5e}   vs D_total {AD_tot-fdE:+.5e}  |diff| {abs(Dsum-(AD_tot-fdE)):.2e}")
print("\nDONE")
