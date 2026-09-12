"""Which ENERGY TERM carries the remaining error in the FORCE-LOSS gradient
w.r.t. the density-coefficient parameters? (follow-up to second_order_check,
2026-09-12, code 51cd604.)

After the cavity/solvent3D reconnection the first-order quantity v.grad_theta E
for theta = field_dependent_charges_maps agrees with FD to 4e-6 (sid 28), but
the second-order quantity v.grad_theta L_F still disagrees by 7.5% on sid 28
(auto 5.680e-05 vs FD 6.143e-05, FD drift 0.2%), and on sid 628 the FD of L_F
itself drifts 103% between eps = 1e-4 and 3e-5, so nothing can be read there
without smaller steps. This script splits the second-order quantity by term.

THE SPLIT. L_F = mean_i (F_i - F_DFT,i)^2 with F = sum_k F_k, F_k = -dE_k/dR.
dL_F/dtheta = g . dF/dtheta with g = dL_F/dF = 2 (F - F_DFT)/N evaluated at
theta and then FROZEN. With g frozen the quantity Q_k = g . F_k(theta) is a
scalar per term whose directional derivative splits exactly:
    AD_k = v . grad_theta Q_k       (autograd through F_k built with create_graph)
    FD_k = [Q_k(theta+eps v) - Q_k(theta-eps v)] / 2 eps    (same frozen g)
    sum_k AD_k = v . grad_theta L_F        exactly (chain rule, g frozen)
    sum_k FD_k = FD(L_F) + O(eps^2)        (the second-order term cancels in the
                                            central difference)
D_k = AD_k - FD_k, sum_k D_k = D_total. Three step sizes so the drift of each
term's own FD is visible; the control group local_electron_energy does not
reach the solve, so only ITS term may carry a non-zero AD_k or FD_k.
Training-cache path, both frames of one NiN44 val pair, LIVE_POS=1,
GRAD_PASSES=0 (the fully unrolled solve, so the analytic adjoint's frozen-J_c
gap is not in play). Direction v for the first group is the same draw as in
second_order_check.py and param_grad_by_term.py (seed 7, first group).
"""
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
assert os.environ.get("MACE_PB1D_LIVE_POS") and not os.environ.get("MACE_PB1D_DFORCE")
assert os.environ.get("MACE_PB1D_GRAD_PASSES") == "0", "run with MACE_PB1D_GRAD_PASSES=0"
from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB
RUN = "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le"; os.chdir(RUN)
GROUPS = os.environ.get("KIT_GROUPS", "field_dependent_charges_maps,local_electron_energy").split(",")
EPS_REL = [float(x) for x in os.environ.get("KIT_EPS", "1e-4,3e-5,1e-5").split(",")]; SEED = 7
SMOKE = bool(int(os.environ.get("KIT_SMOKE", "0")))
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
SOLVES = {"n": 0}; _bd = {}; _bk = PB.PB1DBackend.solve_graph
def _wrap(self, *a, **k): _bd["b"] = self; SOLVES["n"] += 1; return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap
def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict): d.clear()
    torch.cuda.empty_cache()
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
FRAMES = [pair[0]] if SMOKE else list(pair)
if SMOKE: EPS_REL = EPS_REL[:1]
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
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    b["positions"].requires_grad_(True); return b

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

def term_force(tt, k, pos, create_graph):
    """F_k = -dE_k/dR; zero when E_k has no graph or no path to R."""
    if not tt[k].requires_grad: return None
    g = torch.autograd.grad(tt[k], pos, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
    return None if g is None else -g

def set_requires(group):
    for p in model.parameters(): p.requires_grad_(False)
    if group is not None:
        for _, p in PARAMS[group]: p.requires_grad_(True)

print(f"model {os.path.basename(mp)} @ {os.path.basename(cp)}  LIVE_POS=1  GRAD_PASSES=0  training-cache path  eps_rel {EPS_REL}")
for sid in FRAMES:
    q = float(atoms_by_sid[sid].info.get("total_charge", 0.0))
    f_dft = torch.as_tensor(np.asarray(atoms_by_sid[sid].get_array("forces")), dtype=torch.float64, device=device)
    N = f_dft.numel()
    for g in GROUPS:
        print("\n" + "=" * 78); print(f"sid {sid} ({'charged' if q else 'neutral'}), group {g} ({sum(p.numel() for _, p in PARAMS[g])} params)")
        plist = [p for _, p in PARAMS[g]]
        # ---- reference point: total force, L_F, frozen g, and the DIRECT v.dL_F autograd
        set_requires(g)
        b = batch(sid); n0 = SOLVES["n"]
        pred = model(b.to_dict(), compute_force=False, training=False)
        tt = term_tensors(pred, b); E = pred["energy"].sum()
        F = -torch.autograd.grad(E, b["positions"], create_graph=True, retain_graph=True)[0]
        LF = ((F - f_dft) ** 2).mean()
        gF = torch.autograd.grad(LF, plist, allow_unused=True)
        AD_LF = sum(float((gi * v).sum()) for gi, v in zip(gF, DIRS[g]) if gi is not None)
        gvec = (2.0 * (F - f_dft) / N).detach()          # dL_F/dF, frozen from here on
        LF0 = float(LF.detach()); F0 = F.detach().clone()
        print(f"  reference: E {float(E):.9f}  L_F {LF0:.6e}  ({SOLVES['n'] - n0} solves)  direct autograd v.dL_F {AD_LF:+.6e}")
        del pred, tt, E, F, LF, gF, b; _evict()
        # ---- AD_k, one term per forward (memory), each with its own create_graph force
        AD, NOTE = {}, []
        for k in TERMS:
            set_requires(g); b = batch(sid)
            pred = model(b.to_dict(), compute_force=False, training=False); tt = term_tensors(pred, b)
            Fk = term_force(tt, k, b["positions"], create_graph=True)
            if Fk is None or not Fk.requires_grad:
                AD[k] = 0.0; NOTE.append(SHORT[k]); del pred, tt, b; _evict(); continue
            Qk = (gvec * Fk).sum()
            gq = torch.autograd.grad(Qk, plist, allow_unused=True)
            AD[k] = sum(float((gi * v).sum()) for gi, v in zip(gq, DIRS[g]) if gi is not None)
            del pred, tt, Fk, Qk, gq, b; _evict()
        if NOTE: print(f"  terms whose force has no autograd path to theta (AD_k := 0): {', '.join(NOTE)}")
        # ---- FD_k with the frozen g, per eps, both signs; also FD(L_F) and the per-term force closure
        set_requires(None)
        saved = [p.detach().clone() for _, p in PARAMS[g]]
        FD = {}
        for er in EPS_REL:
            eps = er * TN[g]; Q = {}; LFs = {}; Fsum_err = 0.0
            for sign in (+1.0, -1.0):
                with torch.no_grad():
                    for (_, p), v, s0 in zip(PARAMS[g], DIRS[g], saved): p.copy_(s0 + sign * eps * v)
                b = batch(sid)
                with torch.enable_grad():
                    pred = model(b.to_dict(), compute_force=False, training=False); tt = term_tensors(pred, b)
                    Ftot = -torch.autograd.grad(pred["energy"].sum(), b["positions"], retain_graph=True)[0]
                    Fk = {k: term_force(tt, k, b["positions"], create_graph=False) for k in TERMS}
                Fk = {k: (torch.zeros_like(Ftot) if f is None else f.detach()) for k, f in Fk.items()}
                Fsum_err = max(Fsum_err, float((sum(Fk.values()) - Ftot.detach()).abs().max()))
                Q[sign] = {k: float((gvec * Fk[k]).sum()) for k in TERMS}
                LFs[sign] = float(((Ftot.detach() - f_dft) ** 2).mean())
                del pred, tt, Ftot, Fk, b; _evict()
            with torch.no_grad():
                for (_, p), s0 in zip(PARAMS[g], saved): p.copy_(s0)
            FD[er] = ({k: (Q[+1.0][k] - Q[-1.0][k]) / (2 * eps) for k in TERMS},
                      (LFs[+1.0] - LFs[-1.0]) / (2 * eps), Fsum_err)
        # ---- report
        er0, erL = EPS_REL[0], EPS_REL[-1]
        fdk0, fdLF0, fe0 = FD[er0]; fdkL, fdLFL, feL = FD[erL]
        sAD = sum(AD.values()); sFD0 = sum(fdk0.values()); sFDL = sum(fdkL.values())
        print(f"  CLOSURE AD: sum_k AD_k {sAD:+.6e} vs direct v.dL_F {AD_LF:+.6e}  |diff| {abs(sAD-AD_LF):.2e}  (exact by the chain rule with g frozen)")
        for er in EPS_REL:
            fdk, fdLF, fe = FD[er]
            print(f"  eps_rel {er:g}: FD(L_F) {fdLF:+.6e}   sum_k FD_k {sum(fdk.values()):+.6e}   |diff| {abs(sum(fdk.values())-fdLF):.2e} (O(eps^2))   per-term force closure max|sum_k F_k - F| {fe:.1e} eV/A")
        D0 = AD_LF - sFD0; DL = AD_LF - sFDL
        print(f"  D_total = direct AD - sum_k FD_k:  eps {er0:g}: {D0:+.6e} ({100*D0/max(abs(sFD0),1e-300):+.2f}% of FD)   eps {erL:g}: {DL:+.6e} ({100*DL/max(abs(sFDL),1e-300):+.2f}% of FD)")
        print(f"  {'term':>14} {'AD_k':>13} {'FD_k eps'+format(er0,'g'):>16} {'FD_k eps'+format(erL,'g'):>16} {'D_k (eps'+format(erL,'g')+')':>16} {'share':>8} {'FD_k drift':>11}")
        Dsum = 0.0
        for k in sorted(TERMS, key=lambda k: -abs(AD[k]-fdkL[k])):
            Dk = AD[k] - fdkL[k]; Dsum += Dk
            dr = abs(fdk0[k]-fdkL[k]) / max(abs(fdkL[k]), 1e-300)
            print(f"  {SHORT[k]:>14} {AD[k]:+13.5e} {fdk0[k]:+16.5e} {fdkL[k]:+16.5e} {Dk:+16.5e} {100*Dk/max(abs(DL),1e-300):+7.1f}% {dr:11.2e}")
        print(f"  {'sum D_k':>14} {'':>13} {'':>16} {'':>16} {Dsum:+16.5e}   vs D_total {DL:+.5e}  |diff| {abs(Dsum-DL):.2e}")
print("\nREADING: a term whose D_k stands above its own FD_k drift carries a real second-order truncation;")
print("a term whose FD_k drifts by as much as its D_k cannot be read at these step sizes. The control")
print("group must show a non-zero AD_k or FD_k only in its own term (local_e).")
print("DONE")
