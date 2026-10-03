"""Three-arm frozen comparison of the 3-D solvent readout (reviewer 2026-10-03; no training, no production-code change).
The electron model (node features, electron-density coefficients, 1-D solve, cavity) is frozen and captured once per frame;
only the solvent readout is refit (scipy LSQR on a LinearOperator, forward = the model's own grid evaluation, transpose =
autograd), on a fixed set of training pairs; the fixed 20 validation pairs are the acceptance set.
    V0  production as is (reference)
    A   current representation (rho_ch = B_ch(z) + env-projected lateral residual d_ch), production head refit        [same budget]
    B   current representation, head on CHARGE-STATE-augmented features [f, s1 f, s2 f, s3 f] with per-atom scalars
        s = standardized (Q_total, phi_1D(z_a) of the production 1-D solve, q_a = model per-atom charge if the model exposes one,
        else the solute potential cvhar(R_a)); production head kept on f (refit) + zero-initialised augmented head        [input only]
    C   POLARIZATION representation for the bound channel: psi(r) = sum_atoms GTO basis . c_psi, P = grad psi,
        rho_b = -w_b * div( S_diel(r) P ),  S_diel = the production cavity (clamped), w_b = the closure's Gaussian (sigma_b);
        c_psi from a zero-initialised equivariant linear readout of the same augmented features; NO 1-D background (the plane
        average comes from the 3-D field itself); ions = the production ion channel (so the arms are independent jobs).  P = grad psi is chosen so that the field is
        equivariant with the existing scalar basis and matches the VASPsol relation P = -chi grad phi (psi plays chi phi).
Fit objectives: A/B bound and ion = lateral target (the production's design; the plane average is not theirs to change);
C bound = the FULL DFT bound charge at the same points. Metrics on the full DFT grid for every arm and frame: bound / ion /
total (eps_3d, eps_lat, lat_corr, eps_pa), total solvent potential rms error and bound-only potential, spurious charge in
the forbidden zone (DFT-density cavity, smoothed by sigma = 0.25 A, below 1e-3 and 1e-4), charged-minus-neutral responses per
pair (eps_lat, lat_corr, eps_3d, potential). Diagnostic on the first KIT_NDIAG validation pairs: every arm re-evaluated with the
DFT-density cavity substituted (no refit).  Fitted parameters are saved (npy) for the native-grid check of arm C.
Usage: python s3d_head_arms.py <out_dir>   (cwd with ./data); env KIT_ARMS (A+B+C), KIT_HEAD_PAIRS (default: 20 training pairs
by stride), KIT_VAL_PAIRS (default: the fixed 20), KIT_NFIT, KIT_NHOLD, KIT_ITER, KIT_RIDGE, KIT_NDIAG, KIT_STAGES (fit+eval)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

import numpy as np
import torch
from ase.io import read
from e3nn import o3
from scipy.sparse.linalg import LinearOperator, lsqr

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import (_interp1_periodic, _interp3_periodic, normalized_gradient_envelope, poisson_phi_periodic)
from mace.modules.wrapper_ops import Linear

T_START = time.time()
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "arms_out"); OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
ints = lambda k, d: [int(s) for s in os.environ.get(k, d).replace("+", " ").split()]
ARMS = os.environ.get("KIT_ARMS", "A+B+C").replace("+", " ").split()
STAGES = set(os.environ.get("KIT_STAGES", "fit+eval").replace(",", "+").split("+"))
VAL_PAIRS = ints("KIT_VAL_PAIRS", "28 30 43 60 61 62 69 79 83 94 128 134 148 153 159 177 180 185 186 189")
N_HEAD = int(os.environ.get("KIT_N_HEAD_PAIRS", "16"))
NFIT = int(os.environ.get("KIT_NFIT", "300000")); NHOLD = int(os.environ.get("KIT_NHOLD", "100000"))
ITER = int(os.environ.get("KIT_ITER", "250")); TOL = float(os.environ.get("KIT_TOL", "1e-8")); RIDGE = float(os.environ.get("KIT_RIDGE", "1e-2"))
NDIAG = int(os.environ.get("KIT_NDIAG", "3")); NPROBE = int(os.environ.get("KIT_NPROBE", "32")); EVAL_V0 = os.environ.get("KIT_EVAL_V0", "1") == "1"
torch.set_default_dtype(torch.float64)
dev = torch.device(os.environ.get("KIT_DEVICE", "cuda:0"))
if dev.type == "cpu":
    _jl = torch.jit.load
    torch.jit.load = lambda f, map_location=None, **kw: _jl(f, map_location="cpu", **kw)
model = torch.load(MODEL, map_location=dev).to(dev); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend(); tp = backend._tp; params = backend.params
head = model.solvent3d_head
SIGMA_B = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
NSIG = len(head.sigmas); OUT_SCALE = float(getattr(head, "out_scale", head.OUT_SCALE))
IRR_IN = o3.Irreps(head.linear.irreps_in); IRR_OUT2 = o3.Irreps(head.linear.irreps_out); IRR_OUT1 = o3.Irreps(f"{NSIG}x0e + {NSIG}x1o + {NSIG}x2e")
N_AUG = 4                                                     # [f, s1 f, s2 f, s3 f]
man = json.load(open("data/solvent3d_grid_manifest.json")); ENT = {int(k): v for k, v in man["entries"].items()}
man_n = json.load(open("data/density3d_net_grid_manifest_npy.json")); ENT_N = {int(k): v for k, v in man_n["entries"].items()}
LBL = Path("data").resolve()
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
train_charged = sorted(s for s, a in ATOMS.items() if a.info["_split"] == "train" and s <= 200 and (s + 600) in ATOMS)
if os.environ.get("KIT_HEAD_PAIRS"):
    HEAD_PAIRS = ints("KIT_HEAD_PAIRS", "")
else:
    idx = np.unique(np.linspace(0, len(train_charged) - 1, N_HEAD).round().astype(int)); HEAD_PAIRS = [train_charged[i] for i in idx]
RESULTS = {"meta": dict(model=MODEL, arms=ARMS, head_pairs=HEAD_PAIRS, val_pairs=VAL_PAIRS, nfit=NFIT, nhold=NHOLD, iter=ITER, ridge=RIDGE,
                        sigma_b=SIGMA_B, out_scale=OUT_SCALE, irreps_in=str(IRR_IN), irreps_out2=str(IRR_OUT2), irreps_out1=str(IRR_OUT1),
                        n_train_charged=len(train_charged)), "frames": {}, "fits": [], "pairs": {}, "params": {}}
print(f"model {os.path.basename(MODEL)}; device {dev}; arms {ARMS}; stages {sorted(STAGES)}; head pairs ({len(HEAD_PAIRS)}) {HEAD_PAIRS}; val pairs {VAL_PAIRS}; "
      f"NFIT {NFIT} NHOLD {NHOLD} ITER {ITER} RIDGE {RIDGE}; head in {IRR_IN} -> {IRR_OUT2}; sigmas {head.sigmas}; sigma_b {SIGMA_B}; out_scale {OUT_SCALE}", flush=True)


def save():
    json.dump(RESULTS, open(OUT / "results.json", "w"), indent=1, default=float)


def elapsed():
    return f"{time.time() - T_START:7.0f}s"


# ------------------------------------------------------------------ capture (one forward per frame; nothing re-solved)
CTX = {}
CAP = {}
_orig_clo = PB.closure_from_fields


def clo_cap(n_e, cv, grid, params_, tp_):
    CAP["ne"] = n_e.detach().clone(); CAP["cv"] = cv.detach().clone()
    return _orig_clo(n_e, cv, grid, params_, tp_)


PB.closure_from_fields = clo_cap


def dft_to_model(arr_zyx, mshape):
    rho = torch.as_tensor(arr_zyx, device=dev).permute(2, 1, 0).contiguous()
    nx, ny, nz = mshape
    out = torch.empty(mshape, device=dev)
    ii, jj, kk = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev), torch.arange(nz, device=dev), indexing="ij")
    step = max(1, int(2_000_000 // (ny * nz)))
    for x0 in range(0, nx, step):
        x1 = min(x0 + step, nx)
        fm = torch.stack([ii[x0:x1].reshape(-1) / nx, jj[x0:x1].reshape(-1) / ny, kk[x0:x1].reshape(-1) / nz], dim=1).double()
        out[x0:x1] = _interp3_periodic(rho, fm).reshape(x1 - x0, ny, nz)
    return out


def capture(sid):
    atoms = ATOMS[sid]
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)
    hold = {}; orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw)
        cell_np = kw["cell"].detach().cpu().numpy().astype(float).reshape(3, 3)
        grid = backend._grid_for(cell_np, backend._grid_shape(cell_np), kw["positions"].device)
        cav = getattr(grid, "_solv3d_cavity", None)
        hold.update(res=res, kw=kw, grid=grid, cell_np=cell_np, n=hold.get("n", 0) + 1, cav=None if cav is None else (cav[0].clone(), cav[1].clone()))
        return res
    backend.solve_graph = spy
    try:
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    kw, res, grid = hold["kw"], hold["res"], hold["grid"]
    cell64 = torch.as_tensor(hold["cell_np"], device=dev)
    pf = torch.remainder(kw["positions"].detach().to(torch.float64) @ torch.linalg.inv(cell64), 1.0)
    s_ion3, s_diel3 = hold["cav"]
    V = float(grid.volume); mshape = tuple(s_diel3.shape); lz = float(hold["cell_np"][2, 2])
    # per-atom charge-state scalars
    q = float(kw["total_charge"]); n_at = pf.shape[0]
    phi_z = res["phi_z"].detach().double()
    phi_at = _interp1_periodic(phi_z, pf[:, 2])
    qa_key = None
    for k_, v_ in (out.items() if isinstance(out, dict) else []):
        if torch.is_tensor(v_) and "charge" in k_.lower() and v_.numel() == n_at:
            qa_key = k_; break
    if qa_key is not None:
        third = out[qa_key].detach().double().reshape(-1)
    else:
        third = _interp3_periodic(CAP["cv"].double(), pf)
    scal = torch.stack([torch.full((n_at,), q, device=dev), phi_at, third], dim=1)   # [N, 3] raw
    # DFT-density cavity on the model grid (package route), for the forbidden zone and the DFT-cavity diagnostic
    bl_row = backend._bl_index.get(sid)
    neutral_v = torch.from_numpy(np.ascontiguousarray(backend._bl_arr[bl_row][backend._bl_take][0])).to(dev).double()
    net_dft = dft_to_model(np.asarray(np.load(ENT_N[sid]["path"], mmap_mode="r"), dtype=np.float64), mshape)
    ne_dft = torch.clamp((neutral_v - net_dft * V) / V, min=0.0); del net_dft, neutral_v
    si_d, sd_d, _ = tp.create_cavity_torch(ne_dft, grid, params); del ne_dft
    ctx = dict(sid=sid, split=atoms.info["_split"], q=q, grid=grid, cell64=cell64, pf=pf, lz=lz, V=V, mshape=mshape,
               feats=kw["node_feats"].detach().clone(), c0=kw["s3d_coeffs"].detach().clone(), sig=kw["s3d_sigmas"], scal=scal, qa_key=qa_key,
               S=torch.clamp(s_diel3, 0.0, 1.0), env={"b": normalized_gradient_envelope(s_diel3, cell64), "i": torch.clamp(s_ion3, 0.0, 1.0)},
               S_dft=torch.clamp(sd_d, 0.0, 1.0), env_dft={"b": normalized_gradient_envelope(sd_d, cell64), "i": torch.clamp(si_d, 0.0, 1.0)},
               B={"b": res["rho_bound_z"].detach().clone().double(), "i": res["rho_ion_z"].detach().clone().double()},
               dsup={"b": res["s3d_obs"]["d_sup_b"].detach().clone(), "i": res["s3d_obs"]["d_sup_i"].detach().clone()}, nsolve=hold["n"])
    wb = getattr(grid, "_arms_wb", None)
    if wb is None:
        grid._arms_wb = tp._normalized_gaussian_kernel_g(grid, SIGMA_B)
    CTX[sid] = ctx
    return ctx


# ------------------------------------------------------------------ fields on the model grid
def gto(ctx, c):
    grid = ctx["grid"]
    return grid.ifft_real(backend._gto_net_density_g(grid, ctx["pf"], c, ctx["sig"])) / grid.volume


def dfield(ctx, ch, c, env=None):
    env = ctx["env"][ch] if env is None else env
    m = gto(ctx, c)
    r = (env * m).mean(dim=(0, 1)) / torch.clamp(env.mean(dim=(0, 1)), min=1.0e-12)
    return env * m - r[None, None, :] * env


def polfield(ctx, c_psi, S=None):
    """rho_b = -w_b * div( S grad psi ), psi = GTO(c_psi)."""
    grid = ctx["grid"]; S = ctx["S"] if S is None else S
    psi = gto(ctx, c_psi)
    gx, gy, gz, _ = grid.grad_from_recip(grid.fft(psi))
    return -grid.ifft_real(grid._arms_wb * grid.div_real_vector(S * gx, S * gy, S * gz))


def to_blocks(flat, n_out):
    n = flat.shape[0]; blocks = flat.new_zeros(n, n_out, 9); off = 0
    for ell in range(3):
        w = 2 * ell + 1
        blocks[:, :, ell * ell:(ell + 1) * (ell + 1)] = flat[:, off:off + n_out * w].view(n, n_out, w); off += n_out * w
    return blocks


# ------------------------------------------------------------------ readouts (linear, equivariant); parameters as flat vectors
LIN_AUG2 = Linear(IRR_IN * N_AUG, IRR_OUT2).to(dev)          # arm B: augmented features -> both channels
LIN_PSI = Linear(IRR_IN * N_AUG, IRR_OUT1).to(dev)            # arm C: augmented features -> psi coefficients
for mod in (LIN_AUG2, LIN_PSI):
    for p in mod.parameters():
        torch.nn.init.zeros_(p); p.requires_grad_(False)
SCAL_MU = SCAL_SD = None


def aug_feats(ctx):
    s = (ctx["scal"] - SCAL_MU) / SCAL_SD
    f = ctx["feats"]
    return torch.cat([f, s[:, 0:1] * f, s[:, 1:2] * f, s[:, 2:3] * f], dim=-1)


def flat_params(mod):
    return torch.cat([p.detach().reshape(-1) for _, p in mod.named_parameters()])


def unflat(mod, x):
    out, off = {}, 0
    for n, p in mod.named_parameters():
        out[n] = x[off:off + p.numel()].view(p.shape); off += p.numel()
    return out


N_HEAD_P = int(flat_params(head).numel()); N_AUG_P = int(flat_params(LIN_AUG2).numel()); N_PSI_P = int(flat_params(LIN_PSI).numel())


def coeffs_A(x, ctx):
    """production head with parameters x -> [N, 2, nsig, 9] (ion gated)."""
    return torch.func.functional_call(head, unflat(head, x), (ctx["feats"], torch.full((ctx["feats"].shape[0],), ctx["q"], device=dev)))


def coeffs_B(x, ctx):
    xh, xa = x[:N_HEAD_P], x[N_HEAD_P:]
    c = coeffs_A(xh, ctx)
    flat = torch.func.functional_call(LIN_AUG2, unflat(LIN_AUG2, xa), (aug_feats(ctx),))
    ca = to_blocks(flat, 2 * NSIG).view(-1, 2, NSIG, 9) * OUT_SCALE
    return c + torch.stack([ca[:, 0], ca[:, 1] * ctx["q"]], dim=1)


def coeffs_psi(x, ctx):
    flat = torch.func.functional_call(LIN_PSI, unflat(LIN_PSI, x), (aug_feats(ctx),))
    return to_blocks(flat, NSIG) * OUT_SCALE                   # [N, nsig, 9]


# ------------------------------------------------------------------ labels and points
LABEL = {}


def load_label(sid):
    e = ENT[sid]
    rb = np.asarray(np.load(LBL / e["path_b"], mmap_mode="r"), dtype=np.float64); ri = np.asarray(np.load(LBL / e["path_i"], mmap_mode="r"), dtype=np.float64)
    with np.load(LBL / e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return {"b": rb, "i": ri}, lat


def prep_labels(sid):
    lab, lat = load_label(sid); shape = lab["b"].shape; nz, ny, nx = shape
    rng = np.random.default_rng(1000 + sid); lin = rng.permutation(nz * ny * nx)[:NFIT + NHOLD]
    iz, rem = np.divmod(lin, ny * nx); iy, ix = np.divmod(rem, nx)
    frac = np.stack([ix / nx, iy / ny, iz / nz], axis=1)
    L = dict(shape=shape, lat=lat, fr_f=torch.as_tensor(frac[:NFIT], device=dev), fr_h=torch.as_tensor(frac[NFIT:], device=dev), lat_t={}, full_t={})
    for ch in ("b", "i"):
        full = lab[ch].reshape(-1); latp = (lab[ch] - lab[ch].mean(axis=(1, 2))[:, None, None]).reshape(-1)
        L["lat_t"][ch] = (torch.as_tensor(latp[lin[:NFIT]], device=dev), torch.as_tensor(latp[lin[NFIT:]], device=dev))
        L["full_t"][ch] = (torch.as_tensor(full[lin[:NFIT]], device=dev), torch.as_tensor(full[lin[NFIT:]], device=dev))
    LABEL[sid] = L
    return L


# ------------------------------------------------------------------ full-grid evaluation
def on_dft_grid(Bz, dgrid, shape):
    nz, ny, nx = shape; out = torch.empty(shape, device=dev)
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    for z0 in range(0, nz, 25):
        z1 = min(z0 + 25, nz); fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
        frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
        v = _interp3_periodic(dgrid, frac)
        if Bz is not None:
            v = v + _interp1_periodic(Bz, frac[:, 2])
        out[z0:z1] = v.reshape(z1 - z0, ny, nx)
    return out


def metrics(M, D, dV):
    d = M - D; pm, pd = M.mean(dim=(1, 2)), D.mean(dim=(1, 2)); Ml, Dl = M - pm[:, None, None], D - pd[:, None, None]
    sD = float(D.abs().sum()); sDl = float(Dl.abs().sum())
    return dict(eps_3d=float(d.abs().sum()) / max(sD, 1e-300), eps_lat=float((Ml - Dl).abs().sum()) / max(sDl, 1e-300),
                lat_corr=float((Ml * Dl).sum() / torch.clamp(Ml.norm() * Dl.norm(), min=1e-300)),
                eps_pa=float((pm - pd).abs().sum()) / max(float(pd.abs().sum()), 1e-300), norm_ratio=float(M.norm() / torch.clamp(D.norm(), min=1e-300)),
                net_ml_e=float(M.sum()) * dV, net_dft_e=float(D.sum()) * dV)


def physics(M, D, lat):
    cell = torch.as_tensor(lat, device=dev)
    pM = poisson_phi_periodic(M.permute(2, 1, 0).contiguous(), cell); pD = poisson_phi_periodic(D.permute(2, 1, 0).contiguous(), cell)
    return dict(phi_rms_err=float(((pM - pD) ** 2).mean().sqrt()), phi_rms_dft=float((pD ** 2).mean().sqrt()))


ZONE = {}


def zone_masks(sid):
    """forbidden zone on the DFT grid: DFT-density cavity s_diel (model grid) smoothed with sigma = 0.25 A, interpolated, < 1e-3 / 1e-4."""
    if sid in ZONE:
        return ZONE[sid]
    ctx = CTX[sid]; grid = ctx["grid"]; L = LABEL[sid]
    ker = torch.exp(-0.5 * (0.25 ** 2) * grid.gsq * (2.0 * np.pi) ** 2)
    sm = grid.ifft_real(ker * grid.fft(ctx["S_dft"]))
    sm_d = on_dft_grid(None, sm, L["shape"])
    smm = grid.ifft_real(ker * grid.fft(ctx["S"]))
    ZONE[sid] = dict(dft_1e3=sm_d < 1e-3, dft_1e4=sm_d < 1e-4, model_1e3=on_dft_grid(None, smm, L["shape"]) < 1e-3)
    return ZONE[sid]


def evaluate(sid, fields_mg, tag):
    """fields_mg: {'b': (Bz or None, grid), 'i': (Bz or None, grid)} on the model grid -> metrics on the DFT grid."""
    L = LABEL[sid]; lab, lat = load_label(sid); dV = abs(float(np.linalg.det(lat))) / lab["b"].size
    out, F = {}, {}
    for ch in ("b", "i"):
        Bz, dg = fields_mg[ch]
        M = on_dft_grid(Bz, dg, L["shape"]); D = torch.as_tensor(lab[ch], device=dev)
        out[ch] = metrics(M, D, dV)
        for key, fr in (("J_fit", L["fr_f"]), ("J_hold", L["fr_h"])):
            tg = (L["full_t"] if Bz is None else L["lat_t"])[ch][0 if key == "J_fit" else 1]
            pr = _interp3_periodic(dg, fr)
            if Bz is not None:
                pass   # lateral objective: residual only vs lateral target
            else:
                pass
            out[ch][key] = float(((pr - tg) ** 2).sum() / torch.clamp((tg ** 2).sum(), min=1e-300))
        F[ch] = (M, D)
    zm = zone_masks(sid); Mb, Db = F["b"]
    out["b"]["zone_abs_frac"] = {k: float(Mb[m].abs().sum() / Db.abs().sum()) for k, m in zm.items()}
    out["b"]["zone_abs_frac_dft"] = {k: float(Db[m].abs().sum() / Db.abs().sum()) for k, m in zm.items()}
    out["b"]["phi_b"] = physics(Mb, Db, lat)
    Mt, Dt = F["b"][0] + F["i"][0], F["b"][1] + F["i"][1]
    out["t"] = metrics(Mt, Dt, dV); out["t"].update(physics(Mt, Dt, lat))
    RESULTS["frames"].setdefault(str(sid), {})[tag] = out
    return F


def response(k, tag, Fc, Fn):
    lab, lat = load_label(k); dV = abs(float(np.linalg.det(lat))) / lab["b"].size
    o = {ch: metrics(Fc[ch][0] - Fn[ch][0], Fc[ch][1] - Fn[ch][1], dV) for ch in ("b", "i")}
    o["b"].update(phi_b=physics(Fc["b"][0] - Fn["b"][0], Fc["b"][1] - Fn["b"][1], lat))
    Mt = (Fc["b"][0] + Fc["i"][0]) - (Fn["b"][0] + Fn["i"][0]); Dt = (Fc["b"][1] + Fc["i"][1]) - (Fn["b"][1] + Fn["i"][1])
    o["t"] = metrics(Mt, Dt, dV); o["t"].update(physics(Mt, Dt, lat))
    RESULTS["pairs"].setdefault(str(k), {})[tag] = o


# ------------------------------------------------------------------ LSQR on a LinearOperator (Jacobi column scaling + ridge)
def lsqr_fit(blocks, field_of, x0, label, ridge_rel=RIDGE, tol=TOL, iters=ITER, n_probe=NPROBE):
    """blocks: list of (sid, ch, kind) with kind 'lat' or 'full'; field_of(x, sid, ch) -> model-grid field whose values at the
    fit points are compared with the target of that kind."""
    x0_t = torch.as_tensor(x0, device=dev)

    def F_block(x, sid, ch, kind):
        return _interp3_periodic(field_of(x, sid, ch), LABEL[sid]["fr_f"])
    with torch.no_grad():
        F0 = [F_block(x0_t, *b) for b in blocks]
    tg = [(LABEL[b[0]]["lat_t"] if b[2] == "lat" else LABEL[b[0]]["full_t"])[b[1]][0] for b in blocks]
    r0 = torch.cat([t - f for t, f in zip(tg, F0)]); sizes = [int(t.numel()) for t in tg]; ncall = {"mv": 0, "rmv": 0}

    def mv(v):
        ncall["mv"] += 1; v_t = torch.as_tensor(np.asarray(v, dtype=np.float64).ravel(), device=dev)
        with torch.no_grad():
            return torch.cat([F_block(x0_t + v_t, *b) - f for b, f in zip(blocks, F0)]).cpu().numpy()

    def rmv(u):
        ncall["rmv"] += 1; u_t = torch.as_tensor(np.asarray(u, dtype=np.float64).ravel(), device=dev)
        g = torch.zeros_like(x0_t); off = 0
        for b, n in zip(blocks, sizes):
            x = x0_t.clone().requires_grad_(True); y = F_block(x, *b)
            g += torch.autograd.grad(y, x, grad_outputs=u_t[off:off + n])[0]; off += n
        return g.cpu().numpy()
    nrow, ncol = sum(sizes), x0_t.numel(); t0 = time.time()
    v = np.random.default_rng(5).standard_normal(ncol); v /= np.linalg.norm(v)
    for _ in range(8):
        w = rmv(mv(v)); v = w / np.linalg.norm(w)
    ridge = ridge_rel * float(np.sqrt(np.linalg.norm(rmv(mv(v)))))
    rng = np.random.default_rng(7); acc = np.zeros(ncol)
    for _ in range(n_probe):
        acc += rmv(rng.standard_normal(nrow)) ** 2
    cn = np.sqrt(acc / n_probe); cn_aug = np.sqrt(cn ** 2 + ridge ** 2); colscale = 1.0 / np.maximum(cn_aug, 1e-6 * cn_aug.max())
    opS = LinearOperator((nrow + ncol, ncol), matvec=lambda y: np.concatenate([mv(np.asarray(y).ravel() * colscale), ridge * np.asarray(y).ravel() * colscale]),
                         rmatvec=lambda u: (rmv(np.asarray(u)[:nrow]) + ridge * np.asarray(u)[nrow:]) * colscale, dtype=np.float64)
    rhs = np.concatenate([r0.cpu().numpy(), np.zeros(ncol)])
    y, istop, itn, r1norm, r2norm, anorm, acond, arnorm, ynorm = lsqr(opS, rhs, atol=tol, btol=tol, iter_lim=iters)[:9]
    dx = y * colscale; r1 = float(np.linalg.norm(mv(dx) - r0.cpu().numpy()))
    rec = dict(label=label, n_blocks=len(blocks), n_unknown=int(ncol), n_rows=int(nrow), ridge_rel=ridge_rel, ridge=ridge, istop=int(istop), itn=int(itn),
               r0_norm=float(r0.norm()), r1norm=r1, rel_resid=r1 / max(float(r0.norm()), 1e-300), target_norm=float(torch.cat(tg).norm()),
               J_fit=(r1 / float(torch.cat(tg).norm())) ** 2, acond=float(acond), dx_norm=float(np.linalg.norm(dx)), x0_norm=float(x0_t.norm()),
               seconds=time.time() - t0, n_matvec=ncall["mv"], n_rmatvec=ncall["rmv"])
    RESULTS["fits"].append(rec); save()
    print(f"[{elapsed()}] LSQR {label}: istop {istop} itn {itn} rel_resid {rec['rel_resid']:.4f} J_fit {rec['J_fit']:.4f} acond {acond:.2e} |dx| {rec['dx_norm']:.3e} {rec['seconds']:.0f}s mv {ncall['mv']} rmv {ncall['rmv']}", flush=True)
    return x0 + dx, rec


# ------------------------------------------------------------------ frames
hsids = [s for k in HEAD_PAIRS for s in (k, k + 600)]
vsids = [s for k in VAL_PAIRS for s in (k, k + 600)]
for sid in hsids + vsids:
    t0 = time.time(); ctx = capture(sid); prep_labels(sid)
    with torch.no_grad():
        rep = {ch: float((dfield(ctx, ch, ctx["c0"][:, k]) - ctx["dsup"][ch]).abs().max() / torch.clamp(ctx["dsup"][ch].abs().max(), min=1e-300)) for k, ch in ((0, "b"), (1, "i"))}
    ctx["dsup"] = None
    RESULTS["frames"].setdefault(str(sid), {})["info"] = dict(split=ctx["split"], q=ctx["q"], n_atoms=int(ctx["pf"].shape[0]), grid=list(ctx["mshape"]), repro=rep, qa_key=ctx["qa_key"],
                                                              scal_mean=ctx["scal"].mean(0).tolist(), scal_std=ctx["scal"].std(0).tolist())
    if len(CTX) % 10 == 1:
        print(f"[{elapsed()}] captured {len(CTX)} frames (sid {sid}: repro b {rep['b']:.1e} i {rep['i']:.1e}; charge-state scalars: Q {ctx['q']:+.2f}, phi_1D(z_a) mean {float(ctx['scal'][:, 1].mean()):+.3f}, third ({ctx['qa_key'] or 'cvhar(R_a)'}) mean {float(ctx['scal'][:, 2].mean()):+.3f}; {time.time() - t0:.0f}s)", flush=True)
allscal = torch.cat([CTX[s]["scal"] for s in hsids]); SCAL_MU = allscal.mean(0); SCAL_SD = torch.clamp(allscal.std(0), min=1e-8)
RESULTS["meta"]["scal_standardisation"] = dict(mean=SCAL_MU.tolist(), std=SCAL_SD.tolist(), third=CTX[hsids[0]]["qa_key"] or "cvhar(R_a)")
print(f"[{elapsed()}] {len(CTX)} frames captured; charge-state scalar standardisation mean {SCAL_MU.tolist()} std {SCAL_SD.tolist()}; parameters: head {N_HEAD_P}, aug head {N_AUG_P}, psi head {N_PSI_P}", flush=True)
save()

# ------------------------------------------------------------------ fits
X = {}
x0_head = flat_params(head).cpu().numpy(); x0_aug = flat_params(LIN_AUG2).cpu().numpy(); x0_psi = flat_params(LIN_PSI).cpu().numpy()
charged = [s for s in hsids if abs(CTX[s]["q"]) > 1e-6]
if "fit" in STAGES:
    if "A" in ARMS:
        xb, _ = lsqr_fit([(s, "b", "lat") for s in hsids], lambda x, s, ch: dfield(CTX[s], ch, coeffs_A(x, CTX[s])[:, 0]), x0_head, f"A bound (production head refit), {len(hsids)} frames")
        xi, _ = lsqr_fit([(s, "i", "lat") for s in charged], lambda x, s, ch: dfield(CTX[s], ch, coeffs_A(x, CTX[s])[:, 1]), x0_head, f"A ion (gated), {len(charged)} frames")
        X["A"] = dict(head=x0_head + (xb - x0_head) + (xi - x0_head)); np.save(OUT / "params_A.npy", X["A"]["head"])
    if "B" in ARMS:
        x0B = np.concatenate([x0_head, x0_aug])
        xb, _ = lsqr_fit([(s, "b", "lat") for s in hsids], lambda x, s, ch: dfield(CTX[s], ch, coeffs_B(x, CTX[s])[:, 0]), x0B, f"B bound (head + charge-state augmented head), {len(hsids)} frames")
        xi, _ = lsqr_fit([(s, "i", "lat") for s in charged], lambda x, s, ch: dfield(CTX[s], ch, coeffs_B(x, CTX[s])[:, 1]), x0B, f"B ion (gated, augmented), {len(charged)} frames")
        X["B"] = dict(full=x0B + (xb - x0B) + (xi - x0B)); np.save(OUT / "params_B.npy", X["B"]["full"])
    if "C" in ARMS:
        xp, _ = lsqr_fit([(s, "b", "full") for s in hsids], lambda x, s, ch: polfield(CTX[s], coeffs_psi(x, CTX[s])), x0_psi, f"C bound (polarization psi head, FULL target), {len(hsids)} frames")
        X["C"] = dict(psi=xp); np.save(OUT / "params_C_psi.npy", xp)
else:
    if "A" in ARMS: X["A"] = dict(head=np.load(OUT / "params_A.npy"))
    if "B" in ARMS: X["B"] = dict(full=np.load(OUT / "params_B.npy"))
    if "C" in ARMS: X["C"] = dict(psi=np.load(OUT / "params_C_psi.npy"))
save()


def fields_for(arm, sid, dft_cavity=False):
    ctx = CTX[sid]; env = ctx["env_dft"] if dft_cavity else ctx["env"]; S = ctx["S_dft"] if dft_cavity else ctx["S"]
    with torch.no_grad():
        if arm == "V0":
            c = ctx["c0"]
            return {"b": (ctx["B"]["b"], dfield(ctx, "b", c[:, 0], env["b"])), "i": (ctx["B"]["i"], dfield(ctx, "i", c[:, 1], env["i"]))}
        if arm == "A":
            c = coeffs_A(torch.as_tensor(X["A"]["head"], device=dev), ctx)
            return {"b": (ctx["B"]["b"], dfield(ctx, "b", c[:, 0], env["b"])), "i": (ctx["B"]["i"], dfield(ctx, "i", c[:, 1], env["i"]))}
        if arm == "B":
            cB = coeffs_B(torch.as_tensor(X["B"]["full"], device=dev), ctx)
            return {"b": (ctx["B"]["b"], dfield(ctx, "b", cB[:, 0], env["b"])), "i": (ctx["B"]["i"], dfield(ctx, "i", cB[:, 1], env["i"]))}
        if arm == "C":
            return {"b": (None, polfield(ctx, coeffs_psi(torch.as_tensor(X["C"]["psi"], device=dev), ctx), S)), "i": (ctx["B"]["i"], dfield(ctx, "i", ctx["c0"][:, 1], env["i"]))}
    raise ValueError(arm)


# ------------------------------------------------------------------ evaluation: training pairs, validation pairs, DFT-cavity diagnostic
if "eval" in STAGES:
    arms_eval = (["V0"] if EVAL_V0 else []) + [a for a in ARMS if a in X]
    for group, pairs in (("train", HEAD_PAIRS), ("val", VAL_PAIRS)):
        for k in pairs:
            F = {}
            for arm in arms_eval:
                F[arm] = (evaluate(k, fields_for(arm, k), arm), evaluate(k + 600, fields_for(arm, k + 600), arm))
                response(k, arm, F[arm][0], F[arm][1])
            if group == "val" and VAL_PAIRS.index(k) < NDIAG:
                for arm in arms_eval:
                    Fd = (evaluate(k, fields_for(arm, k, True), arm + "+DFTcav"), evaluate(k + 600, fields_for(arm, k + 600, True), arm + "+DFTcav"))
                    response(k, arm + "+DFTcav", Fd[0], Fd[1])
            del F; save()
            r = RESULTS["frames"][str(k)]; rn = RESULTS["frames"][str(k + 600)]; rp = RESULTS["pairs"][str(k)]
            print(f"[{elapsed()}] {group} pair {k}: " + " | ".join(
                f"{arm}: c eps3d {r[arm]['b']['eps_3d']:.3f} lat {r[arm]['b']['eps_lat']:.3f} pa {r[arm]['b']['eps_pa']:.3f} zone {r[arm]['b']['zone_abs_frac']['dft_1e3']:.3f}; n eps3d {rn[arm]['b']['eps_3d']:.3f}; "
                f"resp lat {rp[arm]['b']['eps_lat']:.3f} corr {rp[arm]['b']['lat_corr']:+.2f} eps3d {rp[arm]['b']['eps_3d']:.3f}; phi_t {r[arm]['t']['phi_rms_err']:.3f}/{rp[arm]['t']['phi_rms_err']:.3f}" for arm in arms_eval), flush=True)
            if group == "val" and VAL_PAIRS.index(k) < NDIAG:
                print(f"           DFT-cavity substituted: " + " | ".join(f"{arm}: c eps3d {r[arm + '+DFTcav']['b']['eps_3d']:.3f} lat {r[arm + '+DFTcav']['b']['eps_lat']:.3f} zone {r[arm + '+DFTcav']['b']['zone_abs_frac']['dft_1e3']:.3f}; resp lat {rp[arm + '+DFTcav']['b']['eps_lat']:.3f} corr {rp[arm + '+DFTcav']['b']['lat_corr']:+.2f}" for arm in arms_eval), flush=True)
    # summary over the 20 validation pairs (medians and means)
    summ = {}
    for arm in arms_eval:
        rows_c = [RESULTS["frames"][str(k)][arm] for k in VAL_PAIRS]; rows_n = [RESULTS["frames"][str(k + 600)][arm] for k in VAL_PAIRS]; rows_p = [RESULTS["pairs"][str(k)][arm] for k in VAL_PAIRS]
        def agg(rows, path):
            vals = []
            for r in rows:
                v = r
                for p in path: v = v[p]
                vals.append(v)
            return dict(median=float(np.median(vals)), mean=float(np.mean(vals)))
        summ[arm] = dict(charged_eps3d=agg(rows_c, ["b", "eps_3d"]), charged_lat=agg(rows_c, ["b", "eps_lat"]), charged_pa=agg(rows_c, ["b", "eps_pa"]), charged_zone=agg(rows_c, ["b", "zone_abs_frac", "dft_1e3"]), charged_zone4=agg(rows_c, ["b", "zone_abs_frac", "dft_1e4"]),
                         neutral_eps3d=agg(rows_n, ["b", "eps_3d"]), neutral_lat=agg(rows_n, ["b", "eps_lat"]), neutral_zone=agg(rows_n, ["b", "zone_abs_frac", "dft_1e3"]),
                         resp_lat=agg(rows_p, ["b", "eps_lat"]), resp_corr=agg(rows_p, ["b", "lat_corr"]), resp_eps3d=agg(rows_p, ["b", "eps_3d"]),
                         ion_charged_eps3d=agg(rows_c, ["i", "eps_3d"]), ion_resp_eps3d=agg(rows_p, ["i", "eps_3d"]),
                         phi_total_charged=agg(rows_c, ["t", "phi_rms_err"]), phi_total_neutral=agg(rows_n, ["t", "phi_rms_err"]), phi_total_resp=agg(rows_p, ["t", "phi_rms_err"]),
                         phi_bound_charged=agg(rows_c, ["b", "phi_b", "phi_rms_err"]), phi_bound_resp=agg(rows_p, ["b", "phi_b", "phi_rms_err"]))
    RESULTS["summary_val"] = summ; save()
    print(f"[{elapsed()}] VALIDATION SUMMARY (20 pairs, median / mean):")
    for arm in arms_eval:
        s = summ[arm]
        print(f"   {arm:>3}: charged eps3d {s['charged_eps3d']['median']:.3f}/{s['charged_eps3d']['mean']:.3f} lat {s['charged_lat']['median']:.3f} pa {s['charged_pa']['median']:.3f} zone1e-3 {s['charged_zone']['median']:.4f} zone1e-4 {s['charged_zone4']['median']:.4f} | "
              f"neutral eps3d {s['neutral_eps3d']['median']:.3f} lat {s['neutral_lat']['median']:.3f} | response lat {s['resp_lat']['median']:.3f} corr {s['resp_corr']['median']:+.3f} eps3d {s['resp_eps3d']['median']:.3f} | "
              f"ion c {s['ion_charged_eps3d']['median']:.3f} resp {s['ion_resp_eps3d']['median']:.3f} | phi total c {s['phi_total_charged']['median']:.3f} n {s['phi_total_neutral']['median']:.3f} resp {s['phi_total_resp']['median']:.3f} | phi bound c {s['phi_bound_charged']['median']:.3f} resp {s['phi_bound_resp']['median']:.3f}", flush=True)
print(f"[{elapsed()}] done", flush=True)
