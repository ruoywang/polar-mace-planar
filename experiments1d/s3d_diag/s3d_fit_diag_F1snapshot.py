"""Offline diagnosis of the residual-3D solvent charge of prod500_vsolv_fix (user-approved plan 2026-09-30):
no training, no production-code change. Everything is computed with the model's own functions.

Model field per channel ch (b = bound, i = ionic), exactly the production construction:
    rho_ch(r) = B_ch(z) + d_ch(r),   d_ch = env_ch * m_ch - env_ch * <env_ch m_ch>_xy / <env_ch>_xy
    m_ch      = sum_atoms GTO basis (sigmas 0.5/1/2 A, l <= 2) . c[a, ch]      (backend._gto_net_density_g)
    env_b     = |grad s_diel| / max (frozen cavity of the final solve),  env_i = clamp(s_ion, 0, 1)
    c         = head(node_feats) * out_scale; ion channel multiplied by the frame's total charge q.
The map c -> d_ch(DFT points) is linear (fixed envelopes / projection / background). Fits minimise the squared
error of the LATERAL part only, sum_p (d(p) - t(p))^2 with t = rho_DFT - <rho_DFT>_xy at uniform DFT grid points
(the training pack is uniform too), with scipy LSQR on a LinearOperator: forward = the model's own evaluation,
transpose = autograd of the same forward. Nothing is stored as a dense matrix.

Variants (each reported on held-out points and on the full DFT grid):
  V0  current model coefficients
  V1  free coefficients per frame (bound; ionic on charged frames with the q gate = the current model's freedom;
      ionic on neutral frames WITHOUT the gate is reported separately and is NOT a capability of the current model)
  V2  bound coefficients shared by the charged frame and its neutral twin (each keeps its own envelope/projection)
  V3  the real shared head: node features fixed, the head's linear weights refit on training frames, evaluated on
      validation frames (and the test pair for display)
  V4  ionic background candidate, clearly a candidate: B_i(z) * alpha(r), alpha = env_i / <env_i>_xy
      (plane means unchanged), with the current coefficients and refit
Plus the grid round trip DFT grid -> model grid -> DFT grid (trilinear, the model's own interpolation).

Usage: python s3d_fit_diag.py <out_dir>   (cwd must hold ./data and ./cal1_train.json)
Env: KIT_MODE smoke|full, KIT_FIT_PAIRS, KIT_HEAD_PAIRS, KIT_VAL_PAIRS, KIT_TEST_PAIRS (space-separated charged sids),
     KIT_NFIT, KIT_NHOLD (points per frame), KIT_ITER, KIT_TOL, KIT_STAGES (plus-separated: repro+rt+free+pair+head+bg)
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
from scipy.sparse.linalg import LinearOperator, lsqr

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.solvent3d import (_interp1_periodic, _interp3_periodic, normalized_gradient_envelope,
                                    poisson_phi_periodic)

T_START = time.time()
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "diag_out"); OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
MODE = os.environ.get("KIT_MODE", "smoke")
ints = lambda k, d: [int(s) for s in os.environ.get(k, d).replace("+", " ").split()]
FIT_PAIRS = ints("KIT_FIT_PAIRS", "1")
HEAD_PAIRS = ints("KIT_HEAD_PAIRS", "")
VAL_PAIRS = ints("KIT_VAL_PAIRS", "")
TEST_PAIRS = ints("KIT_TEST_PAIRS", "")
NFIT = int(os.environ.get("KIT_NFIT", "200000")); NHOLD = int(os.environ.get("KIT_NHOLD", "200000"))
ITER = int(os.environ.get("KIT_ITER", "400")); TOL = float(os.environ.get("KIT_TOL", "1e-8"))
RIDGES = [float(v) for v in os.environ.get("KIT_RIDGES", "1e-2+3e-3").split("+")]          # bound fits (V1, V2)
ION_RIDGES = [float(v) for v in os.environ.get("KIT_ION_RIDGES", "3e-3").split("+")]       # ionic fits (V1, V1u, V4)
HEAD_RIDGE = os.environ.get("KIT_HEAD_RIDGE", "none"); HEAD_RIDGE = None if HEAD_RIDGE == "none" else float(HEAD_RIDGE)
HEAD_ITER = int(os.environ.get("KIT_HEAD_ITER", "3000"))
STAGES = set(os.environ.get("KIT_STAGES", "repro+rt+free+pair+head+bg").replace(",", "+").split("+"))  # sbatch --export splits on commas
torch.set_default_dtype(torch.float64)
dev = torch.device("cuda:0")   # the backend keys its grid cache by str(positions.device)
model = torch.load(MODEL, map_location=dev).to(dev)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend()
head = model.solvent3d_head
print(f"model {os.path.basename(MODEL)}; mode {MODE}; stages {sorted(STAGES)}; fit pairs {FIT_PAIRS}; head pairs {HEAD_PAIRS}; "
      f"val pairs {VAL_PAIRS}; test pairs {TEST_PAIRS}; NFIT {NFIT} NHOLD {NHOLD} ITER {ITER} TOL {TOL}", flush=True)
print(f"head irreps in {head.linear.irreps_in} -> out {head.linear.irreps_out}; sigmas {head.sigmas}; out_scale {head.out_scale}", flush=True)

man = json.load(open("data/solvent3d_grid_manifest.json")); ENT = {int(k): v for k, v in man["entries"].items()}
LBL = Path("data").resolve()
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp
        ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
               "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces"})
RESULTS = {"meta": dict(model=MODEL, mode=MODE, nfit=NFIT, nhold=NHOLD, iter=ITER, tol=TOL, fit_pairs=FIT_PAIRS,
                        head_pairs=HEAD_PAIRS, val_pairs=VAL_PAIRS, test_pairs=TEST_PAIRS), "frames": {}, "fits": [],
           "pairs": {}, "roundtrip": {}}


def save():
    json.dump(RESULTS, open(OUT / "results.json", "w"), indent=1, default=float)


def elapsed():
    return f"{time.time() - T_START:7.0f}s"


# ------------------------------------------------------------------ capture one frame (one forward, nothing re-solved)
CTX = {}


def capture(sid):
    atoms = ATOMS[sid]
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)
    hold = {}
    orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw)
        cell_np = kw["cell"].detach().cpu().numpy().astype(float).reshape(3, 3)
        grid = backend._grid_for(cell_np, backend._grid_shape(cell_np), kw["positions"].device)
        cav = getattr(grid, "_solv3d_cavity", None)
        hold.update(res=res, kw=kw, grid=grid, cell_np=cell_np, n=hold.get("n", 0) + 1,
                    cav=None if cav is None else (cav[0].clone(), cav[1].clone()))
        return res
    backend.solve_graph = spy
    try:
        with torch.no_grad():
            model(batch.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    kw, res, grid = hold["kw"], hold["res"], hold["grid"]
    cell64 = torch.as_tensor(hold["cell_np"], device=dev)
    pf = torch.remainder(kw["positions"].detach().to(torch.float64) @ torch.linalg.inv(cell64), 1.0)
    s_ion3, s_diel3 = hold["cav"]
    ctx = dict(sid=sid, split=atoms.info["_split"], q=float(kw["total_charge"]), grid=grid, cell64=cell64, pf=pf,
               feats=kw["node_feats"].detach().clone(), c0=kw["s3d_coeffs"].detach().clone(),
               sig=kw["s3d_sigmas"], env={"b": normalized_gradient_envelope(s_diel3, cell64), "i": torch.clamp(s_ion3, 0.0, 1.0)},
               B={"b": res["rho_bound_z"].detach().clone().to(torch.float64), "i": res["rho_ion_z"].detach().clone().to(torch.float64)},
               dsup={"b": res["s3d_obs"]["d_sup_b"].detach().clone(), "i": res["s3d_obs"]["d_sup_i"].detach().clone()},
               nsolve=hold["n"])
    CTX[sid] = ctx
    return ctx


def dfield(ctx, ch, c):
    """Production lateral residual on the model grid [nx, ny, nz] for coefficients c [N, n_sigma, 9]."""
    grid = ctx["grid"]
    m = grid.ifft_real(backend._gto_net_density_g(grid, ctx["pf"], c, ctx["sig"])) / grid.volume
    env = ctx["env"][ch]
    r = (env * m).mean(dim=(0, 1)) / torch.clamp(env.mean(dim=(0, 1)), min=1.0e-12)
    return env * m - r[None, None, :] * env


def head_coeffs(ctx, params=None):
    if params is None:
        return head(ctx["feats"], torch.full((ctx["feats"].shape[0],), ctx["q"], device=dev))
    return torch.func.functional_call(head, params, (ctx["feats"], torch.full((ctx["feats"].shape[0],), ctx["q"], device=dev)))


# ------------------------------------------------------------------ labels and points
LABEL = {}


def load_label(sid):
    e = ENT[sid]
    rb = np.asarray(np.load(LBL / e["path_b"], mmap_mode="r"), dtype=np.float64)
    ri = np.asarray(np.load(LBL / e["path_i"], mmap_mode="r"), dtype=np.float64)
    with np.load(LBL / e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return {"b": rb, "i": ri}, lat


def points_for(sid, shape):
    nz, ny, nx = shape
    rng = np.random.default_rng(1000 + sid)
    lin = rng.permutation(nz * ny * nx)[:NFIT + NHOLD]
    iz, rem = np.divmod(lin, ny * nx); iy, ix = np.divmod(rem, nx)
    frac = np.stack([ix / nx, iy / ny, iz / nz], axis=1)
    return lin[:NFIT], torch.as_tensor(frac[:NFIT], device=dev), lin[NFIT:], torch.as_tensor(frac[NFIT:], device=dev)


def prep_labels(sid):
    lab, lat = load_label(sid)
    shape = lab["b"].shape
    lin_f, fr_f, lin_h, fr_h = points_for(sid, shape)
    tgt = {}
    for ch in ("b", "i"):
        lat_part = lab[ch] - lab[ch].mean(axis=(1, 2))[:, None, None]
        flat = lat_part.reshape(-1)
        tgt[ch] = (torch.as_tensor(flat[lin_f], device=dev), torch.as_tensor(flat[lin_h], device=dev))
    LABEL[sid] = dict(shape=shape, lat=lat, fr_f=fr_f, fr_h=fr_h, tgt=tgt)
    return LABEL[sid]


# ------------------------------------------------------------------ full-grid evaluation
def on_dft_grid(Bz, dgrid, shape):
    nz, ny, nx = shape
    out = torch.empty(shape, device=dev)
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    CH = 25
    for z0 in range(0, nz, CH):
        z1 = min(z0 + CH, nz)
        fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
        frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
        v = _interp1_periodic(Bz, frac[:, 2]) + _interp3_periodic(dgrid, frac)
        out[z0:z1] = v.reshape(z1 - z0, ny, nx)
    return out


def metrics(M, D, dV):
    """M, D torch (nz, ny, nx)."""
    d = M - D
    pm, pd = M.mean(dim=(1, 2)), D.mean(dim=(1, 2))
    Ml, Dl = M - pm[:, None, None], D - pd[:, None, None]
    sD = float(D.abs().sum()); sDl = float(Dl.abs().sum())
    return dict(eps_3d=float(d.abs().sum()) / max(sD, 1e-300), eps_lat=float((Ml - Dl).abs().sum()) / max(sDl, 1e-300),
                eps_pa=float((pm - pd).abs().sum()) / max(float(pd.abs().sum()), 1e-300),
                lat_sq=float(((Ml - Dl) ** 2).sum()) / max(float((Dl ** 2).sum()), 1e-300),
                abs_err_e=float(d.abs().sum()) * dV, ref_abs_e=sD * dV, lat_abs_err_e=float((Ml - Dl).abs().sum()) * dV,
                lat_ref_e=sDl * dV, net_err_e=float(d.sum()) * dV, net_ml_e=float(M.sum()) * dV, net_dft_e=float(D.sum()) * dV)


def physics(M, D, lat):
    """Potential and electrostatic self-energy of a charge field (e/A^3) on the DFT grid, ML vs DFT."""
    cell = torch.as_tensor(lat, device=dev)
    dV = abs(float(np.linalg.det(lat))) / M.numel()
    pM = poisson_phi_periodic(M.permute(2, 1, 0).contiguous(), cell)
    pD = poisson_phi_periodic(D.permute(2, 1, 0).contiguous(), cell)
    Mt, Dt = M.permute(2, 1, 0), D.permute(2, 1, 0)
    return dict(phi_rms_err=float(((pM - pD) ** 2).mean().sqrt()), phi_rms_dft=float((pD ** 2).mean().sqrt()),
                eself_ml=0.5 * float((Mt * pM).sum()) * dV, eself_dft=0.5 * float((Dt * pD).sum()) * dV)


def evaluate(sid, dfields, tag, extra_bg=None):
    """dfields: {'b': grid, 'i': grid}; returns and stores per-channel metrics on the full DFT grid + held-out J."""
    ctx, L = CTX[sid], LABEL[sid]
    lab, lat = load_label(sid)
    dV = abs(float(np.linalg.det(lat))) / lab["b"].size
    out, fields = {}, {}
    for ch in ("b", "i"):
        dg = dfields[ch] + (extra_bg[ch] if extra_bg and ch in extra_bg else 0.0)
        M = on_dft_grid(ctx["B"][ch], dg, L["shape"]); D = torch.as_tensor(lab[ch], device=dev)
        out[ch] = metrics(M, D, dV)
        for key, fr, tg in (("J_fit", L["fr_f"], L["tgt"][ch][0]), ("J_hold", L["fr_h"], L["tgt"][ch][1])):
            pr = _interp3_periodic(dg, fr)
            out[ch][key] = float(((pr - tg) ** 2).sum() / torch.clamp((tg ** 2).sum(), min=1e-300))
        fields[ch] = (M, D)
    Mt, Dt = fields["b"][0] + fields["i"][0], fields["b"][1] + fields["i"][1]
    out["t"] = metrics(Mt, Dt, dV)
    out["t"].update(physics(Mt, Dt, lat))
    RESULTS["frames"].setdefault(str(sid), {})[tag] = out
    return fields


def response(sid_c, tag, fc, fn):
    lab, lat = load_label(sid_c)
    dV = abs(float(np.linalg.det(lat))) / lab["b"].size
    o = {}
    for ch in ("b", "i"):
        o[ch] = metrics(fc[ch][0] - fn[ch][0], fc[ch][1] - fn[ch][1], dV)
    Mt = (fc["b"][0] + fc["i"][0]) - (fn["b"][0] + fn["i"][0]); Dt = (fc["b"][1] + fc["i"][1]) - (fn["b"][1] + fn["i"][1])
    o["t"] = metrics(Mt, Dt, dV); o["t"].update(physics(Mt, Dt, lat))
    RESULTS["pairs"].setdefault(str(sid_c), {})[tag] = o


# ------------------------------------------------------------------ LSQR on a LinearOperator
def lsqr_fit(blocks, to_coeffs, x0, label, damp=0.0, tol=TOL, iters=ITER, precond=None, n_probe=64, damp_rel=None, ridge_rel=None):
    """blocks: list of (sid, ch, extra) ; to_coeffs(x, sid, ch) -> [N, n_sigma, 9]; fits the lateral target at the
    fit points of every block: minimise ||F(x0 + dx) - t||^2 + damp^2 ||dx||^2 over dx (F affine in x)."""
    x0_t = torch.as_tensor(x0, device=dev)

    def F_block(x, sid, ch, extra):
        d = dfield(CTX[sid], ch, to_coeffs(x, sid, ch))
        v = _interp3_periodic(d, LABEL[sid]["fr_f"])
        if extra is not None:
            v = v + extra
        return v
    with torch.no_grad():
        F0 = [F_block(x0_t, *b) for b in blocks]
    tg = [LABEL[b[0]]["tgt"][b[1]][0] for b in blocks]
    r0 = torch.cat([t - f for t, f in zip(tg, F0)])
    sizes = [int(t.numel()) for t in tg]
    ncall = {"mv": 0, "rmv": 0}

    def mv(v):
        ncall["mv"] += 1
        v_t = torch.as_tensor(np.asarray(v, dtype=np.float64).ravel(), device=dev)
        with torch.no_grad():
            return torch.cat([F_block(x0_t + v_t, *b) - f for b, f in zip(blocks, F0)]).cpu().numpy()

    def rmv(u):
        ncall["rmv"] += 1
        u_t = torch.as_tensor(np.asarray(u, dtype=np.float64).ravel(), device=dev)
        g = torch.zeros_like(x0_t); off = 0
        for b, n in zip(blocks, sizes):
            x = x0_t.clone().requires_grad_(True)
            y = F_block(x, *b)
            g += torch.autograd.grad(y, x, grad_outputs=u_t[off:off + n])[0]
            off += n
        return g.cpu().numpy()
    nrow, ncol = sum(sizes), x0_t.numel()
    t0 = time.time()
    colscale = np.ones(ncol)
    ridge = 0.0
    if ridge_rel is not None:
        # ridge on the ORIGINAL coefficient change dx: augmented system [A; ridge I] dx = [r0; 0],
        # ridge = ridge_rel * ||A||_2 (power-iteration estimate of the unscaled operator norm)
        v = np.random.default_rng(5).standard_normal(ncol); v /= np.linalg.norm(v)
        for _ in range(8):
            w = rmv(mv(v)); v = w / np.linalg.norm(w)
        ridge = ridge_rel * float(np.sqrt(np.linalg.norm(rmv(mv(v)))))
    if precond == "jacobi":
        # Hutchinson: E_g[(A^T g)_j^2] = ||A e_j||^2 for g ~ N(0, I); columns scaled to unit (estimated) norm,
        # columns with an estimated norm below 1e-6 of the largest are kept but floored (they barely touch the fit)
        rng = np.random.default_rng(7)
        acc = np.zeros(ncol)
        for _ in range(n_probe):
            acc += rmv(rng.standard_normal(nrow)) ** 2
        cn = np.sqrt(acc / n_probe)
        # augmented column norm sqrt(||A e_j||^2 + ridge^2): with a ridge no column is left near zero
        cn_aug = np.sqrt(cn ** 2 + ridge ** 2)
        colscale = 1.0 / np.maximum(cn_aug, 1e-6 * cn_aug.max())
    if ridge > 0.0:
        opS = LinearOperator((nrow + ncol, ncol),
                             matvec=lambda y: np.concatenate([mv(np.asarray(y).ravel() * colscale), ridge * np.asarray(y).ravel() * colscale]),
                             rmatvec=lambda u: (rmv(np.asarray(u)[:nrow]) + ridge * np.asarray(u)[nrow:]) * colscale, dtype=np.float64)
        rhs = np.concatenate([r0.cpu().numpy(), np.zeros(ncol)])
    else:
        opS = LinearOperator((nrow, ncol), matvec=lambda y: mv(np.asarray(y).ravel() * colscale),
                             rmatvec=lambda u: rmv(u) * colscale, dtype=np.float64)
        rhs = r0.cpu().numpy()
    if damp_rel is not None:
        # damp relative to the (scaled) operator norm estimate from a short power iteration
        v = np.random.default_rng(3).standard_normal(ncol); v /= np.linalg.norm(v)
        for _ in range(8):
            w = opS.rmatvec(opS.matvec(v)); v = w / np.linalg.norm(w)
        damp = damp_rel * float(np.sqrt(np.linalg.norm(opS.rmatvec(opS.matvec(v)))))
    y, istop, itn, r1norm, r2norm, anorm, acond, arnorm, ynorm = lsqr(opS, rhs, damp=damp, atol=tol, btol=tol,
                                                                      iter_lim=iters)[:9]
    dx = y * colscale; xnorm = float(np.linalg.norm(x0 + dx))
    if ridge > 0.0:
        # r1norm of the augmented system includes the ridge rows; report the data residual itself
        r1norm = float(np.linalg.norm(mv(dx) - r0.cpu().numpy()))
    rec = dict(label=label, blocks=[[int(b[0]), b[1]] for b in blocks], n_unknown=int(x0_t.numel()), n_rows=int(sum(sizes)),
               precond=precond, damp_rel=damp_rel, damp=damp, ridge_rel=ridge_rel, ridge=ridge, tol=tol, iter_lim=iters, istop=int(istop), itn=int(itn), r0_norm=float(r0.norm()),
               r1norm=float(r1norm), rel_resid=float(r1norm) / max(float(r0.norm()), 1e-300), anorm=float(anorm),
               acond=float(acond), arnorm=float(arnorm), rel_optimality=float(arnorm) / max(float(anorm) * float(r1norm), 1e-300),
               xnorm=float(xnorm), dx_norm=float(np.linalg.norm(dx)), x0_norm=float(x0_t.norm()), seconds=time.time() - t0,
               n_matvec=ncall["mv"], n_rmatvec=ncall["rmv"],
               target_norm=float(torch.cat(tg).norm()))
    RESULTS["fits"].append(rec); save()
    print(f"[{elapsed()}] LSQR {label}: istop {istop} itn {itn} rel_resid {rec['rel_resid']:.4f} (J = {(float(r1norm) / rec['target_norm']) ** 2:.4f}) "
          f"rel_opt {rec['rel_optimality']:.2e} acond {acond:.2e} |dx| {rec['dx_norm']:.3e} |x0| {rec['x0_norm']:.3e} "
          f"{rec['seconds']:.0f}s mv {ncall['mv']} rmv {ncall['rmv']}", flush=True)
    return x0 + dx, rec


def coeff_vec(ctx, ch):
    k = 0 if ch == "b" else 1
    return ctx["c0"][:, k].reshape(-1).cpu().numpy().copy()


def free_to_coeffs(shape):
    return lambda x, sid, ch: x.view(shape)


# ------------------------------------------------------------------ grid round trip
def roundtrip(sid):
    ctx = CTX[sid]; lab, lat = load_label(sid)
    gshape = tuple(ctx["env"]["b"].shape)  # (nx, ny, nz) model grid
    nx, ny, nz = gshape
    ii, jj, kk = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev), torch.arange(nz, device=dev), indexing="ij")
    fm = torch.stack([ii.reshape(-1) / nx, jj.reshape(-1) / ny, kk.reshape(-1) / nz], dim=1).double()
    dV = abs(float(np.linalg.det(lat))) / lab["b"].size
    o = {}
    for ch in ("b", "i"):
        D = torch.as_tensor(lab[ch], device=dev)
        Dg = D.permute(2, 1, 0).contiguous()                       # [nxD, nyD, nzD]
        Fm = _interp3_periodic(Dg, fm).reshape(gshape)             # DFT sampled at model grid points
        back = on_dft_grid(torch.zeros(8, device=dev), Fm, lab[ch].shape)
        m = metrics(back, D, dV)
        pd, pb = D.mean(dim=(1, 2)), back.mean(dim=(1, 2))
        zz = np.arange(lab[ch].shape[0]) * lat[2, 2] / lab[ch].shape[0]
        def fwhm(p):
            p = p.cpu().numpy(); k = int(np.argmax(p)); h = p[k] / 2
            lo = k
            while lo > 0 and p[lo] > h: lo -= 1
            hi = k
            while hi < len(p) - 1 and p[hi] > h: hi += 1
            return float(zz[k]), float(p[k]), float(zz[hi] - zz[lo])
        zk, pk, wk = fwhm(pd); zk2, pk2, wk2 = fwhm(pb)
        m.update(peak_z_dft=zk, peak_dft=pk, fwhm_dft=wk, peak_z_rt=zk2, peak_rt=pk2, fwhm_rt=wk2,
                 max3d_dft=float(D.max()), max3d_rt=float(back.max()), min3d_dft=float(D.min()), min3d_rt=float(back.min()))
        o[ch] = m
    RESULTS["roundtrip"][str(sid)] = o
    print(f"[{elapsed()}] roundtrip sid {sid}: " + "  ".join(
        f"{ch}: eps3d {o[ch]['eps_3d']:.4f} lat {o[ch]['eps_lat']:.4f} pa {o[ch]['eps_pa']:.4f} net {o[ch]['net_err_e']:+.2e} "
        f"peak {o[ch]['peak_dft']:.4e}->{o[ch]['peak_rt']:.4e} fwhm {o[ch]['fwhm_dft']:.2f}->{o[ch]['fwhm_rt']:.2f} "
        f"max3d {o[ch]['max3d_dft']:.4f}->{o[ch]['max3d_rt']:.4f}" for ch in ("b", "i")), flush=True)


# ================================================================== run
def prep(sid):
    t0 = time.time()
    ctx = capture(sid); prep_labels(sid)
    rep = {}
    for k, ch in ((0, "b"), (1, "i")):
        with torch.no_grad():
            d = dfield(ctx, ch, ctx["c0"][:, k])
        ref = ctx["dsup"][ch]
        rep[ch] = float((d - ref).abs().max() / torch.clamp(ref.abs().max(), min=1e-300))
    with torch.no_grad():
        ch_ = head_coeffs(ctx)
    rep["head"] = float((ch_ - ctx["c0"]).abs().max() / torch.clamp(ctx["c0"].abs().max(), min=1e-300))
    ctx["dsup"] = None                                  # only needed for the reproduction check; frees 48 MB per frame
    L = LABEL[sid]; share = {}
    for ch in ("b", "i"):
        e_at = _interp3_periodic(ctx["env"][ch], L["fr_f"]); t = L["tgt"][ch][0]; tt = float((t ** 2).sum())
        share[ch] = {f"{thr:g}": float((t[e_at < thr] ** 2).sum()) / max(tt, 1e-300) for thr in (1e-4, 1e-3, 1e-2, 1e-1)}
    rep["target_share_where_env_below"] = share
    RESULTS["frames"].setdefault(str(sid), {})["repro"] = dict(rep, nsolve=ctx["nsolve"], q=ctx["q"], split=ctx["split"],
                                                                n_atoms=int(ctx["pf"].shape[0]), grid=list(ctx["env"]["b"].shape),
                                                                nz_1d=int(ctx["B"]["b"].numel()))
    print(f"[{elapsed()}] sid {sid} ({ctx['split']}, q {ctx['q']:+.3f}, {ctx['pf'].shape[0]} atoms, grid {tuple(ctx['env']['b'].shape)}, "
          f"1-D nz {ctx['B']['b'].numel()}, solves {ctx['nsolve']}): reproduction max rel diff bound {rep['b']:.2e} ion {rep['i']:.2e} "
          f"head {rep['head']:.2e}  ({time.time() - t0:.0f}s); lateral-target share where env < 1e-3 / 1e-2: bound "
          f"{share['b']['0.001']:.3f} / {share['b']['0.01']:.3f}, ion {share['i']['0.001']:.3f} / {share['i']['0.01']:.3f}", flush=True)
    return ctx


def d0(sid):
    ctx = CTX[sid]
    with torch.no_grad():
        return {"b": dfield(ctx, "b", ctx["c0"][:, 0]), "i": dfield(ctx, "i", ctx["c0"][:, 1])}


def zero_like(sid):
    return torch.zeros_like(CTX[sid]["env"]["i"])


fit_sids = [s for k in FIT_PAIRS for s in (k, k + 600)]
for sid in fit_sids:
    prep(sid)
save()

# timing of one operator application (forward, and forward+backward) on the first frame
s0 = fit_sids[0]; c_t = CTX[s0]["c0"][:, 0].clone()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(3):
    with torch.no_grad():
        _ = _interp3_periodic(dfield(CTX[s0], "b", c_t), LABEL[s0]["fr_f"])
torch.cuda.synchronize(); t_fwd = (time.time() - t0) / 3
t0 = time.time()
for _ in range(3):
    x = c_t.clone().requires_grad_(True)
    y = _interp3_periodic(dfield(CTX[s0], "b", x), LABEL[s0]["fr_f"])
    torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
torch.cuda.synchronize(); t_bwd = (time.time() - t0) / 3
RESULTS["meta"]["t_forward_s"] = t_fwd; RESULTS["meta"]["t_fwd_bwd_s"] = t_bwd
print(f"[{elapsed()}] operator timing: forward {t_fwd:.3f}s, forward+backward {t_bwd:.3f}s; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

FIELDS = {}
for sid in fit_sids:
    FIELDS[(sid, "V0")] = evaluate(sid, d0(sid), "V0")
    if "rt" in STAGES:
        roundtrip(sid)
for k in FIT_PAIRS:
    response(k, "V0", FIELDS[(k, "V0")], FIELDS[(k + 600, "V0")])
save()

if MODE == "ridge":
    # ridge path on the ORIGINAL coefficients (preconditioned solver), bound and ionic channel of the first fit frame
    ctx = CTX[s0]; shp = ctx["c0"][:, 0].shape
    RPLAN = {"b": (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5), "i": (1e-2, 1e-3, 1e-4)}
    for ch in ("b", "i"):
        for rr in RPLAN[ch]:
            lab = f"RIDGE {ch} sid {s0} ridge_rel={rr}"
            x, rec = lsqr_fit([(s0, ch, None)], free_to_coeffs(shp), coeff_vec(ctx, ch), lab, iters=4000, precond="jacobi", ridge_rel=rr)
            with torch.no_grad():
                dnew = dfield(ctx, ch, torch.as_tensor(x, device=dev).view(shp))
            base = d0(s0); base[ch] = dnew
            evaluate(s0, base, lab)
            o = RESULTS["frames"][str(s0)][lab]
            print(f"[{elapsed()}]    -> {ch} ridge_rel {rr:g}: |x| {np.linalg.norm(x):.3e} (|x0| {np.linalg.norm(coeff_vec(ctx, ch)):.3e}) J_fit {o[ch]['J_fit']:.4f} "
                  f"J_hold {o[ch]['J_hold']:.4f} lat L1 {o[ch]['eps_lat']:.4f}; total phi rms err {o['t']['phi_rms_err']:.4f} Eself {o['t']['eself_ml']:.4f} "
                  f"(DFT {o['t']['eself_dft']:.4f})", flush=True)
            save()
    print(f"[{elapsed()}] done", flush=True)
    sys.exit(0)

if MODE == "conv":
    # convergence / preconditioning / regularisation study on the first fit frame, bound and ionic channel
    ctx = CTX[s0]; shp = ctx["c0"][:, 0].shape
    PLAN = {"b": [("none", None, 4000)] + [("jacobi", None, it) for it in (1000, 2000, 4000, 8000)] + [("jacobi", dr, 8000) for dr in (1e-3, 1e-2)],
            "i": [("jacobi", None, it) for it in (2000, 8000)]}
    for ch in ("b", "i"):
        for pc, dr, it in PLAN[ch]:
            lab = f"CONV {ch} sid {s0} precond={pc} damp_rel={dr} iter={it}"
            x, rec = lsqr_fit([(s0, ch, None)], free_to_coeffs(shp), coeff_vec(ctx, ch), lab, iters=it, precond=pc, damp_rel=dr)
            with torch.no_grad():
                dnew = dfield(ctx, ch, torch.as_tensor(x, device=dev).view(shp))
            base = d0(s0); base[ch] = dnew
            evaluate(s0, base, lab)
            o = RESULTS["frames"][str(s0)][lab]
            print(f"[{elapsed()}]    -> {ch}: J_fit {o[ch]['J_fit']:.4f} J_hold {o[ch]['J_hold']:.4f} lat L1 {o[ch]['eps_lat']:.4f} lat sq {o[ch]['lat_sq']:.4f}; "
                  f"total phi rms err {o['t']['phi_rms_err']:.4f} Eself {o['t']['eself_ml']:.4f} (DFT {o['t']['eself_dft']:.4f})", flush=True)
            save()
    print(f"[{elapsed()}] done", flush=True)
    sys.exit(0)

DV1 = {}
FITKW = dict(precond="jacobi", iters=ITER)


def fit_free(sid, ch, rr, x0=None, label=None, extra=None):
    ctx = CTX[sid]; shp = ctx["c0"][:, 0].shape
    x, rec = lsqr_fit([(sid, ch, extra)], free_to_coeffs(shp), coeff_vec(ctx, ch) if x0 is None else x0,
                      label or f"V1 free {ch} sid {sid} ridge {rr:g}", ridge_rel=rr, **FITKW)
    with torch.no_grad():
        return dfield(ctx, ch, torch.as_tensor(x, device=dev).view(shp)), rec


if "free" in STAGES:
    # ionic: the gated freedom of the current model = charged frames only; neutral ungated reported apart (V1u)
    ION_FIT = {}
    for sid in fit_sids:
        if abs(CTX[sid]["q"]) > 1e-6:
            for ri in ION_RIDGES:
                ION_FIT[(sid, ri)] = fit_free(sid, "i", ri, label=f"V1 free ion (gated) sid {sid} ridge {ri:g}")[0]
        else:
            for ri in ION_RIDGES:
                du, _ = fit_free(sid, "i", ri, x0=np.zeros_like(coeff_vec(CTX[sid], "i")),
                                 label=f"V1u free ion UNGATED (neutral; not a capability of the current model) sid {sid} ridge {ri:g}")
                evaluate(sid, {"b": d0(sid)["b"], "i": du}, f"V1u@{ri:g}")
    ri_ref = ION_RIDGES[-1]
    for rr in RIDGES:
        tag = f"V1@{rr:g}"
        for sid in fit_sids:
            db, _ = fit_free(sid, "b", rr)
            di = ION_FIT[(sid, ri_ref)] if abs(CTX[sid]["q"]) > 1e-6 else d0(sid)["i"]
            DV1[(sid, rr)] = {"b": db, "i": di}
            FIELDS[(sid, tag)] = evaluate(sid, DV1[(sid, rr)], tag)
        for k in FIT_PAIRS:
            response(k, tag, FIELDS[(k, tag)], FIELDS[(k + 600, tag)])
            print(f"[{elapsed()}] {tag} pair {k}: bound lat L1 charged {RESULTS['frames'][str(k)][tag]['b']['eps_lat']:.3f} neutral "
                  f"{RESULTS['frames'][str(k + 600)][tag]['b']['eps_lat']:.3f} response {RESULTS['pairs'][str(k)][tag]['b']['eps_lat']:.3f}", flush=True)
        save()

if "pair" in STAGES and DV1:
    for rr in RIDGES:
        tag = f"V2@{rr:g}"
        for k in FIT_PAIRS:
            shp = CTX[k]["c0"][:, 0].shape
            xb, _ = lsqr_fit([(k, "b", None), (k + 600, "b", None)], free_to_coeffs(shp), coeff_vec(CTX[k], "b"),
                             f"V2 pair-shared bound {k}/{k + 600} ridge {rr:g}", ridge_rel=rr, **FITKW)
            cb = torch.as_tensor(xb, device=dev).view(shp)
            f2 = {}
            for sid in (k, k + 600):
                with torch.no_grad():
                    db = dfield(CTX[sid], "b", cb)
                f2[sid] = evaluate(sid, {"b": db, "i": DV1[(sid, rr)]["i"]}, tag)   # ionic part as in V1 -> isolates the bound channel
            response(k, tag, f2[k], f2[k + 600])
            print(f"[{elapsed()}] {tag} pair {k}: bound lat L1 charged {RESULTS['frames'][str(k)][tag]['b']['eps_lat']:.3f} neutral "
                  f"{RESULTS['frames'][str(k + 600)][tag]['b']['eps_lat']:.3f} response {RESULTS['pairs'][str(k)][tag]['b']['eps_lat']:.3f}", flush=True)
        save()


# ------------------------------------------------------------------ V4: ionic background candidate (charged frames)
def ion_bg_lateral(sid):
    """B_i(z) * (alpha - 1) on the model grid, alpha = env_i / <env_i>_xy; plane means of the full field unchanged.
    Planes where <env_i> is ~0 but B_i is not are REPORTED (no floor, no silent fallback)."""
    ctx = CTX[sid]; env = ctx["env"]["i"]; nx, ny, nz = env.shape
    Bm = _interp1_periodic(ctx["B"]["i"], torch.arange(nz, device=dev, dtype=torch.float64) / nz)
    em = env.mean(dim=(0, 1))
    bad = (em < 1e-6) & (Bm.abs() > 1e-9)
    info = dict(n_bad_planes=int(bad.sum()), max_absB_bad=float(Bm[bad].abs().max()) if bool(bad.any()) else 0.0,
                min_env_mean_where_B=float(em[Bm.abs() > 1e-9].min()) if bool((Bm.abs() > 1e-9).any()) else None,
                max_absB=float(Bm.abs().max()))
    alpha = torch.where(em[None, None, :] > 1e-6, env / torch.clamp(em, min=1e-300)[None, None, :], torch.ones_like(env))
    return Bm[None, None, :] * (alpha - 1.0), info


if "bg" in STAGES:
    for k in FIT_PAIRS:
        Lb, info = ion_bg_lateral(k)
        RESULTS["frames"][str(k)]["bg_info"] = info
        print(f"[{elapsed()}] V4 background sid {k}: {info}", flush=True)
        base = d0(k)
        evaluate(k, {"b": base["b"], "i": base["i"] + Lb}, "V4c0")
        extra = _interp3_periodic(Lb, LABEL[k]["fr_f"])
        for ri in ION_RIDGES:
            di, _ = fit_free(k, "i", ri, extra=extra, label=f"V4 ion background candidate, refit sid {k} ridge {ri:g}")
            evaluate(k, {"b": base["b"], "i": di + Lb}, f"V4@{ri:g}")
            print(f"[{elapsed()}] V4 sid {k}: ion lat L1 V0 {RESULTS['frames'][str(k)]['V0']['i']['eps_lat']:.3f} V4c0 "
                  f"{RESULTS['frames'][str(k)]['V4c0']['i']['eps_lat']:.3f} V4@{ri:g} {RESULTS['frames'][str(k)][f'V4@{ri:g}']['i']['eps_lat']:.3f}", flush=True)
    save()

# ------------------------------------------------------------------ V3: the real shared head (node features fixed)
if "head" in STAGES and HEAD_PAIRS:
    names = [n for n, _ in head.named_parameters()]
    shapes = [p.shape for _, p in head.named_parameters()]
    sizes_p = [p.numel() for _, p in head.named_parameters()]
    x0h = torch.cat([p.detach().reshape(-1) for _, p in head.named_parameters()]).cpu().numpy()

    def unflat(x):
        out, off = {}, 0
        for n, shp_, sz in zip(names, shapes, sizes_p):
            out[n] = x[off:off + sz].view(shp_); off += sz
        return out

    def head_to_coeffs(x, sid, ch):
        return head_coeffs(CTX[sid], unflat(x))[:, 0 if ch == "b" else 1]
    hsids = [s for k in HEAD_PAIRS for s in (k, k + 600)]
    for sid in hsids:
        if sid not in CTX:
            prep(sid)
            FIELDS[(sid, "V0")] = evaluate(sid, d0(sid), "V0")
    for k in HEAD_PAIRS:
        if str(k) not in RESULTS["pairs"] or "V0" not in RESULTS["pairs"][str(k)]:
            response(k, "V0", FIELDS[(k, "V0")], FIELDS[(k + 600, "V0")])
    print(f"[{elapsed()}] V3 head fit on {len(hsids)} training frames; head parameters {int(x0h.size)} ({list(zip(names, sizes_p))})", flush=True)
    xb, recb = lsqr_fit([(s, "b", None) for s in hsids], head_to_coeffs, x0h, f"V3 shared head, bound, {len(hsids)} train frames",
                        precond="jacobi", ridge_rel=HEAD_RIDGE, iters=HEAD_ITER)
    xi, reci = lsqr_fit([(s, "i", None) for s in hsids if abs(CTX[s]["q"]) > 1e-6], head_to_coeffs, x0h,
                        f"V3 shared head, ion (gated), {len(HEAD_PAIRS)} charged train frames", precond="jacobi", ridge_rel=HEAD_RIDGE,
                        iters=HEAD_ITER)
    xh = x0h + (xb - x0h) + (xi - x0h)
    # the two objectives touch disjoint output columns; check that the combined vector reproduces both fits
    xh_t = torch.as_tensor(xh, device=dev)
    np.save(OUT / "head_params_V3.npy", xh)
    RESULTS["meta"]["head_V3"] = dict(n_params=int(x0h.size), bound_dx=float(np.linalg.norm(xb - x0h)), ion_dx=float(np.linalg.norm(xi - x0h)),
                                      overlap=float(np.abs((xb - x0h) * (xi - x0h)).sum()))

    def v3_fields(sid):
        with torch.no_grad():
            c = head_coeffs(CTX[sid], unflat(xh_t))
            return {"b": dfield(CTX[sid], "b", c[:, 0]), "i": dfield(CTX[sid], "i", c[:, 1])}
    for sid in hsids:
        FIELDS[(sid, "V3")] = evaluate(sid, v3_fields(sid), "V3")
    for k in HEAD_PAIRS:
        response(k, "V3", FIELDS[(k, "V3")], FIELDS[(k + 600, "V3")])
    save()
    # held-out frames: validation pairs (and the test pair for display only); V0 and V3, nothing fitted on them
    for k in VAL_PAIRS + TEST_PAIRS:
        for sid in (k, k + 600):
            if sid not in CTX:
                prep(sid)
            FIELDS[(sid, "V0")] = evaluate(sid, d0(sid), "V0")
            FIELDS[(sid, "V3")] = evaluate(sid, v3_fields(sid), "V3")
            if "rt" in STAGES:
                roundtrip(sid)
            # free the big per-frame tensors of held-out frames once evaluated
            CTX[sid]["dsup"] = None
        response(k, "V0", FIELDS[(k, "V0")], FIELDS[(k + 600, "V0")]); response(k, "V3", FIELDS[(k, "V3")], FIELDS[(k + 600, "V3")])
        for sid in (k, k + 600):
            FIELDS.pop((sid, "V0")); FIELDS.pop((sid, "V3"))
        save()
        print(f"[{elapsed()}] held-out pair {k}: V0 bound lat {RESULTS['frames'][str(k)]['V0']['b']['eps_lat']:.3f} -> V3 "
              f"{RESULTS['frames'][str(k)]['V3']['b']['eps_lat']:.3f}; response bound lat {RESULTS['pairs'][str(k)]['V0']['b']['eps_lat']:.3f} -> "
              f"{RESULTS['pairs'][str(k)]['V3']['b']['eps_lat']:.3f}", flush=True)
print(f"[{elapsed()}] done", flush=True)
