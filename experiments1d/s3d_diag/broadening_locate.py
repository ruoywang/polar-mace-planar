"""Where does the ML bound-charge layer get broad? (reviewer 2026-10-02; diagnosis only, no training.)
Along the production construction, on the DFT grid (500 x 168 x 168, dz 0.09 A):
    DFT_b                      the label
    DFT_b round trip           DFT sampled on the model grid (100 x 100 x 300) and trilinearly back: what that grid carries
    1-D part    B_b(z)         broadcast profile of the final solve (plane-average width by construction)
    envelope    env_b          |grad s_diel| / max on the model grid, trilinear to the DFT grid: the layer the residual may live in
    GTO field   m_b            sum of atom-centred Gaussians (sigmas 0.5 / 1 / 2 A) with the head coefficients
    raw         env_b * m_b    before the per-plane projection
    residual    d_b            after the projection (= what the loss scores)
    ML_b        B_b + d_b      the full field
Per column (ix, iy) inside the interface window (plane-averaged DFT |rho_b| > 1 % of its maximum): the z-thickness holding
half of the POSITIVE charge of that column (same metric as the 122-722 slice), the FWHM of the main positive peak, and the
peak position; medians over columns. Plane-averaged FWHM of DFT_b, ML_b, B_b. Signed region integrals over DFT > 0 and
DFT < 0 (bound; charged, neutral, response). Ionic lateral position: per column, z of max |lateral ion| (DFT, ML) relative to
the z where env_i rises through 0.5 (ion-accessible edge). Pairs from KIT_PAIRS (default "28 94 122").
Usage: python broadening_locate.py <out.json>   (cwd with ./data, ./cal1_train.json)
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

import numpy as np
import torch
from ase.io import read

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.solvent3d import _interp1_periodic, _interp3_periodic, normalized_gradient_envelope

OUT = sys.argv[1] if len(sys.argv) > 1 else "broadening_locate.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "28 94 122").replace("+", " ").split()]
torch.set_default_dtype(torch.float64)
dev = torch.device("cuda:0")
model = torch.load(MODEL, map_location=dev).to(dev); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend()
man = json.load(open("data/solvent3d_grid_manifest.json")); ENT = {int(k): v for k, v in man["entries"].items()}
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()


def label(sid):
    e = ENT[sid]
    rb = torch.as_tensor(np.asarray(np.load("data/" + e["path_b"], mmap_mode="r"), dtype=np.float64), device=dev)
    ri = torch.as_tensor(np.asarray(np.load("data/" + e["path_i"], mmap_mode="r"), dtype=np.float64), device=dev)
    with np.load("data/" + e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return rb, ri, lat


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
        cav = grid._solv3d_cavity
        hold.update(res=res, kw=kw, grid=grid, cell_np=cell_np, cav=(cav[0].clone(), cav[1].clone()))
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
    c0 = kw["s3d_coeffs"].detach(); sig = kw["s3d_sigmas"]
    out = dict(q=float(kw["total_charge"]), B_b=res["rho_bound_z"].detach().double(), B_i=res["rho_ion_z"].detach().double(),
               env_b=normalized_gradient_envelope(s_diel3, cell64), env_i=torch.clamp(s_ion3, 0.0, 1.0), s_ion=s_ion3)
    with torch.no_grad():
        for ch, k in (("b", 0), ("i", 1)):
            m = grid.ifft_real(backend._gto_net_density_g(grid, pf, c0[:, k], sig)) / grid.volume
            env = out["env_" + ch]; raw = env * m
            r = raw.mean(dim=(0, 1)) / torch.clamp(env.mean(dim=(0, 1)), min=1.0e-12)
            out["m_" + ch] = m; out["raw_" + ch] = raw; out["d_" + ch] = raw - r[None, None, :] * env
    assert float((out["d_b"] - res["s3d_obs"]["d_sup_b"]).abs().max()) == 0.0
    return out


def to_dft(F3, shape):
    """model grid [nx,ny,nz] -> DFT grid (nz,ny,nx), trilinear periodic (the production interpolation)."""
    nz, ny, nx = shape
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    M = torch.empty(shape, device=dev)
    for z0 in range(0, nz, 25):
        z1 = min(z0 + 25, nz)
        fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
        frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
        M[z0:z1] = _interp3_periodic(F3, frac).reshape(z1 - z0, ny, nx)
    return M


def bcast(Bz, shape):
    nz = shape[0]
    return _interp1_periodic(Bz, torch.arange(nz, device=dev, dtype=torch.float64) / nz)[:, None, None].expand(shape).contiguous()


def roundtrip(D, gshape):
    nx, ny, nz = gshape
    ii, jj, kk = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev), torch.arange(nz, device=dev), indexing="ij")
    fm = torch.stack([ii.reshape(-1) / nx, jj.reshape(-1) / ny, kk.reshape(-1) / nz], dim=1).double()
    Fm = _interp3_periodic(D.permute(2, 1, 0).contiguous(), fm).reshape(gshape)
    return to_dft(Fm, tuple(D.shape))


def column_stats(F, win, dz):
    """F (nz, ny, nx); columns along z inside the window mask win (nz,). Positive part: half-charge thickness, FWHM of the
    main positive peak, peak z (index). Medians over columns with positive charge > 1 % of the strongest column."""
    sub = F[win]                                      # (nw, ny, nx)
    pos = torch.clamp(sub, min=0.0)
    q = pos.sum(dim=0)                                # (ny, nx)
    keep = q > 0.01 * q.max()
    srt, _ = torch.sort(pos.reshape(pos.shape[0], -1), dim=0, descending=True)
    cum = torch.cumsum(srt, dim=0); half = 0.5 * cum[-1]
    n_half = (cum < half[None, :]).sum(dim=0) + 1
    thick = (n_half.double() * dz).reshape(q.shape)[keep]
    pk = sub.max(dim=0).values; kpk = sub.argmax(dim=0)
    above = sub >= 0.5 * pk[None]
    # FWHM: contiguous run around the peak
    nw = sub.shape[0]; idx = torch.arange(nw, device=dev)[:, None, None]
    lo = torch.where(above & (idx <= kpk[None]), idx, torch.full_like(idx, nw)).min(dim=0).values
    hi = torch.where(above & (idx >= kpk[None]), idx, torch.full_like(idx, -1)).max(dim=0).values
    # restrict to the contiguous run: walk from the peak outward
    fw = torch.zeros_like(pk)
    for s in (-1, 1):
        k = kpk.clone(); run = torch.ones_like(keep)
        for step in range(1, nw):
            kk = kpk + s * step
            ok = (kk >= 0) & (kk < nw)
            val = sub[kk.clamp(0, nw - 1), torch.arange(sub.shape[1], device=dev)[:, None], torch.arange(sub.shape[2], device=dev)[None, :]]
            run = run & ok & (val >= 0.5 * pk)
            fw = fw + run.double()
    fwhm = ((fw + 1.0) * dz)[keep]
    zpk = (kpk.double() * dz)[keep]
    return dict(half_thick=float(thick.median()), half_thick_q25=float(thick.quantile(0.25)), half_thick_q75=float(thick.quantile(0.75)),
                fwhm=float(fwhm.median()), zpeak_med=float(zpk.median()), zpeak_std=float(zpk.std()), n_cols=int(keep.sum()),
                pos_charge_share=float(pos.sum() / torch.clamp(sub.abs().sum(), min=1e-300)))


def plane_fwhm(F, dz):
    p = F.mean(dim=(1, 2)); k = int(p.argmax()); h = 0.5 * p[k]
    lo = k
    while lo > 0 and p[lo] > h:
        lo -= 1
    hi = k
    while hi < p.numel() - 1 and p[hi] > h:
        hi += 1
    return float((hi - lo) * dz), float(k * dz)


def region_integrals(M, D, dV):
    out = {}
    for nm, msk in (("pos", D > 0), ("neg", D < 0)):
        out[nm] = dict(dft=float(D[msk].sum()) * dV, ml=float(M[msk].sum()) * dV, signed=float((M[msk] - D[msk]).sum()) * dV,
                       abs=float((M[msk] - D[msk]).abs().sum()) * dV, ratio=float(M[msk].sum() / D[msk].sum()))
    return out


RES = []
for k in PAIRS:
    rec = dict(pair=k, split=ATOMS[k].info["_split"], frames={})
    Db, Di, lat = label(k); Dbn, Din, _ = label(k + 600)
    shape = tuple(Db.shape); nz = shape[0]; dz = lat[2, 2] / nz; dV = abs(float(np.linalg.det(lat))) / Db.numel()
    C = {k: capture(k), k + 600: capture(k + 600)}
    gshape = tuple(C[k]["env_b"].shape)
    ML = {}
    for sid, D in ((k, Db), (k + 600, Dbn)):
        c = C[sid]
        fields = {"DFT": D, "DFT round trip": roundtrip(D, gshape), "1-D part B(z)": bcast(c["B_b"], shape),
                  "envelope env_b": to_dft(c["env_b"], shape), "GTO field m_b": to_dft(c["m_b"], shape), "raw env*m": to_dft(c["raw_b"], shape),
                  "residual d_b": to_dft(c["d_b"], shape)}
        fields["ML"] = fields["1-D part B(z)"] + fields["residual d_b"]; ML[sid] = fields["ML"]
        pz = D.abs().mean(dim=(1, 2)); win = pz > 0.01 * pz.max()
        fr = dict(q=c["q"], window_A=[float(win.nonzero().min()) * dz, float(win.nonzero().max()) * dz], columns={}, plane_fwhm={})
        for nm, F in fields.items():
            fr["columns"][nm] = column_stats(F, win, dz)
        for nm in ("DFT", "DFT round trip", "ML", "1-D part B(z)"):
            fr["plane_fwhm"][nm] = plane_fwhm(fields[nm], dz)
        # how much does m_b vary across the envelope layer? (ratio of |m| at the envelope maximum column-wise to its column mean in the window)
        env = fields["envelope env_b"][win]; m = fields["GTO field m_b"][win]
        kpk = env.argmax(dim=0); mpk = m.gather(0, kpk[None])[0]
        fr["m_at_env_peak_over_m_window_mean"] = float((mpk.abs() / torch.clamp(m.abs().mean(dim=0), min=1e-300)).median())
        fr["region"] = region_integrals(fields["ML"], D, dV)
        # ionic lateral position vs the ion-accessible edge (charged frames)
        if abs(c["q"]) > 1e-6:
            envi = to_dft(c["env_i"], shape); Dil = Di - Di.mean(dim=(1, 2), keepdim=True)
            Mi = bcast(c["B_i"], shape) + to_dft(c["d_i"], shape); Mil = Mi - Mi.mean(dim=(1, 2), keepdim=True)
            edge = (envi >= 0.5).double().argmax(dim=0).double() * dz           # first z where env_i >= 0.5, per column
            zD = Dil.abs().argmax(dim=0).double() * dz; zM = Mil.abs().argmax(dim=0).double() * dz
            fr["ion_lateral"] = dict(edge_z_med=float(edge.median()), dft_peak_minus_edge=float((zD - edge).median()),
                                     ml_peak_minus_edge=float((zM - edge).median()), ml_minus_dft=float((zM - zD).median()),
                                     ml_minus_dft_q25=float((zM - zD).quantile(0.25)), ml_minus_dft_q75=float((zM - zD).quantile(0.75)))
        rec["frames"][str(sid)] = fr
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {c['q']:+.3f}) window {fr['window_A'][0]:.1f}-{fr['window_A'][1]:.1f} A; per-column half-charge thickness (A): "
              + ", ".join(f"{nm} {fr['columns'][nm]['half_thick']:.2f}" for nm in fields) + f"; FWHM: " + ", ".join(f"{nm} {fr['columns'][nm]['fwhm']:.2f}" for nm in fields)
              + f"; plane FWHM DFT {fr['plane_fwhm']['DFT'][0]:.2f} ML {fr['plane_fwhm']['ML'][0]:.2f} B {fr['plane_fwhm']['1-D part B(z)'][0]:.2f}"
              + f"; m variation {fr['m_at_env_peak_over_m_window_mean']:.3f}; region pos ratio {fr['region']['pos']['ratio']:.3f} signed {fr['region']['pos']['signed']:+.3f} e, neg ratio {fr['region']['neg']['ratio']:.3f} signed {fr['region']['neg']['signed']:+.3f} e"
              + (f"; ion lateral peak - edge: DFT {fr['ion_lateral']['dft_peak_minus_edge']:+.2f} ML {fr['ion_lateral']['ml_peak_minus_edge']:+.2f} (ML-DFT {fr['ion_lateral']['ml_minus_dft']:+.2f}) A" if "ion_lateral" in fr else ""), flush=True)
    rec["response_region"] = region_integrals(ML[k] - ML[k + 600], Db - Dbn, dV)
    rr = rec["response_region"]
    print(f"[{time.time() - T0:5.0f}s] pair {k} response: pos ratio {rr['pos']['ratio']:.3f} signed {rr['pos']['signed']:+.3f} e (DFT {rr['pos']['dft']:+.3f}), neg ratio {rr['neg']['ratio']:.3f} signed {rr['neg']['signed']:+.3f} e (DFT {rr['neg']['dft']:+.3f})", flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1)
    del C, ML, Db, Di, Dbn, Din; torch.cuda.empty_cache()
print(f"[{time.time() - T0:5.0f}s] done; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
