"""Definitive fair comparison: both models, same process, same cache, all val structs."""
import os, sys
os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
import numpy as np, torch
from ase.io import read
xyz, ref_npz = sys.argv[1], sys.argv[2]
m_sup, m_uns = sys.argv[3], sys.argv[4]
sigma = 0.25
torch.set_default_dtype(torch.float64); device="cuda"
from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.loss import _load_solvent_rhob_1d_npz, _gaussian_smear_periodic_1d
kspec = KeySpecification(info_keys={"energy":"energy","total_charge":"total_charge","total_spin":"total_spin","sample_id":"sample_id","fermi_level":"Fermi","potential":"potential_diff"}, arrays_keys={"forces":"forces","charges":"REF_charges"})
refs=_load_solvent_rhob_1d_npz(ref_npz); tgt=refs["targets"]; lz=refs["lz_A"]
frames=[a for a in read(xyz,":") if int(a.info["sample_id"]) in tgt]
def profiles(mpath):
    model=torch.load(mpath,map_location=device).to(device); model.eval()
    zt=utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    res={}
    for a in frames:
        sid=int(a.info["sample_id"])
        cfg=mace_data.config_from_atoms(a,key_specification=kspec)
        ds=[mace_data.AtomicData.from_config(cfg,z_table=zt,cutoff=float(model.r_max))]
        b=next(iter(torch_geometric.dataloader.DataLoader(ds,batch_size=1))).to(device)
        with torch.no_grad(): p=model(b.to_dict(),training=False,compute_force=False)
        if float(p["solvent_rho_bound_1d_mask"].view(-1)[0])<0.5: continue
        res[sid]=p["solvent_rho_bound_1d"][0].detach().cpu()
    del model; torch.cuda.empty_cache(); return res
S=profiles(m_sup); U=profiles(m_uns)
sids=sorted(set(S)&set(U))
def sm(x): return _gaussian_smear_periodic_1d(x.unsqueeze(0),sigma,lz)[0]
esup=eun=base=n=0.0; ndiff=0; maxdiff=0.0
for sid in sids:
    r=sm(torch.tensor(tgt[sid])); ms=sm(S[sid]); mu=sm(U[sid])
    esup+=float(((ms-r)**2).sum()); eun+=float(((mu-r)**2).sum()); base+=float((r**2).sum()); n+=r.numel()
    md=float((S[sid]-U[sid]).abs().max()); maxdiff=max(maxdiff,md)
    if md>1e-5: ndiff+=1
print(f"n={len(sids)} structs")
print(f"supervised   aggregate smeared RMSE = {(esup/n)**0.5:.8f} e/A^3")
print(f"unsupervised aggregate smeared RMSE = {(eun/n)**0.5:.8f} e/A^3")
print(f"zero baseline                       = {(base/n)**0.5:.8f}")
print(f"structs with raw max|sup-uns|>1e-5: {ndiff}/{len(sids)}; global max raw diff {maxdiff:.3e}")
