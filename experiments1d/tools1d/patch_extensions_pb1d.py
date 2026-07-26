"""One-shot wiring patch: add the pb1d solvent mode to extensions.py.

Scheme C wiring:
  stage 1 (pre-recursion)  = cached previous-encounter profile -> SCF features
                             (first encounter: fresh prior-only solve, cached)
  stage 2 (post-recursion) = fresh solve with the FINAL density + residual head
                             -> energy (detached) + observables (grad) + cache
Applied once; asserts every anchor is unique before editing.
"""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "polar-mace-planar" / "mace" / "modules" / "extensions.py"
s = P.read_text()
n0 = len(s)

def rep(old, new, count=1):
    global s
    assert s.count(old) == count, f"anchor x{s.count(old)}: {old[:80]!r}"
    s = s.replace(old, new)

# ---- 1) constructor: accept pb1d + head hyperparameters ----
rep('''        if solvent_model not in ("planar", "pb"):
            raise ValueError(
                f"solvent_model must be 'planar' or 'pb', got {solvent_model!r}"
            )
        if solvent_model == "pb" and not solvent_pb_config:
            raise ValueError("solvent_model='pb' requires solvent_pb_config")''',
'''        if solvent_model not in ("planar", "pb", "pb1d"):
            raise ValueError(
                f"solvent_model must be 'planar', 'pb' or 'pb1d', got {solvent_model!r}"
            )
        if solvent_model in ("pb", "pb1d") and not solvent_pb_config:
            raise ValueError(f"solvent_model={solvent_model!r} requires solvent_pb_config")''')

rep('''        solvent_pb_learn_center_shift: bool = False,
        solvent_pb_differentiable: bool = False,''',
'''        solvent_pb_learn_center_shift: bool = False,
        solvent_pb_differentiable: bool = False,
        solvent_pb1d_zones: int = 8,
        solvent_pb1d_sigma_z: float = 0.2,
        solvent_pb1d_c_max: float = 0.25,
        solvent_pb1d_upsample: int = 2,
        solvent_pb1d_max_outer: int = 12,''')

rep('''        self.solvent_pb_differentiable = bool(solvent_pb_differentiable)
        self._pb_solver = None''',
'''        self.solvent_pb_differentiable = bool(solvent_pb_differentiable)
        self.solvent_pb1d_zones = int(solvent_pb1d_zones)
        self.solvent_pb1d_sigma_z = float(solvent_pb1d_sigma_z)
        self.solvent_pb1d_c_max = float(solvent_pb1d_c_max)
        self.solvent_pb1d_upsample = int(solvent_pb1d_upsample)
        self.solvent_pb1d_max_outer = int(solvent_pb1d_max_outer)
        self._pb1d_backend = None
        self._pb_solver = None''')

# head module (registered parameters) — placed where hidden_irreps is in scope
rep('''        self.solvent_scalar_feature_dim = 3
        self.solvent_interface_pool_sigma = 1.0''',
'''        self.solvent_scalar_feature_dim = 3
        self.solvent_interface_pool_sigma = 1.0
        self.pb1d_head = None
        if solvent_model == "pb1d":
            from .pb1d_head import PB1DResidualHead

            self.pb1d_head = PB1DResidualHead(
                feat_dim=hidden_irreps.dim * num_interactions,
                n_zones=self.solvent_pb1d_zones,
                sigma_z=self.solvent_pb1d_sigma_z,
                c_max=self.solvent_pb1d_c_max,
            )''')

# ---- 2) backend factory + stage methods + cache helpers ----
rep('''    def _get_pb_solver(self):''',
'''    def _get_pb1d_backend(self):
        if self._pb1d_backend is None:
            from .pb1d_backend import PB1DBackend

            self._pb1d_backend = PB1DBackend(
                config_path=self.solvent_pb_config,
                repo_path=self.solvent_pb_repo,
                grid_spacing=self.solvent_pb_grid_spacing,
                baseline_cache=self.solvent_pb_baseline_cache,
                solve_upsample=self.solvent_pb1d_upsample,
                fixsol_steps=self.solvent_pb_fixsol_steps,
                tol=self.solvent_pb_tol,
                max_outer=self.solvent_pb1d_max_outer,
                axis=self.solvent_potential_axis,
            )
        return self._pb1d_backend

    def _pb1d_cached_profile(self, sid: int):
        d = self.solvent_pb_phi_cache_dir
        if not d:
            return self._pb_profile_cache.get(("pb1d", sid))
        try:
            z = np.load(os.path.join(d, f"prof1d_{sid}.npz"))
            return {
                "feat": torch.from_numpy(z["feat"]),
                "energy": torch.from_numpy(z["energy"]),
                "q": float(z["q"]), "mu": float(z["mu"]),
                "layer_mean": float(z["lm"]),
            }
        except (OSError, ValueError, KeyError):
            return None

    def _pb1d_store_profile(self, sid: int, entry) -> None:
        if os.environ.get("MACE_PB1D_CACHE_READONLY"):
            return
        d = self.solvent_pb_phi_cache_dir
        if not d:
            self._pb_profile_cache[("pb1d", sid)] = entry
            return
        p = os.path.join(d, f"prof1d_{sid}.npz")
        tmp = f"{p}.tmp.{os.getpid()}.npz"
        np.savez(
            tmp[:-4], feat=entry["feat"].numpy(), energy=entry["energy"].numpy(),
            q=entry["q"], mu=entry["mu"], lm=entry["layer_mean"],
        )
        os.replace(tmp, p)

    def _pb1d_planar_result(self, g, cell_g, H_g, planar_center, total_charge_g,
                            positions):
        """Truncated-Gaussian planar fallback in the pb result format."""
        axis = self.solvent_potential_axis
        cell_g33 = cell_g.view(3, 3)
        area_g = _cell_area_for_axis_batch(
            cell_g33.unsqueeze(0).to(positions.dtype), axis
        )[0]
        c0 = planar_center[g].to(positions.dtype)
        sg = float(self.solvent_sigma_g)
        nzf = 512
        zf = torch.arange(
            nzf, device=positions.device, dtype=positions.dtype
        ) * (H_g / nzf)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        normc = torch.clamp(
            0.5 * (torch.erf((H_g - c0) / sg * inv_sqrt2)
                   - torch.erf((0.0 - c0) / sg * inv_sqrt2)),
            min=1.0e-12,
        )
        q_fb = -float(total_charge_g[g].item())
        prof = (q_fb / area_g) * torch.exp(
            -0.5 * torch.square((zf - c0) / sg)
        ) / (math.sqrt(2.0 * math.pi) * sg * normc)
        lm_fb = _truncated_gaussian_mean(
            center=c0.view(1), sigma=sg,
            lower=torch.zeros(1, device=positions.device, dtype=positions.dtype),
            upper=torch.full((1,), H_g, device=positions.device, dtype=positions.dtype),
        ).view(())
        return {
            "z": zf, "rho_layer_z": prof, "height": H_g,
            "q_ion": q_fb, "layer_mean": float(lm_fb.item()),
            "mu": q_fb * float(lm_fb.item()),
        }

    def _pb1d_run_graphs(
        self,
        data,
        positions,
        cell,
        radial_blocks,
        node_valence_electrons,
        num_graphs,
        planar_center,
        node_feats_mixed=None,
        use_head=False,
        want_grad=False,
        prev_data=None,
        write_cache=True,
        use_cache_rows=False,
    ):
        """Shared per-graph loop for the pb1d stages.

        use_cache_rows: stage-1 semantics — a cached profile short-circuits
        the solve (scheme C: the SCF features see the previous encounter's
        solvent). Fresh solves happen for cache misses (first encounter).
        """
        from .pb_solvent import resample_profile_periodic_torch

        backend = self._get_pb1d_backend()
        axis = self.solvent_potential_axis
        cells = cell.detach()
        if cells.dim() == 2 and cells.shape[1] == 3:
            cells = cells.view(-1, 3, 3)
        pbc_g = data["pbc"].view(-1, 3).to(torch.bool)
        slab = torch.ones(3, dtype=torch.bool, device=pbc_g.device)
        slab[axis] = False
        slab_mask = torch.all(pbc_g == slab, dim=1)
        total_charge_g = data["total_charge"].view(-1)
        sample_ids = data.get("sample_id")
        if sample_ids is not None:
            sample_ids = sample_ids.view(-1)

        q_ion = positions.new_zeros(num_graphs)
        solvent_mu = positions.new_zeros(num_graphs)
        layer_mean = positions.new_zeros(num_graphs)
        prof_feat = positions.new_zeros(num_graphs, 1024)
        prof_energy = positions.new_zeros(num_graphs, 512)
        prof_feat_grad = positions.new_zeros(num_graphs, 1024) if want_grad else None

        for g in range(num_graphs):
            cell_g = cells[g]
            H_g = float(_axis_box_length(cell_g, axis).item())
            atom_mask = data["batch"] == g
            if not bool(slab_mask[g].item()) or not bool(torch.any(atom_mask).item()):
                continue
            sid = (
                int(sample_ids[g].detach().cpu().item())
                if sample_ids is not None else None
            )
            cached = self._pb1d_cached_profile(sid) if sid is not None else None
            if use_cache_rows and cached is not None:
                prof_feat[g] = cached["feat"].to(positions.device, positions.dtype)
                prof_energy[g] = cached["energy"].to(positions.device, positions.dtype)
                q_ion[g] = cached["q"]
                solvent_mu[g] = cached["mu"]
                layer_mean[g] = cached["layer_mean"]
                if prof_feat_grad is not None:
                    prof_feat_grad[g] = prof_feat[g]
                continue

            pos_g = positions[atom_mask].detach()
            coeffs_g = (
                radial_blocks[atom_mask]
                if want_grad else radial_blocks[atom_mask].detach()
            )
            zval_g = node_valence_electrons[atom_mask].detach()
            feats_g = None
            if use_head and node_feats_mixed is not None:
                feats_g = (
                    node_feats_mixed[atom_mask]
                    if want_grad else node_feats_mixed[atom_mask].detach()
                )
            try:
                result = backend.solve_graph(
                    positions=pos_g,
                    cell=cell_g,
                    z_valence=zval_g,
                    total_charge=float(total_charge_g[g].item()),
                    sample_id=sid,
                    radial_coeffs=coeffs_g,
                    sigmas=self.atomic_density_sigmas,
                    node_feats=feats_g,
                    head=self.pb1d_head if use_head else None,
                    q_tot=total_charge_g[g].detach(),
                )
                solved_ok = True
            except RuntimeError as exc:
                if os.environ.get("MACE_PB_DEBUG"):
                    print(f"PB1DDBG-ERROR sid={sid}: {exc}", flush=True)
                result = None
                solved_ok = False

            healthy = False
            if solved_ok:
                rms_h = float(result["rms_last"])
                lm_h = float(result["layer_mean"])
                mb_h = float(result["mu_bound"])
                healthy = (
                    rms_h == rms_h
                    and rms_h < 10.0 * self.solvent_pb_tol
                    and 0.0 <= lm_h <= H_g
                    and abs(mb_h) <= 2.0 * H_g
                )
            if not healthy:
                if os.environ.get("MACE_PB_DEBUG"):
                    print(f"PB1DDBG-FALLBACK sid={sid} healthy={healthy}", flush=True)
                if cached is not None:
                    prof_feat[g] = cached["feat"].to(positions.device, positions.dtype)
                    prof_energy[g] = cached["energy"].to(positions.device, positions.dtype)
                    q_ion[g] = cached["q"]
                    solvent_mu[g] = cached["mu"]
                    layer_mean[g] = cached["layer_mean"]
                    if prof_feat_grad is not None:
                        prof_feat_grad[g] = prof_feat[g]
                    continue
                fb = self._pb1d_planar_result(
                    g, cell_g, H_g, planar_center, total_charge_g, positions)
                layer = fb["rho_layer_z"].to(positions.dtype)
                prof_feat[g] = resample_profile_periodic_torch(layer, H_g, 1024, False).detach()
                prof_energy[g] = resample_profile_periodic_torch(layer, H_g, 512, True).detach()
                q_ion[g] = fb["q_ion"]
                solvent_mu[g] = fb["mu"]
                layer_mean[g] = fb["layer_mean"]
                if prof_feat_grad is not None:
                    prof_feat_grad[g] = prof_feat[g]
                continue

            layer = result["rho_layer_z"].to(positions.dtype)
            mu_g = result["q_ion_t"] * result["layer_mean_t"] + result["mu_bound_t"]
            prof_feat[g] = resample_profile_periodic_torch(
                layer, H_g, 1024, False).detach()
            prof_energy[g] = resample_profile_periodic_torch(
                layer, H_g, 512, True).detach()
            if prof_feat_grad is not None:
                prof_feat_grad[g] = resample_profile_periodic_torch(layer, H_g, 1024, False)
                q_ion[g] = result["q_ion_t"].to(positions.dtype)
                solvent_mu[g] = mu_g.to(positions.dtype)
            else:
                q_ion[g] = float(result["q_ion"])
                solvent_mu[g] = float(mu_g.detach())
            layer_mean[g] = float(result["layer_mean"])  # detached: feeds solv_center/energy
            if os.environ.get("MACE_PB_DEBUG"):
                print(
                    f"PB1DDBG sid={sid} q_ion={float(result['q_ion']):+.4f} "
                    f"layer_mean={float(result['layer_mean']):+.3f} "
                    f"mu={float(mu_g.detach()):+.3f} "
                    f"diag={backend.last_diagnostics}",
                    flush=True,
                )
            if write_cache and sid is not None:
                self._pb1d_store_profile(sid, {
                    "feat": prof_feat[g].detach().to(torch.float32).cpu(),
                    "energy": prof_energy[g].detach().to(torch.float32).cpu(),
                    "q": float(result["q_ion"]),
                    "mu": float(mu_g.detach()),
                    "layer_mean": float(result["layer_mean"]),
                })

        out = {
            "profile_features": prof_feat,
            "profile_energy": prof_energy,
            "q_ion": q_ion,
            "layer_mean": layer_mean,
            "solvent_mu": solvent_mu,
        }
        if prof_feat_grad is not None:
            out["profile_features_grad"] = prof_feat_grad
        return out

    def _pb1d_stage1(self, data, positions, cell, radial_blocks,
                     node_valence_electrons, num_graphs, planar_center):
        """Scheme C stage 1: SCF features from the cached previous-encounter
        profile; first encounter = fresh prior-only solve on the pre-recursion
        density (detached), which also seeds the cache."""
        return self._pb1d_run_graphs(
            data, positions, cell, radial_blocks, node_valence_electrons,
            num_graphs, planar_center,
            use_head=False, want_grad=False, use_cache_rows=True, write_cache=True,
        )

    def _pb1d_stage2(self, data, positions, cell, radial_blocks,
                     node_valence_electrons, num_graphs, planar_center,
                     node_feats_mixed, prev_data):
        """Scheme C stage 2: fresh solve with the FINAL (post-recursion)
        density + residual head. Observables carry gradients; the energy rows
        are detached; the cache is refreshed for the next encounter."""
        out = self._pb1d_run_graphs(
            data, positions, cell, radial_blocks, node_valence_electrons,
            num_graphs, planar_center,
            node_feats_mixed=node_feats_mixed, use_head=True,
            want_grad=self.training or torch.is_grad_enabled(),
            use_cache_rows=False, write_cache=True,
        )
        return out

    def _get_pb_solver(self):''')

# ---- 3) pre-recursion stage-1 branch ----
rep('''        pb_solvent_data: Optional[Dict[str, torch.Tensor]] = None
        if self.solvent_model == "pb":
            pb_solvent_data = self._solve_pb_profiles(
                data=data,
                positions=positions,
                cell=cell,
                radial_blocks=self._radial_flat_to_blocks(comp_charge_density),
                node_valence_electrons=node_valence_electrons,
                num_graphs=num_graphs,
                planar_center=comp_center_init,
            )''',
'''        pb_solvent_data: Optional[Dict[str, torch.Tensor]] = None
        if self.solvent_model in ("pb", "pb1d"):
            if self.solvent_model == "pb1d":
                pb_solvent_data = self._pb1d_stage1(
                    data=data,
                    positions=positions,
                    cell=cell,
                    radial_blocks=self._radial_flat_to_blocks(comp_charge_density),
                    node_valence_electrons=node_valence_electrons,
                    num_graphs=num_graphs,
                    planar_center=comp_center_init,
                )
            else:
                pb_solvent_data = self._solve_pb_profiles(
                    data=data,
                    positions=positions,
                    cell=cell,
                    radial_blocks=self._radial_flat_to_blocks(comp_charge_density),
                    node_valence_electrons=node_valence_electrons,
                    num_graphs=num_graphs,
                    planar_center=comp_center_init,
                )''')

# ---- 4) post-recursion stage-2 injection ----
rep('''        solvent_raw_shift = self.solvent_center_residual(solvent_head_inputs).view(-1)''',
'''        solvent_raw_shift = self.solvent_center_residual(solvent_head_inputs).view(-1)
        if self.solvent_model == "pb1d":
            pb_solvent_data = self._pb1d_stage2(
                data=data,
                positions=positions,
                cell=cell,
                radial_blocks=charge_density_radial_coefficients,
                node_valence_electrons=node_valence_electrons,
                num_graphs=num_graphs,
                planar_center=comp_center_init,
                node_feats_mixed=node_feats_out,
                prev_data=pb_solvent_data,
            )''')

P.write_text(s)
print(f"patched extensions.py ({n0} -> {len(s)} chars)")
