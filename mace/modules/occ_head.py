"""Equivariant readout of PAW augmentation occupancies (CHGCAR aug blocks).

Layout (decoded from the 600-structure zero mask, 2026-08-12, and verified
slot-by-slot: the seven always-zero runs in the Ni blocks land exactly on the
parity-allowed L=3,4 components, which VASP truncates at the default
LMAXMIX=2):

  for i in channels:            # POTCAR partial-wave order (Ni: d d s s p p)
    for j in channels[i:]:      # upper triangle, i outer
      for L in |li-lj| .. li+lj step 2:   # parity-allowed, ascending
        (2L+1) real components, m = -L..L

Each (pair, L) block transforms as irrep L, so the head reads it from the
matching-L channels of the equivariant node features with a per-element
linear channel mix — the same construction as the model's own multipole
readouts, hence exactly equivariant. L>2 blocks carry no signal here
(LMAXMIX truncation) and are emitted as exact zeros. L=0 slots get a
learnable bias (their targets have O(1..7) means); L>0 biases would break
equivariance and are omitted.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def build_slot_map(l_list: List[int], l_max_signal: int = 2):
    """Packed layout for one element.

    Returns (total_len, slots) where slots = [(offset, L)] for the
    signal-carrying blocks (L <= l_max_signal); everything else is a
    structural zero under the LMAXMIX truncation.
    """
    slots: List[Tuple[int, int]] = []
    off = 0
    n = len(l_list)
    for i in range(n):
        for j in range(i, n):
            li, lj = l_list[i], l_list[j]
            for L in range(abs(li - lj), li + lj + 1, 2):
                if L <= l_max_signal:
                    slots.append((off, L))
                off += 2 * L + 1
    return off, slots


def parse_channel_spec(spec: str) -> Dict[int, List[int]]:
    """'1:0,0,1;6:0,0,1,1;28:2,2,0,0,1,1' -> {Z: l_list} (POTCAR order)."""
    out: Dict[int, List[int]] = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        z, ls = part.split(":")
        out[int(z)] = [int(x) for x in ls.split(",")]
    return out


# NiN-mix dataset (PAW_PBE POTCARs, partial-wave order verified 2026-08-12)
DEFAULT_CHANNEL_SPEC = "1:0,0,1;6:0,0,1,1;7:0,0,1,1;8:0,0,1,1;28:2,2,0,0,1,1"


class OccAugHead(torch.nn.Module):
    def __init__(
        self,
        irreps_per_layer: List[str],
        zs: List[int],
        channel_spec: str = DEFAULT_CHANNEL_SPEC,
    ) -> None:
        super().__init__()
        import re

        # concatenated node_feats layout -> per-L feature channel offsets
        blocks: List[Tuple[int, int, int]] = []  # (offset, l, mul)
        off = 0
        for s in irreps_per_layer:
            for part in str(s).split("+"):
                m = re.match(r"\s*(\d+)x(\d+)[eo]\s*", part)
                if m is None:
                    raise ValueError(f"cannot parse irreps chunk: {part!r}")
                mul, l = int(m.group(1)), int(m.group(2))
                blocks.append((off, l, mul))
                off += mul * (2 * l + 1)
        self.feat_dim = off
        self.feat_blocks_by_l: Dict[int, List[Tuple[int, int]]] = {}
        for l in (0, 1, 2):
            self.feat_blocks_by_l[l] = [(o, m) for (o, lb, m) in blocks if lb == l]

        chan = parse_channel_spec(channel_spec)
        self.zs = [int(z) for z in zs]
        self.max_len = 0
        self.slot_meta: Dict[int, dict] = {}
        self.weights = torch.nn.ParameterDict()
        self.bias0 = torch.nn.ParameterDict()
        for z in self.zs:
            if z not in chan:
                continue
            total, slots = build_slot_map(chan[z])
            self.max_len = max(self.max_len, total)
            by_l: Dict[int, List[int]] = {0: [], 1: [], 2: []}
            for off_s, L in slots:
                by_l[L].append(off_s)
            self.slot_meta[z] = {"total": total, "by_l": by_l}
            for L in (0, 1, 2):
                n_slots = len(by_l[L])
                if n_slots == 0:
                    continue
                c_l = sum(m for (_, m) in self.feat_blocks_by_l[L])
                w = torch.nn.Parameter(torch.zeros(n_slots, c_l))
                self.weights[f"z{z}_l{L}"] = w
            self.bias0[f"z{z}"] = torch.nn.Parameter(
                torch.zeros(len(by_l[0]))
            )

    def _gather_feats_l(self, node_feats: torch.Tensor, L: int) -> torch.Tensor:
        """[N, D] -> [N, C_L, 2L+1] stacking all mul-blocks of angular order L."""
        chunks = [
            node_feats[:, o : o + m * (2 * L + 1)].reshape(-1, m, 2 * L + 1)
            for (o, m) in self.feat_blocks_by_l[L]
        ]
        return torch.cat(chunks, dim=1)

    def forward(self, node_feats: torch.Tensor, node_z: torch.Tensor) -> torch.Tensor:
        """-> [N, max_len] packed occupancies (structural zeros included)."""
        n = node_feats.shape[0]
        out = node_feats.new_zeros((n, self.max_len))
        for z in self.zs:
            meta = self.slot_meta.get(z)
            if meta is None:
                continue
            sel = (node_z == z).nonzero(as_tuple=False).view(-1)
            if sel.numel() == 0:
                continue
            feats = node_feats[sel]
            for L in (0, 1, 2):
                offs = meta["by_l"][L]
                if not offs:
                    continue
                fL = self._gather_feats_l(feats, L)      # [n_z, C_L, 2L+1]
                w = self.weights[f"z{z}_l{L}"].to(fL.dtype)
                vals = torch.einsum("sc,nck->nsk", w, fL)  # [n_z, n_slots, 2L+1]
                if L == 0:
                    vals = vals + self.bias0[f"z{z}"].to(fL.dtype)[None, :, None]
                width = 2 * L + 1
                for k, off_s in enumerate(offs):
                    out[sel[:, None], torch.arange(off_s, off_s + width,
                        device=out.device)[None, :]] = vals[:, k, :]
        return out
