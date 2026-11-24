"""Stage 2 Design (B): Pairwise GQA-Unlocked Hardware-Tile-Aligned Iso-MAC Quota Search.

Key engineering guarantees (incorporating Codex 012 + external review):
  1. compute_initial_quotas() is shared identically by Stage 1 (search OFF) and Stage 1+2 (search ON),
     so both legs start from the EXACT same Tile-aligned quota, mask rule, and init_high_MACs.
  2. Per-projection local probe step delta_p = snap_tile(round(0.02 * C_p), T) ensures GQA k_proj/v_proj
     always have valid in-bounds sensitivity probes (fixing the global max_O lockout bug).
  3. Pairwise adaptive step builder compute_pairwise_swap(r, d) scales each candidate pair (r, d)
     by its own (O_r, O_d) ratio (e.g., 128:32 for 4:1 GQA k/v <-> q/o on LLaMA-3-8B, 256:32 for
     8:1 GQA on TinyLlama-1.1B, 224:64 for q/o <-> up/gate on 8B), strictly maintaining:
       - Tile alignment: (C_p - q_p) % tile == 0
       - Strict iso-MAC: final_high_MACs <= init_high_MACs
       - Box bounds: q_p in [min_low[p], max_low[p]]
"""

import math
import random
from typing import Dict, Optional, Tuple
import torch


def _snap_tile(val: int, tile: int) -> int:
    if tile <= 1:
        return max(1, int(val))
    return max(tile, int(round(val / tile)) * tile)


def compute_initial_quotas(
    bd: Dict[str, object],
    low_p: float,
    tile: int = 0,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Compute initial low-saliency quotas, box bounds, and local probe steps for all projections.

    Returns: (init_quotas, min_low, max_low, probe_delta)
    """
    T = tile if tile > 0 else 1
    init_quotas: Dict[str, int] = {}
    min_low: Dict[str, int] = {}
    max_low: Dict[str, int] = {}
    probe_delta: Dict[str, int] = {}

    for name, dw in bd.items():
        C = int(dw.in_features)
        raw_hi = C - int(low_p * C)
        if tile > 0:
            init_hi = max(tile, (raw_hi // tile) * tile)
            min_hi = tile
            max_hi = max(init_hi, max(tile, (int(0.30 * C) // tile) * tile))
            max_hi = min(max_hi, ((C - 1) // tile) * tile)
        else:
            init_hi = max(1, raw_hi)
            min_hi = max(1, C - min(C - 1, int(0.98 * C)))
            max_hi = max(init_hi, C - max(1, int(0.70 * C)))
            max_hi = min(max_hi, C - 1)

        init_quotas[name] = C - init_hi
        min_low[name] = C - max_hi
        max_low[name] = C - min_hi

        # Local per-projection probe step (always in-bounds, unlocks GQA k/v!)
        dp = _snap_tile(int(round(0.02 * C)), T)
        max_span = max(T, max_hi - min_hi)
        probe_delta[name] = min(dp, (max_span // T) * T) if tile > 0 else min(dp, max_span)
        probe_delta[name] = max(T, probe_delta[name])

    return init_quotas, min_low, max_low, probe_delta


def compute_pairwise_swap(
    rec: str,
    don: str,
    bd: Dict[str, object],
    quotas: Dict[str, int],
    min_low: Dict[str, int],
    max_low: Dict[str, int],
    probe_delta: Dict[str, int],
    tile: int = 0,
) -> Optional[Tuple[int, int]]:
    """Compute a strictly tile-aligned, strictly iso-MAC channel swap (dk_rec, dk_don) for pair (rec, don).

    Guarantees:
      - dk_rec * O_rec <= dk_don * O_don
      - dk_rec % T == 0 and dk_don % T == 0 (when tile > 0)
      - quotas[rec] - dk_rec >= min_low[rec] and quotas[don] + dk_don <= max_low[don]
    """
    Or = int(bd[rec].out_features)
    Od = int(bd[don].out_features)
    Cr = int(bd[rec].in_features)
    Cd = int(bd[don].in_features)
    room_r = quotas[rec] - min_low[rec]
    room_d = max_low[don] - quotas[don]
    T = tile if tile > 0 else 1

    if room_r < T or room_d < T:
        return None

    if tile > 0:
        # Search compact tile multiples on the larger-O side; minimize MAC slack with cap on single-step jump.
        target_big = probe_delta[rec] if Or >= Od else probe_delta[don]
        max_jump_r = max(T, int(0.15 * Cr))
        max_jump_d = max(T, int(0.15 * Cd))

        best = None  # (key, dk_r, dk_d)
        fallback = None
        for m in (1, 2, 3, 4):
            if Or >= Od:
                dk_r = m * T
                dk_d = int(math.ceil((dk_r * Or) / (Od * T))) * T
            else:
                dk_d = m * T
                dk_r = (dk_d * Od // (Or * T)) * T
            if dk_r < T or dk_d < T:
                continue
            if dk_r > room_r or dk_d > room_d:
                continue
            used = dk_r * Or
            freed = dk_d * Od
            if used > freed:
                continue
            slack = (freed - used) / float(freed)
            step_dist = abs(dk_r - probe_delta[rec]) + abs(dk_d - probe_delta[don])
            score = (round(slack, 4), step_dist, m)
            if fallback is None or score < fallback[0]:
                fallback = (score, dk_r, dk_d)
            if dk_r <= max_jump_r and dk_d <= max_jump_d:
                if best is None or score < best[0]:
                    best = (score, dk_r, dk_d)

        chosen = best if best is not None else fallback
        if chosen is None:
            return None
        return chosen[1], chosen[2]

    # Unaligned mode (tile == 0): pairwise scaled step with exact shrinking to box bounds
    if Or >= Od:
        dk_r = min(probe_delta[rec], room_r, (room_d * Od) // Or)
        if dk_r < 1:
            return None
        dk_d = int(math.ceil((dk_r * Or) / float(Od)))
        if dk_d > room_d:
            return None
        return dk_r, dk_d
    else:
        dk_d = min(probe_delta[don], room_d, int(math.ceil((room_r * Or) / float(Od))))
        if dk_d < 1:
            return None
        dk_r = (dk_d * Od) // Or
        if dk_r < 1 or dk_r > room_r:
            return None
        return dk_r, dk_d


def apply_mask_for_quota(dw, q_low: int) -> None:
    score = dw.saliency_score
    thresh = torch.sort(score)[0][q_low - 1]
    dw.mask_low = (score <= thresh)


@torch.no_grad()
def run_stage2_search(
    bd: Dict[str, object],
    eval_mse_fn,
    init_quotas: Dict[str, int],
    min_low: Dict[str, int],
    max_low: Dict[str, int],
    probe_delta: Dict[str, int],
    tile: int = 32,
    policy: str = "adaptive",
    max_steps: int = 8,
    seed: int = 2,
    layer_idx: int = 0,
) -> Tuple[Dict[str, int], float, float, int, int, int]:
    """Run Stage 2 iso-MAC quota reallocation on a single decoder block.

    Returns: (final_quotas, init_mse, final_mse, init_macs, final_macs, n_swaps)
    """
    quotas = dict(init_quotas)
    init_macs = sum((int(bd[n].in_features) - quotas[n]) * int(bd[n].out_features) for n in bd)
    for n in bd:
        apply_mask_for_quota(bd[n], quotas[n])

    init_mse = cur_mse = eval_mse_fn()
    n_swaps = 0

    if policy == "adaptive":
        for _ in range(max_steps):
            gain_rate: Dict[str, float] = {}
            cost_rate: Dict[str, float] = {}
            for n, dw in bd.items():
                dp = probe_delta[n]
                q0 = quotas[n]
                O = int(dw.out_features)
                if q0 - dp >= min_low[n]:
                    apply_mask_for_quota(dw, q0 - dp)
                    gain_rate[n] = (eval_mse_fn() - cur_mse) / float(dp * O)
                if q0 + dp <= max_low[n]:
                    apply_mask_for_quota(dw, q0 + dp)
                    cost_rate[n] = (eval_mse_fn() - cur_mse) / float(dp * O)
                apply_mask_for_quota(dw, q0)

            candidates = []
            for r, gr in gain_rate.items():
                for d, cd in cost_rate.items():
                    if r != d and (gr + cd) < -1e-12:
                        candidates.append((gr + cd, r, d))
            candidates.sort(key=lambda x: x[0])

            accepted = False
            for _, r, d in candidates:
                swap = compute_pairwise_swap(r, d, bd, quotas, min_low, max_low, probe_delta, tile=tile)
                if swap is None:
                    continue
                dk_r, dk_d = swap
                quotas[r] -= dk_r
                quotas[d] += dk_d
                apply_mask_for_quota(bd[r], quotas[r])
                apply_mask_for_quota(bd[d], quotas[d])
                new_mse = eval_mse_fn()
                if new_mse < cur_mse - 1e-6:
                    cur_mse = new_mse
                    n_swaps += 1
                    accepted = True
                    break
                # Revert rejected candidate and try next best feasible pair
                quotas[r] += dk_r
                quotas[d] -= dk_d
                apply_mask_for_quota(bd[r], quotas[r])
                apply_mask_for_quota(bd[d], quotas[d])
            if not accepted:
                break

    elif policy == "random":
        rng = random.Random(seed * 1000 + layer_idx)
        names = list(bd.keys())
        for _ in range(max_steps):
            pairs = [(r, d) for r in names for d in names if r != d]
            rng.shuffle(pairs)
            moved = False
            for r, d in pairs:
                swap = compute_pairwise_swap(r, d, bd, quotas, min_low, max_low, probe_delta, tile=tile)
                if swap is not None:
                    dk_r, dk_d = swap
                    quotas[r] -= dk_r
                    quotas[d] += dk_d
                    apply_mask_for_quota(bd[r], quotas[r])
                    apply_mask_for_quota(bd[d], quotas[d])
                    n_swaps += 1
                    moved = True
                    break
            if not moved:
                break
        cur_mse = eval_mse_fn()

    elif policy == "heuristic_down_o":
        recs = [n for n in ("down_proj", "o_proj") if n in bd]
        dons = [n for n in ("q_proj", "gate_proj", "up_proj", "k_proj", "v_proj") if n in bd]
        for step in range(max_steps):
            rec = recs[step % len(recs)]
            moved = False
            for don in dons:
                if rec == don:
                    continue
                swap = compute_pairwise_swap(rec, don, bd, quotas, min_low, max_low, probe_delta, tile=tile)
                if swap is not None:
                    dk_r, dk_d = swap
                    quotas[rec] -= dk_r
                    quotas[don] += dk_d
                    apply_mask_for_quota(bd[rec], quotas[rec])
                    apply_mask_for_quota(bd[don], quotas[don])
                    n_swaps += 1
                    moved = True
                    break
            if not moved:
                break
        cur_mse = eval_mse_fn()
    else:
        raise ValueError(f"Unknown Stage 2 policy: {policy}")

    final_macs = sum((int(bd[n].in_features) - quotas[n]) * int(bd[n].out_features) for n in bd)
    assert final_macs <= init_macs, f"Iso-MAC violation at layer {layer_idx}: {final_macs} > {init_macs}"
    if tile > 0:
        for n in bd:
            hi = int(bd[n].in_features) - quotas[n]
            assert hi % tile == 0, f"Tile-{tile} alignment violation on {n}: hi={hi}"

    return quotas, init_mse, cur_mse, init_macs, final_macs, n_swaps
