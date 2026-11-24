"""Official spcl/QuaRot-GPTQ + OSDSQ v2 (Stage 1 + Stage 2 Design B Tile-Aligned) Production Runner.

Supports LLaMA-2-7B (MHA, Paley 172), LLaMA-3-8B (GQA, Paley 28), TinyLlama-1.1B (GQA, Paley 44),
and power-of-2 LLaMA models (e.g., llama-68m) with:
  - Single official spcl/QuaRot-GPTQ calibration pass cached and reused identically across all modes
  - Strict controlled comparison (Codex 012):
      * pure              : Official spcl/QuaRot-GPTQ W4A4 anchor
      * spikellm          : Original SpikeLLM-QuaRot (mean saliency, shield=off, tile=0, search=off)
      * stage1_mean_only  : Decoupled Stage 1 shielding only (mean saliency, shield=mean, tile=T, search=off)
      * stage1_tile       : Controlled Stage 1 anchor (mean+std saliency, shield=mean, tile=T, search=off)
      * stage1_2_tile     : Controlled Stage 1+2 Design B (mean+std, shield=mean, tile=T, Pairwise GQA-unlocked search=on)
      * stage1_unaligned  : Unaligned Stage 1 (tile=0, search=off)
      * stage1_2_unaligned: Unaligned Stage 1+2 (tile=0, search=on)
      * stage1_2_heuristic: Fixed heuristic_down_o policy ablation (tile=T)
      * stage1_2_random   : Iso-MAC random policy ablation (tile=T)
  - Separate cal_peakVRAM and eval_peakVRAM logging
  - Optional lm_eval zero-shot evaluation (aligned with SpikeLLM Table 4 / lm-eval-v0.4.2)

NOTE: The bundled pure-PyTorch fast_hadamard_transform.py fallback is strictly for accuracy/PPL
evaluation on Windows/environments without CUDA fast-hadamard-transform; never use it for wall-clock latency claims.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

# Put VENDOR_DIR first (so vendor/utils.py is not shadowed by SpikeLLM/utils.py), then PLUGIN_DIR, then REPO_ROOT.
PLUGIN_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PLUGIN_DIR / "vendor"
REPO_ROOT = PLUGIN_DIR.parent
for p in (str(REPO_ROOT), str(PLUGIN_DIR), str(VENDOR_DIR)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import torch
import torch.nn as nn
import transformers

import model_utils
import rotation_utils
import quant_utils
import gptq_utils
import hadamard_utils
import data_utils as quarot_data_utils
from dual_branch import DualBranchQuaRotWrapper
from stage2_search import compute_initial_quotas, apply_mask_for_quota, run_stage2_search


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def load_tokenizer_safe(model_path: str):
    try:
        return transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception:
        return transformers.AutoTokenizer.from_pretrained(model_path, use_fast=True)


def load_calib_and_eval_data(args):
    """Load calibration dataloader and wikitext2/c4 evaluation tensors, with cache reuse + SHA-256."""
    import datautils as spike_datautils

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    net_name = args.net or Path(args.model.rstrip("/\\")).name
    model_family = net_name.split("-")[0]

    meta = {"cache_dir": str(cache_dir), "model_family": model_family, "c4_eval_mode": args.c4_eval_mode}

    cal_path = cache_dir / f"dataloader_{model_family}_{args.cal_dataset}_{args.nsamples}.cache"
    if cal_path.exists():
        trainloader = torch.load(cal_path, weights_only=False)
        meta["calib_source"] = f"cache:{cal_path.name}"
    else:
        trainloader, _ = spike_datautils.get_loaders(
            args.cal_dataset, nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, model=args.model
        )
        torch.save(trainloader, cal_path)
        meta["calib_source"] = f"generated:{cal_path.name}"
    meta["calib_sha16"] = sha256_file(str(cal_path))

    wt2_path = cache_dir / f"testloader_{model_family}_wikitext2_all.cache"
    if wt2_path.exists():
        wt2_obj = torch.load(wt2_path, weights_only=False)
    else:
        _, wt2_obj = spike_datautils.get_loaders("wikitext2", seed=2, seqlen=args.seqlen, model=args.model)
        torch.save(wt2_obj, wt2_path)
    wt2_ids = wt2_obj.input_ids if hasattr(wt2_obj, "input_ids") else wt2_obj
    meta["wt2_sha16"] = sha256_tensor(wt2_ids)
    meta["wt2_shape"] = list(wt2_ids.shape)

    c4_suffix = "c4_all" if args.c4_eval_mode == "spikellm" else "c4_new_all"
    c4_path = cache_dir / f"testloader_{model_family}_{c4_suffix}.cache"
    if c4_path.exists():
        c4_obj = torch.load(c4_path, weights_only=False)
    else:
        if args.c4_eval_mode == "spikellm":
            _, c4_obj = spike_datautils.get_loaders("c4", seed=2, seqlen=args.seqlen, model=args.model)
        else:
            c4_obj = quarot_data_utils.get_c4_new(
                nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, model=args.model, eval_mode=True
            )
        torch.save(c4_obj, c4_path)
    c4_ids = c4_obj.input_ids if hasattr(c4_obj, "input_ids") else c4_obj
    meta["c4_sha16"] = sha256_tensor(c4_ids)
    meta["c4_shape"] = list(c4_ids.shape)

    return trainloader, wt2_ids, c4_ids, meta


@torch.no_grad()
def eval_ppl(model, testenc: torch.Tensor, dev: torch.device, seqlen: int = 2048) -> float:
    model.eval()
    model.to(dev)
    nsamples = testenc.numel() // seqlen
    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        outputs = model.model(batch)
        logits = model.lm_head(outputs[0])
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * seqlen)
        del outputs, logits, shift_logits, loss
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    return ppl


def eval_zero_shot(model, model_path: str, tasks_csv: str, dev: torch.device, batch_size: int = 8) -> dict:
    """Evaluate zero-shot tasks via lm_eval (aligned with SpikeLLM Table 4 / lm-eval-v0.4.2)."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    ver = getattr(lm_eval, "__version__", "unknown")
    tasks = [t.strip() for t in tasks_csv.split(",") if t.strip()]
    model.eval()
    model.to(dev)
    tokenizer = load_tokenizer_safe(model_path)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    raw = lm_eval.simple_evaluate(model=hflm, tasks=tasks, batch_size=batch_size)["results"]

    per_task = {}
    acc_list = []
    norm_or_acc_list = []
    for t in tasks:
        r = raw[t]
        acc = r.get("acc,none", r.get("acc"))
        acc_norm = r.get("acc_norm,none", r.get("acc_norm"))
        chosen = acc_norm if acc_norm is not None else acc
        per_task[t] = {
            "acc": round(float(acc) * 100.0, 4) if acc is not None else None,
            "acc_norm": round(float(acc_norm) * 100.0, 4) if acc_norm is not None else None,
            "primary": round(float(chosen) * 100.0, 4) if chosen is not None else None,
        }
        if acc is not None:
            acc_list.append(float(acc) * 100.0)
        if chosen is not None:
            norm_or_acc_list.append(float(chosen) * 100.0)

    return {
        "lm_eval_version": ver,
        "tasks": per_task,
        "mean_acc": round(sum(acc_list) / len(acc_list), 4) if acc_list else None,
        "mean_primary": round(sum(norm_or_acc_list) / len(norm_or_acc_list), 4) if norm_or_acc_list else None,
    }


def extract_decoder_linear_weights(model) -> list:
    out = []
    for layer in model.model.layers:
        d = {}
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                k = name.replace(".base.module", ".module")
                d[k] = mod.weight.data.detach().cpu().clone()
        out.append(d)
    return out


def build_rotated_model_and_gptq_cache(args, trainloader, wt2_ids, c4_ids, dev, data_meta):
    """Build a single rotated model instance and compute/load cached official QuaRot-GPTQ weights."""
    transformers.set_seed(args.seed)
    model = model_utils.get_llama(args.model, hf_token=args.hf_token)
    model.seqlen = args.seqlen
    model.eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    kv_groupsize = head_dim if args.kv_groupsize == -2 else args.kv_groupsize

    fp16_check = {}
    if args.check_fp16_invariance:
        torch.cuda.reset_peak_memory_stats(dev)
        u_wt2 = eval_ppl(model, wt2_ids, dev, args.seqlen)
        u_c4 = eval_ppl(model, c4_ids, dev, args.seqlen)
        model.cpu()
        torch.cuda.empty_cache()
        fp16_check["unrotated_wt2"] = round(u_wt2, 6)
        fp16_check["unrotated_c4"] = round(u_c4, 6)
        print(f"  [FP16 unrotated] WT2={u_wt2:.4f}, C4={u_c4:.4f}", flush=True)

    qargs = types.SimpleNamespace(
        rotate=True,
        rotate_mode="hadamard",
        fp32_had=args.fp32_had,
        w_bits=args.w_bits,
        w_clip=args.w_clip,
        w_asym=args.w_asym,
        w_groupsize=-1,
        w_rtn=False,
        nsamples=args.nsamples,
        percdamp=args.percdamp,
        act_order=args.act_order,
        int8_down_proj=False,
        seed=args.seed,
    )

    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, qargs)
    quant_utils.add_actquant(model)
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had
        if "o_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = head_dim
            qlayers[name].fp32_had = args.fp32_had

    if args.check_fp16_invariance:
        r_wt2 = eval_ppl(model, wt2_ids, dev, args.seqlen)
        r_c4 = eval_ppl(model, c4_ids, dev, args.seqlen)
        model.cpu()
        torch.cuda.empty_cache()
        fp16_check["rotated_wt2"] = round(r_wt2, 6)
        fp16_check["rotated_c4"] = round(r_c4, 6)
        fp16_check["delta_wt2"] = round(abs(r_wt2 - fp16_check["unrotated_wt2"]), 6)
        print(f"  [FP16 rotated  ] WT2={r_wt2:.4f}, C4={r_c4:.4f} (|dWT2|={fp16_check['delta_wt2']:.6f})", flush=True)

    # Load or compute official spcl/QuaRot-GPTQ weights (SINGLE pass, cached for all legs!)
    cache_path = Path(args.gptq_cache) if args.gptq_cache else None
    wcache = None
    if cache_path and cache_path.exists():
        print(f"Loading cached QuaRot-GPTQ weights from {cache_path} ...", flush=True)
        wcache = torch.load(cache_path, weights_only=False)
        assert len(wcache["gptq"]) == len(model.model.layers), "Layer count mismatch in GPTQ cache!"
        if "fp16" not in wcache:
            # Backward compatibility if cache only stored gptq weights
            wcache["fp16"] = extract_decoder_linear_weights(model)
    else:
        print(f"Running official spcl/QuaRot-GPTQ (nsamples={args.nsamples}, w_bits={args.w_bits}, w_clip={args.w_clip}) ...", flush=True)
        t_gptq = time.time()
        fp16_weights = extract_decoder_linear_weights(model)
        gptq_utils.gptq_fwrd(model, trainloader, dev, qargs)
        gptq_weights = extract_decoder_linear_weights(model)
        gptq_sec = time.time() - t_gptq
        wcache = {
            "fp16": fp16_weights,
            "gptq": gptq_weights,
            "meta": {
                "model": args.model,
                "seed": args.seed,
                "nsamples": args.nsamples,
                "w_bits": args.w_bits,
                "w_clip": args.w_clip,
                "w_asym": args.w_asym,
                "calib_sha16": data_meta["calib_sha16"],
                "gptq_sec": round(gptq_sec, 1),
            },
        }
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(wcache, cache_path)
            print(f"Saved QuaRot-GPTQ cache to {cache_path} ({gptq_sec:.1f}s)", flush=True)

    # Configure activation & KV quantizers (matching official QuaRot / SpikeLLM Table 4)
    a_sym = not args.a_asym
    kv_sym = not args.kv_asym
    act_wrappers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    for name, w in act_wrappers.items():
        bits = 16 if "lm_head" in name else args.a_bits
        if "v_proj" in name and args.kv_bits < 16:
            w.out_quantizer.configure(bits=args.kv_bits, groupsize=kv_groupsize, sym=kv_sym, clip_ratio=args.kv_clip_ratio)
        w.quantizer.configure(bits=bits, groupsize=-1, sym=a_sym, clip_ratio=args.a_clip_ratio)

    if args.kv_bits < 16:
        rope_fn = model_utils.get_rope_function_name(model)
        for layer in model.model.layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn,
                rope_fn,
                config=model.config,
                k_bits=args.kv_bits,
                k_groupsize=kv_groupsize,
                k_sym=kv_sym,
                k_clip_ratio=args.kv_clip_ratio,
            )

    # Wrap all 7 decoder projections ONCE in DualBranchQuaRotWrapper
    dual_blocks = []
    for layer in model.model.layers:
        bd = {}
        for parent, attr in (
            (layer.self_attn, "q_proj"),
            (layer.self_attn, "k_proj"),
            (layer.self_attn, "v_proj"),
            (layer.self_attn, "o_proj"),
            (layer.mlp, "gate_proj"),
            (layer.mlp, "up_proj"),
            (layer.mlp, "down_proj"),
        ):
            base_w = getattr(parent, attr)
            dw = DualBranchQuaRotWrapper(
                base_w,
                low_p=0.9,
                addbit=args.addbit,
                shield_mode="mean",
                a_bits=args.a_bits,
                a_sym=a_sym,
                clip_ratio=args.a_clip_ratio,
            )
            dw.enabled_dual = False
            setattr(parent, attr, dw)
            bd[attr] = dw
        dual_blocks.append((layer, bd))

    gc.collect()
    torch.cuda.empty_cache()
    return model, dual_blocks, wcache, fp16_check


@torch.no_grad()
def capture_first_layer_inputs(model, trainloader, n_eval: int, seqlen: int, dev: torch.device):
    """Capture layer-0 inputs, attention_mask, position_ids, and optional position_embeddings (transformers 4.44+)."""
    model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.to(dev)
    model.model.layers[0].to(dev)

    base_inps = torch.zeros((n_eval, seqlen, model.config.hidden_size), dtype=torch.float16, device="cpu")
    cache = {"i": 0, "mask": None, "pos": None, "pos_emb": None}

    class _Catch(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, **kw):
            base_inps[cache["i"]] = x.squeeze(0).cpu()
            cache["i"] += 1
            cache["mask"] = kw.get("attention_mask")
            cache["pos"] = kw.get("position_ids")
            if "position_embeddings" in kw and kw["position_embeddings"] is not None:
                cache["pos_emb"] = tuple(t.detach().to(dev) for t in kw["position_embeddings"])
            raise ValueError

    model.model.layers[0] = _Catch(model.model.layers[0])
    for j in range(n_eval):
        try:
            model(trainloader[j][0].to(dev))
        except ValueError:
            pass
    model.model.layers[0] = model.model.layers[0].m.cpu()
    model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    attn_mask = cache["mask"]
    if attn_mask is not None and attn_mask.ndim == 4 and attn_mask.shape[-1] > seqlen:
        attn_mask = attn_mask[:, :, :seqlen, :seqlen]
    layer_kwargs = {"attention_mask": attn_mask, "position_ids": cache["pos"]}
    if cache["pos_emb"] is not None:
        layer_kwargs["position_embeddings"] = cache["pos_emb"]
    return base_inps, layer_kwargs


def parse_mode_spec(mode_name: str, default_tile: int):
    """Map mode preset name -> explicit decoupled configuration dict."""
    specs = {
        "pure": dict(dual=False, saliency="none", shield="off", tile=0, stage2=False, policy="none"),
        "spikellm": dict(dual=True, saliency="mean", shield="off", tile=0, stage2=False, policy="none"),
        "stage1_mean_only": dict(dual=True, saliency="mean", shield="mean", tile=default_tile, stage2=False, policy="none"),
        "stage1_tile": dict(dual=True, saliency="mean_std", shield="mean", tile=default_tile, stage2=False, policy="none"),
        "stage1_2_tile": dict(dual=True, saliency="mean_std", shield="mean", tile=default_tile, stage2=True, policy="adaptive"),
        "stage1_unaligned": dict(dual=True, saliency="mean_std", shield="mean", tile=0, stage2=False, policy="none"),
        "stage1_2_unaligned": dict(dual=True, saliency="mean_std", shield="mean", tile=0, stage2=True, policy="adaptive"),
        "stage1_2_heuristic": dict(dual=True, saliency="mean_std", shield="mean", tile=default_tile, stage2=True, policy="heuristic_down_o"),
        "stage1_2_random": dict(dual=True, saliency="mean_std", shield="mean", tile=default_tile, stage2=True, policy="random"),
    }
    if mode_name not in specs:
        raise ValueError(f"Unknown mode '{mode_name}'. Valid: {list(specs.keys())}")
    return specs[mode_name]


@torch.no_grad()
def run_single_mode(
    mode_name: str,
    low_p: float,
    model,
    dual_blocks,
    wcache,
    base_inps,
    layer_kwargs,
    wt2_ids,
    c4_ids,
    dev,
    args,
):
    spec = parse_mode_spec(mode_name, args.tile)
    t0 = time.time()
    n_eval = base_inps.shape[0]
    loss_fn = nn.MSELoss()

    # Always reload identical GPTQ weights & reset wrappers
    for li, (layer, bd) in enumerate(dual_blocks):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                k = n.replace(".base.module", ".module")
                if k in wcache["gptq"][li]:
                    m.weight.data.copy_(wcache["gptq"][li][k])
        for dw in bd.values():
            dw.base.quantizer.bits = args.a_bits
            dw.enabled_dual = False

    torch.cuda.reset_peak_memory_stats(dev)
    cal_sec = 0.0
    cal_peak_gb = 0.0
    mse_drops = []
    qdir = {k: [0, 0] for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
    total_init_macs = 0
    total_final_macs = 0
    total_swaps = 0

    if spec["dual"]:
        inps = base_inps.clone().to(dev)
        delta_rel = 1.0 / float((2 ** args.a_bits) - 1)

        for li, (layer, bd) in enumerate(dual_blocks):
            layer = layer.to(dev)

            # Step 1 (Design B): Compute TRUE W16A16 Y_ref on EXACT SAME X_l^quant
            y_ref = []
            if spec["stage2"]:
                for n, m in layer.named_modules():
                    if isinstance(m, nn.Linear):
                        k = n.replace(".base.module", ".module")
                        if k in wcache["fp16"][li]:
                            m.weight.data.copy_(wcache["fp16"][li][k].to(dev))
                for m in layer.modules():
                    if isinstance(m, quant_utils.ActQuantizer):
                        m._sb = m.bits
                        m.bits = 16
                for j in range(n_eval):
                    y_ref.append(layer(inps[j].unsqueeze(0), **layer_kwargs)[0].detach())
                # Restore GPTQ weights & KV quant bits
                for n, m in layer.named_modules():
                    if isinstance(m, nn.Linear):
                        k = n.replace(".base.module", ".module")
                        if k in wcache["gptq"][li]:
                            m.weight.data.copy_(wcache["gptq"][li][k].to(dev))
                for m in layer.modules():
                    if isinstance(m, quant_utils.ActQuantizer) and hasattr(m, "_sb"):
                        m.bits = m._sb

            # Step 2: Hessian saliency on rotated X_l^quant
            sal_sum = {k: None for k in bd}
            hooks = []

            def _mk_sal(proj_name, wrapper):
                def _h(mod, inp, out):
                    x = wrapper.apply_online_had(inp[0]).float()
                    W = wrapper.base.module.weight.data.float()
                    g = torch.mm(torch.mm(x.squeeze(0), W.t()), W).unsqueeze(0)
                    sc = g * x
                    sal_sum[proj_name] = sc if sal_sum[proj_name] is None else (sal_sum[proj_name] + sc)
                return _h

            for proj_name, dw in bd.items():
                dw.low_p = low_p
                dw.shield_mode = spec["shield"]
                dw.enabled_dual = False
                dw.base.quantizer.bits = 16
                hooks.append(dw.register_forward_hook(_mk_sal(proj_name, dw)))
            for j in range(n_eval):
                layer(inps[j].unsqueeze(0), **layer_kwargs)
            for h in hooks:
                h.remove()

            for proj_name, dw in bd.items():
                s = sal_sum[proj_name] / float(n_eval)
                if spec["saliency"] == "mean_std":
                    score = (s.mean(1) + delta_rel * s.std(1)).flatten()
                else:
                    score = s.mean(1).flatten()
                dw.saliency_score = score
                dw.enabled_dual = True

            # Shared exact initial quota computation (Stage 1 & Stage 1+2 start IDENTICAL!)
            init_quotas, min_low, max_low, probe_delta = compute_initial_quotas(bd, low_p=low_p, tile=spec["tile"])
            layer_init_macs = sum((int(bd[n].in_features) - init_quotas[n]) * int(bd[n].out_features) for n in bd)
            total_init_macs += layer_init_macs

            for proj_name, dw in bd.items():
                apply_mask_for_quota(dw, init_quotas[proj_name])

            # Step 3: Stage 2 Pairwise GQA-Unlocked Iso-MAC Block-MSE Search
            if spec["stage2"]:
                def eval_mse():
                    tot = 0.0
                    for j in range(n_eval):
                        qo = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
                        tot += loss_fn(y_ref[j], qo).item()
                    return tot / float(n_eval)

                final_q, init_m, final_m, i_macs, f_macs, n_sw = run_stage2_search(
                    bd=bd,
                    eval_mse_fn=eval_mse,
                    init_quotas=init_quotas,
                    min_low=min_low,
                    max_low=max_low,
                    probe_delta=probe_delta,
                    tile=spec["tile"],
                    policy=spec["policy"],
                    max_steps=args.max_steps,
                    seed=args.seed,
                    layer_idx=li,
                )
                total_final_macs += f_macs
                total_swaps += n_sw
                mse_drops.append((final_m - init_m) / max(init_m, 1e-12) * 100.0)
                for n in bd:
                    if final_q[n] < init_quotas[n]:
                        qdir[n][0] += 1
                    elif final_q[n] > init_quotas[n]:
                        qdir[n][1] += 1
            else:
                total_final_macs += layer_init_macs

            for dw in bd.values():
                dw.saliency_score = None

            # Step 4: Advance X_{l+1}^quant and free y_ref
            for j in range(n_eval):
                inps[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0].squeeze(0)
            del y_ref
            layer.cpu()

        cal_peak_gb = torch.cuda.max_memory_allocated(dev) / float(1024 ** 3)
        cal_sec = time.time() - t0
        del inps
        torch.cuda.empty_cache()

    # Evaluation phase (separate eval_peakVRAM tracking per Codex 012!)
    torch.cuda.reset_peak_memory_stats(dev)
    t_eval = time.time()
    wt2_ppl = eval_ppl(model, wt2_ids, dev, args.seqlen)
    c4_ppl = eval_ppl(model, c4_ids, dev, args.seqlen)
    zs_res = None
    if args.eval_tasks:
        zs_res = eval_zero_shot(model, args.model, args.eval_tasks, dev, batch_size=args.lm_eval_batch_size)
    eval_peak_gb = torch.cuda.max_memory_allocated(dev) / float(1024 ** 3)
    eval_sec = time.time() - t_eval
    model.cpu()
    torch.cuda.empty_cache()

    rec = {
        "mode": mode_name,
        "low_p": low_p,
        "spec": spec,
        "wt2_ppl": round(wt2_ppl, 6),
        "c4_ppl": round(c4_ppl, 6),
        "cal_sec": round(cal_sec, 1),
        "eval_sec": round(eval_sec, 1),
        "cal_peak_vram_gb": round(cal_peak_gb, 3),
        "eval_peak_vram_gb": round(eval_peak_gb, 3),
        "total_init_high_macs": total_init_macs,
        "total_final_high_macs": total_final_macs,
    }
    extra = ""
    if mse_drops:
        avg_d = sum(mse_drops) / len(mse_drops)
        rec["avg_block_mse_drop_pct"] = round(avg_d, 3)
        rec["total_swaps"] = total_swaps
        rec["quota_dirs"] = qdir
        q_s = ", ".join(f"{k.replace('_proj', '')}:+{v[0]}/-{v[1]}" for k, v in qdir.items())
        extra = f" | avg_MSE={avg_d:+.2f}% | swaps={total_swaps} | MACs={total_final_macs}/{total_init_macs} | dirs({q_s})"
    if zs_res is not None:
        rec["zero_shot"] = zs_res
        extra += f" | ZS(v{zs_res['lm_eval_version']})={zs_res['mean_primary']:.2f}%"

    print(
        f"  [{mode_name:18s} low_p={low_p} tile={spec['tile']:2d}] "
        f"WT2={wt2_ppl:.4f}, C4={c4_ppl:.4f} | cal={cal_sec:.1f}s eval={eval_sec:.1f}s | "
        f"calVRAM={cal_peak_gb:.2f}GB evalVRAM={eval_peak_gb:.2f}GB{extra}",
        flush=True,
    )
    return rec


def main():
    parser = argparse.ArgumentParser(description="Official spcl/QuaRot-GPTQ + OSDSQ v2 Runner")
    parser.add_argument("--model", type=str, required=True, help="Local path or HF model ID")
    parser.add_argument("--net", type=str, default=None, help="Short model tag for cache naming")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Dataset cache directory (MUST be model-vocab isolated!)")
    parser.add_argument("--gptq_cache", type=str, default=None, help="Path to save/load cached QuaRot-GPTQ weights (.pt)")
    parser.add_argument("--output_json", type=str, default=None, help="Path to write full JSON results")
    parser.add_argument("--cal_dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--c4_eval_mode", type=str, default="spikellm", choices=["spikellm", "quarot_new"])
    parser.add_argument("--nsamples", type=int, default=128, help="GPTQ calibration samples")
    parser.add_argument("--n_eval", type=int, default=16, help="Stage 1/2 calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--hf_token", type=str, default=None)

    # Official QuaRot quantization flags
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--w_clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_asym", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--act_order", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fp32_had", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--a_asym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--a_clip_ratio", type=float, default=1.0)
    parser.add_argument("--kv_bits", type=int, default=4)
    parser.add_argument("--kv_asym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kv_groupsize", type=int, default=-2, help="-2 = auto head_dim (128 for 7B/8B, 64 for 1.1B)")
    parser.add_argument("--kv_clip_ratio", type=float, default=1.0)

    # OSDSQ v2 dual-branch & Stage 2 flags
    parser.add_argument("--addbit", type=int, default=1)
    parser.add_argument("--low_p_list", type=str, default="0.9", help="Comma-separated low_p values, e.g., '0.9,0.8'")
    parser.add_argument("--modes", type=str, default="pure,spikellm,stage1_tile,stage1_2_tile")
    parser.add_argument("--tile", type=int, default=32, help="Hardware tile size (locked at 32 for Round 14)")
    parser.add_argument("--max_steps", type=int, default=8, help="Max greedy swaps per block in Stage 2")
    parser.add_argument("--check_fp16_invariance", action="store_true")

    # Optional zero-shot evaluation
    parser.add_argument("--eval_tasks", type=str, default="", help="Comma-separated lm_eval tasks (e.g. piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande)")
    parser.add_argument("--lm_eval_batch_size", type=int, default=8)
    args = parser.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== OSDSQ v2 Official spcl/QuaRot-GPTQ Runner | model={args.model} | seed={args.seed} | tile={args.tile} ===", flush=True)

    trainloader, wt2_ids, c4_ids, data_meta = load_calib_and_eval_data(args)
    print(f"Data meta: {json.dumps(data_meta)}", flush=True)

    model, dual_blocks, wcache, fp16_check = build_rotated_model_and_gptq_cache(
        args, trainloader, wt2_ids, c4_ids, dev, data_meta
    )
    base_inps, layer_kwargs = capture_first_layer_inputs(model, trainloader, args.n_eval, args.seqlen, dev)

    low_ps = [float(x.strip()) for x in args.low_p_list.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    results = []
    if "pure" in modes:
        results.append(
            run_single_mode("pure", low_ps[0], model, dual_blocks, wcache, base_inps, layer_kwargs, wt2_ids, c4_ids, dev, args)
        )
    dual_modes = [m for m in modes if m != "pure"]
    for lp in low_ps:
        print(f"\n--- low_p={lp} (high-precision 2-step GIF channel ratio = {int(round((1.0 - lp) * 100))}%) ---", flush=True)
        for m in dual_modes:
            results.append(
                run_single_mode(m, lp, model, dual_blocks, wcache, base_inps, layer_kwargs, wt2_ids, c4_ids, dev, args)
            )

    summary = {
        "args": vars(args),
        "data_meta": data_meta,
        "fp16_invariance": fp16_check,
        "gptq_meta": wcache.get("meta", {}),
        "results": results,
    }
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved JSON summary to {out_p}", flush=True)


if __name__ == "__main__":
    main()
