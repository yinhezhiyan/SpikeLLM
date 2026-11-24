import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import json
import time
import torch
import torch.nn as nn
from argparse import Namespace
from lm_eval import evaluator
from models.LMClass import LMClass
from models.spike_llama_layer import QuantLlamaDecoderLayer
from spike_driven_quant.spike_omniquant import static
from spike_driven_quant.utils import set_quant_state, smooth_and_quant_inplace, register_scales_and_zeros, omni_state_dict


def make_args(let=True, orig_spikellm=False, legacy_multibit_bug=False, wbits=4, abits=4):
    return Namespace(
        model='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        net='TinyLlama-1.1B-intermediate-step-1431k-3T', model_family='TinyLlama', batch_size=1,
        wbits=wbits, abits=abits, group_size=None, alpha=0.5, let=let, lwc=True, aug_loss=True,
        symmetric=False, disable_zero_point=False, a_dynamic_method='per_token', w_dynamic_method='per_channel',
        addbit=1, low_p=0.9, orig_spikellm=orig_spikellm, attn_implementation='eager',
        ablate_orig_saliency=False, ablate_no_kv_shield=False, ablate_orig_loss=False,
        legacy_multibit_bug=legacy_multibit_bug,
        weight_quant_params={'n_bits':wbits,'per_channel_axes':[0],'symmetric':False,'dynamic_method':'per_channel','group_size':None,'lwc':True,'disable_zero_point':False},
        act_quant_params={'n_bits':abits,'per_channel_axes':[],'symmetric':False,'dynamic_method':'per_token','addbit':1,'low_p':0.9,'orig_spikellm':orig_spikellm,'legacy_multibit_bug':legacy_multibit_bug},
        q_quant_params={'n_bits':abits,'per_channel_axes':[],'symmetric':False,'dynamic_method':'per_token'},
        k_quant_params={'n_bits':abits,'per_channel_axes':[],'symmetric':False,'dynamic_method':'per_token','addbit':1,'low_p':0.9,'orig_spikellm':orig_spikellm},
        v_quant_params={'n_bits':abits,'per_channel_axes':[],'symmetric':False,'dynamic_method':'per_token','addbit':1,'low_p':0.9,'orig_spikellm':orig_spikellm},
        p_quant_params={'n_bits':16,'metric':'fix0to1'},
    )


@torch.no_grad()
def build_quant_model(ckpt_path, let=True, orig_spikellm=False, sim_mode='fake_binary_simulate', legacy_multibit_bug=False):
    args = make_args(let=let, orig_spikellm=orig_spikellm, legacy_multibit_bug=legacy_multibit_bug)
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    lm.model.config.use_cache = False

    if ckpt_path is None:
        # FP16 model
        lm.model = lm.model.cuda()
        return lm

    omni_p = torch.load(ckpt_path)
    has_mask = any('mask_low_mse' in k for k in omni_p[0].keys())

    act_scales = torch.load('./act_scales/TinyLlama-1.1B-intermediate-step-1431k-3T.pt') if let else None
    act_shifts = torch.load('./act_shifts/TinyLlama-1.1B-intermediate-step-1431k-3T.pt') if let else None
    pairs = {"q_proj": "qkv", "o_proj": "out", "up_proj": "fc1"}

    if not has_mask:
        print(f"Pre-caching mask_low_mse into {ckpt_path} ...", flush=True)
        dataloader = torch.load('./cache/dataloader_TinyLlama_wikitext2_128.cache', weights_only=False)
        ns = 8
        h = torch.zeros((ns, 2048, 2048), dtype=torch.float16, device='cuda')
        emb = lm.model.model.embed_tokens.cuda()
        for j in range(ns):
            h[j] = emb(dataloader[j][0].cuda())
        pos = torch.arange(2048, device='cuda').unsqueeze(0)
        mask = torch.triu(torch.full((1, 1, 2048, 2048), torch.finfo(torch.float16).min, device='cuda', dtype=torch.float16), diagonal=1)

    for i in range(22):
        layer = lm.model.model.layers[i].cuda()
        qlayer = QuantLlamaDecoderLayer(lm.model.config, layer, args).cuda()
        if not has_mask:
            set_quant_state(qlayer, False, False)
            with torch.cuda.amp.autocast():
                static(qlayer, ns, h, mask, pos)
        set_quant_state(qlayer, False, True)
        qlayer.let = let
        if let:
            qlayer.register_parameter("qkt_smooth_scale", nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features, device='cuda', dtype=torch.float16)))
            for name, module in qlayer.named_modules():
                if hasattr(module, "in_features"):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"model.layers.{i}.{name}"].to(device='cuda', dtype=torch.float16).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)
                            shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift", nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale", nn.Parameter(scale))

        submods = dict(qlayer.named_modules())
        for k, v in omni_p[i].items():
            if 'mask_low_mse' in k:
                submods[k.rsplit('.mask_low_mse', 1)[0]].mask_low_mse = v.cuda()
        qlayer.load_state_dict(omni_p[i], strict=False)
        qlayer.half()
        smooth_and_quant_inplace(qlayer, args, True)
        if not has_mask:
            with torch.cuda.amp.autocast():
                for j in range(ns):
                    h[j] = qlayer(h[j].unsqueeze(0), attention_mask=mask, position_ids=pos)[0]
            omni_p[i] = omni_state_dict(qlayer)
        register_scales_and_zeros(qlayer)
        lm.model.model.layers[i] = qlayer

    if not has_mask:
        torch.save(omni_p, ckpt_path)
        print(f"Saved mask_low_mse into {ckpt_path}!", flush=True)

    lm.model = lm.model.cuda()
    lm.model.eval()
    for name, module in lm.model.named_modules():
        if hasattr(module, "mode"):
            module.mode = sim_mode
        if hasattr(module, "legacy_multibit_bug"):
            module.legacy_multibit_bug = legacy_multibit_bug
    return lm


@torch.no_grad()
def eval_ppl_fast(lm):
    res = {}
    for ds in ['wikitext2', 'c4']:
        testloader = torch.load(f'./cache/testloader_TinyLlama_{ds}_all.cache', weights_only=False)
        testenc = testloader if ds == 'c4' else testloader.input_ids
        nsamples = 256 if ds == 'c4' else (testenc.numel() // 2048)
        nlls = []
        for i in range(nsamples):
            batch = testenc[:, i*2048:(i+1)*2048].cuda()
            out = lm.model.model(batch)[0]
            logits = lm.model.lm_head(out)
            loss = nn.CrossEntropyLoss()(logits[:, :-1, :].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            nlls.append(loss.float() * 2048)
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * 2048)).item()
        res[ds] = round(ppl, 4)
        print(f"  {ds}: {ppl:.4f}", flush=True)
    return res


def main():
    # Part 1: P3 Multi-step Spike Simulation on 1.1B
    p3_path = "./log/p3_1.1b_multibit_summary.json"
    p3_results = {}
    if os.path.exists(p3_path):
        with open(p3_path, "r", encoding="utf-8") as f:
            p3_results = json.load(f)

    p3_configs = [
        ("1.1b_osdsq_v2_let_fake_multibit", "./log/1.1b_osdsq_with_gqa_let/omni_parameters.pth", True, False, "fake_multibit_simulate", False),
        ("1.1b_osdsq_v2_let_real_multibit_legacy_bug", "./log/1.1b_osdsq_with_gqa_let/omni_parameters.pth", True, False, "real_multibit_simulate", True),
        ("1.1b_osdsq_v2_let_real_multibit_fixed", "./log/1.1b_osdsq_with_gqa_let/omni_parameters.pth", True, False, "real_multibit_simulate", False),
        ("1.1b_orig_let_real_multibit_legacy_bug", "./log/1.1b_orig_with_gqa_let/omni_parameters.pth", True, True, "real_multibit_simulate", True),
        ("1.1b_orig_let_real_multibit_fixed", "./log/1.1b_orig_with_gqa_let/omni_parameters.pth", True, True, "real_multibit_simulate", False),
    ]

    for tag, ckpt, let, orig, mode, bug in p3_configs:
        if tag in p3_results:
            print(f"[SKIP P3] {tag}: {p3_results[tag]}", flush=True)
            continue
        print(f"=== [P3] Evaluating {tag} ===", flush=True)
        t0 = time.time()
        lm = build_quant_model(ckpt, let=let, orig_spikellm=orig, sim_mode=mode, legacy_multibit_bug=bug)
        ppls = eval_ppl_fast(lm)
        ppls["elapsed_sec"] = round(time.time() - t0, 2)
        p3_results[tag] = ppls
        with open(p3_path, "w", encoding="utf-8") as f:
            json.dump(p3_results, f, indent=2)
        del lm
        torch.cuda.empty_cache()

    # Part 2: P2 Zero-Shot Reasoning Evaluation on 1.1B (6 Tasks)
    p2_path = "./log/p2_1.1b_zeroshot_summary.json"
    p2_results = {}
    if os.path.exists(p2_path):
        with open(p2_path, "r", encoding="utf-8") as f:
            p2_results = json.load(f)

    p2_configs = [
        ("fp16", None, False, False),
        ("orig_no_let", "./log/1.1b_orig_no_let/omni_parameters.pth", False, True),
        ("orig_with_gqa_let", "./log/1.1b_orig_with_gqa_let/omni_parameters.pth", True, True),
        ("osdsq_v2_no_let", "./log/1.1b_osdsq_v2_no_let/omni_parameters.pth", False, False),
        ("osdsq_v2_with_gqa_let", "./log/1.1b_osdsq_with_gqa_let/omni_parameters.pth", True, False),
    ]

    for tag, ckpt, let, orig in p2_configs:
        if tag in p2_results and len(p2_results[tag].get("tasks", {})) >= 6:
            print(f"[SKIP P2] {tag}: {p2_results[tag]}", flush=True)
            continue
        print(f"=== [P2 Zero-Shot] Evaluating {tag} ===", flush=True)
        t0 = time.time()
        lm = build_quant_model(ckpt, let=let, orig_spikellm=orig, sim_mode="fake_binary_simulate", legacy_multibit_bug=False)
        lm.batch_size_per_gpu = 8
        res_5 = evaluator.simple_evaluate(lm, tasks="piqa,arc_easy,arc_challenge,boolq,winogrande", num_fewshot=0, limit=None)
        res_hs = evaluator.simple_evaluate(lm, tasks="hellaswag", num_fewshot=0, limit=1000)
        task_accs = {}
        for k, v in res_5["results"].items():
            task_accs[k] = round(float(v.get("acc_norm", v["acc"])) * 100, 2)
        for k, v in res_hs["results"].items():
            task_accs[k] = round(float(v.get("acc_norm", v["acc"])) * 100, 2)
        avg_acc = round(sum(task_accs.values()) / len(task_accs), 2)
        p2_results[tag] = {
            "tasks": task_accs,
            "avg_acc": avg_acc,
            "elapsed_sec": round(time.time() - t0, 2)
        }
        with open(p2_path, "w", encoding="utf-8") as f:
            json.dump(p2_results, f, indent=2)
        print(f"=== [P2 Finished] {tag}: Avg={avg_acc}% | {task_accs} ===", flush=True)
        del lm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
