import os
import json
import csv
import argparse
import torch
from argparse import Namespace
from transformers import AutoModelForCausalLM
from models.spike_llama_layer import QuantLlamaDecoderLayer
from spike_driven_quant.spike_omniquant import static, find_layers
from spike_driven_quant.spike_linear import SpikeQuantLinear

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--low_p", type=float, default=0.9)
    parser.add_argument("--output_prefix", type=str, default=None)
    cmd_args = parser.parse_args()

    model_name = cmd_args.model
    net = model_name.split("/")[-1]
    model_family = net.split("-")[0]
    cache_path = f"./cache/dataloader_{model_family}_wikitext2_128.cache"
    dataloader = torch.load(cache_path, weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    dev = "cuda"
    nsamples = cmd_args.nsamples
    seqlen = 2048

    args = Namespace(
        wbits=4, abits=4, group_size=None, lwc=False, symmetric=False, disable_zero_point=False,
        a_dynamic_method="per_token", w_dynamic_method="per_channel", addbit=1, low_p=cmd_args.low_p, orig_spikellm=False,
        weight_quant_params={"n_bits": 4, "per_channel_axes": [0], "symmetric": False, "dynamic_method": "per_channel", "group_size": None, "lwc": False, "disable_zero_point": False},
        act_quant_params={"n_bits": 4, "per_channel_axes": [], "symmetric": False, "dynamic_method": "per_token", "addbit": 1, "low_p": cmd_args.low_p, "orig_spikellm": False},
        q_quant_params={"n_bits": 4, "per_channel_axes": [], "symmetric": False, "dynamic_method": "per_token"},
        k_quant_params={"n_bits": 4, "per_channel_axes": [], "symmetric": False, "dynamic_method": "per_token", "addbit": 1, "low_p": cmd_args.low_p, "orig_spikellm": False},
        v_quant_params={"n_bits": 4, "per_channel_axes": [], "symmetric": False, "dynamic_method": "per_token", "addbit": 1, "low_p": cmd_args.low_p, "orig_spikellm": False},
        p_quant_params={"n_bits": 16, "metric": "fix0to1"},
    )

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers = model.model.layers
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=torch.float16, device=dev)
    cache = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0].to(dev))
    for batch in dataloader:
        if cache["i"] >= nsamples:
            break
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print(f"{'Layer':>5} | {'r = R_all/R_low':>15} | {'r^2 = (R_all/R_low)^2':>21}")
    print("-" * 49)
    all_r = []
    all_r2 = []
    layer_records = []
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        qlayer = QuantLlamaDecoderLayer(model.config, layer, args).to(dev)
        with torch.cuda.amp.autocast():
            static(qlayer, nsamples, inps, attention_mask, position_ids)
        subset = find_layers(qlayer, layers=[SpikeQuantLinear])
        layer_ratios = []
        def make_hook(mod):
            def hook(_, inp, out):
                x = inp[0].float()
                mask_low = mod.mask_low_mse
                r_all = (x.amax(dim=-1) - x.amin(dim=-1)).clamp(min=1e-6)
                low_count = mask_low.sum(dim=-1, keepdim=True).clamp(min=1)
                mean_low = (x * mask_low).sum(dim=-1, keepdim=True) / low_count
                x_low = torch.where(mask_low, x, mean_low)
                r_low = (x_low.amax(dim=-1) - x_low.amin(dim=-1)).clamp(min=1e-6)
                ratio = (r_all / r_low).mean().item()
                layer_ratios.append(ratio)
            return hook
        handles = [m.register_forward_hook(make_hook(m)) for m in subset.values()]
        with torch.cuda.amp.autocast():
            for j in range(nsamples):
                inps[j] = qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        r_mean = sum(layer_ratios) / len(layer_ratios)
        r2_mean = r_mean ** 2
        all_r.append(r_mean)
        all_r2.append(r2_mean)
        layer_records.append({"layer": i, "r_mean": round(r_mean, 4), "r2_mean": round(r2_mean, 4)})
        print(f"{i:>5d} | {r_mean:>15.4f} | {r2_mean:>21.4f}")
        layers[i] = layer.cpu()
        del qlayer, layer
        torch.cuda.empty_cache()

    overall_r = round(sum(all_r) / len(all_r), 4)
    overall_r2 = round(sum(all_r2) / len(all_r2), 4)
    print("-" * 49)
    print(f"Overall Mean r = {overall_r:.4f}, Mean r^2 = {overall_r2:.4f}")

    prefix = cmd_args.output_prefix or f"./log/hijack_ratio_{model_family.lower()}"
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "nsamples": nsamples,
            "seqlen": seqlen,
            "low_p": cmd_args.low_p,
            "overall_mean_r": overall_r,
            "overall_mean_r2": overall_r2,
            "layers": layer_records,
        }, f, indent=2)
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "r_mean", "r2_mean"])
        writer.writeheader()
        writer.writerows(layer_records)
    print(f"Saved results to {prefix}.json and {prefix}.csv")

if __name__ == "__main__":
    main()
