"""Read local model/text snapshots, compare two-step encodings and print counts."""

import math
import os
import argparse
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_CACHE_DISABLE"] = "1"

import torch
from transformers import LlamaForCausalLM


def affine_codes(x):
    smallest = x.amin(-1, keepdim=True)
    scale = ((x.amax(-1, keepdim=True) - smallest) / 31).clamp(1e-5, 1e4)
    zero = (-smallest / scale).round()
    return ((x / scale).round() + zero).clamp(0, 31), zero


def packet_count(codes):
    return (codes > 0).int() + (codes > 15).int()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["68m", "1.1b"], default="68m")
    parser.add_argument("--eval-text", choices=["wiki", "c4"], default="wiki")
    args = parser.parse_args()
    torch.set_num_threads(4)
    project = Path(__file__).resolve().parents[2] / "SpikeLLM"
    hub = Path.home() / ".cache/huggingface/hub"
    if args.model == "68m":
        snapshot = hub / "models--JackFram--llama-68m/snapshots/9de84537b6aa98de634ad0fbb1608e9d6a019355"
        cache_name = "llama"
        load_options = {}
    else:
        snapshots = hub / "models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots"
        snapshot = next(snapshots.glob("*/model.safetensors")).parent
        cache_name = "TinyLlama"
        load_options = {"torch_dtype": torch.float16, "device_map": {"": 0},
                        "low_cpu_mem_usage": True, "attn_implementation": "eager"}
    print("snapshot", snapshot.name, "model", args.model, "eval", args.eval_text)
    model = LlamaForCausalLM.from_pretrained(str(snapshot), local_files_only=True, **load_options).eval().model
    samples = torch.load(project / f"cache/dataloader_{cache_name}_wikitext2_128.cache",
                         weights_only=True, map_location="cpu")
    modules = {name: mod for name, mod in model.named_modules() if name.endswith("_proj")}
    positions = 256
    calibration = range(0, 8)
    held_out = range(64, 72)
    scores = {}
    masks = {}
    normal_calib = {}
    reverse_calib = {}
    channel_calib = {}
    totals = {}
    phase = "mask"

    def hook_for(name):
        def hook(mod, inputs, unused):
            x = inputs[0][0].float()
            if phase == "mask":
                w = mod.weight.float()
                h = ((x @ w.T) @ w) * x
                if name not in scores:
                    scores[name] = torch.zeros_like(h)
                scores[name].add_(h)
                return
            code, zero = affine_codes(x)
            high = code[:, masks[name]]
            q = high.clamp(max=30)
            a = packet_count(q)
            b = packet_count(30 - q)
            if phase == "static":
                normal_calib[name] = normal_calib.get(name, 0) + a.sum().item()
                reverse_calib[name] = reverse_calib.get(name, 0) + b.sum().item()
                channel_calib[name] = channel_calib.get(name, 0) + (a - b).sum(0)
                return
            flip = zero > 15
            c = torch.where(flip, 30 - q, q)
            p1 = c.clamp(max=15)
            p2 = c - p1
            assert p2.min() >= 0 and p2.max() <= 15
            recovered = torch.where(flip, (30 - zero) - (p1 + p2), p1 + p2 - zero)
            mismatch = (recovered != q - zero).sum().item()
            signed_magnitude = ((q - zero).abs() / 15).ceil()
            oracle_events = torch.minimum(a.sum(-1), b.sum(-1))
            # An arbitrary shared integer origin can move to the closest
            # feasible end, since packet_count is monotone in the sent code.
            left = q.amin(-1, keepdim=True)
            right = q.amax(-1, keepdim=True)
            movable_origin = torch.minimum(packet_count(q - left).sum(-1),
                                           packet_count(right - q).sum(-1))
            raw_left = high.amin(-1, keepdim=True)
            raw_right = high.amax(-1, keepdim=True)
            fits = raw_right - raw_left <= 30
            recovered_top = ((high == 31) & fits).sum().item()
            # Keep all proxy codes when the group fits. On the rest keep
            # the old q exactly; both groups then use reachable values.
            target = torch.where(fits, high, q)
            left_target = target.amin(-1, keepdim=True)
            right_target = target.amax(-1, keepdim=True)
            up = target - left_target
            down = right_target - target
            reverse = packet_count(down).sum(-1, keepdim=True) < packet_count(up).sum(-1, keepdim=True)
            sent = torch.where(reverse, down, up)
            assert sent.max() <= 30 and sent.min() >= 0
            reconstructed = torch.where(reverse, right_target - sent, left_target + sent)
            assert torch.equal(reconstructed, target)
            counts = (q.numel(), (high == 31).sum().item(), a.sum().item(), b.sum().item(),
                      (b if reverse_calib[name] < normal_calib[name] else a).sum().item(),
                      packet_count(c).sum().item(), mismatch,
                      torch.where(channel_calib[name] > 0, b, a).sum().item(),
                      oracle_events.sum().item(), signed_magnitude.sum().item(),
                      (signed_magnitude > 2).sum().item(), movable_origin.sum().item(),
                      fits.numel(), fits.sum().item(), recovered_top, packet_count(sent).sum().item())
            if name not in totals:
                totals[name] = [0] * len(counts)
            for i, value in enumerate(counts):
                totals[name][i] += value
        return hook

    handles = [mod.register_forward_hook(hook_for(name)) for name, mod in modules.items()]
    def run(indices):
        for sample_id in indices:
            tokens = samples[sample_id][0][:, :positions].to(model.embed_tokens.weight.device)
            model(tokens, use_cache=False)
    run(calibration)
    for name, h in scores.items():
        h /= len(calibration)
        saliency = h.mean(0) + h.std(0) / 15
        threshold = saliency.kthvalue(math.floor(0.9 * len(saliency))).values
        masks[name] = saliency > threshold
    scores.clear()
    phase = "static"
    run(calibration)
    phase = "heldout"
    if args.eval_text == "wiki":
        run(held_out)
    else:
        c4 = torch.load(project / f"cache/testloader_{cache_name}_c4_all.cache",
                        weights_only=True, map_location="cpu")
        for index in range(8):
            start = index * 2048
            tokens = c4[:, start:start + positions].to(model.embed_tokens.weight.device)
            model(tokens, use_cache=False)
    for handle in handles:
        handle.remove()
    print("module slots top31 forward reverse static zero_rule integer_mismatches"
          " fixed_channel best_token signed_events signed_over_two best_shared_origin"
          " groups groups_fit rescued_31 alternative_packets")
    for key, counts in totals.items():
        print(key, *counts)
    print("sum", *map(sum, zip(*totals.values())))
    print("fanout_sum", *[sum(totals[k][i] * modules[k].out_features for k in totals)
                          for i in range(len(next(iter(totals.values()))))])


if __name__ == "__main__":
    main()
