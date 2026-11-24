import torch
import torch.nn as nn
from models.spike_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from spike_driven_quant.spike_linear import SpikeQuantLinear
from spike_driven_quant.spike_matmul import SpikeQuantMatMul
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from spike_driven_quant.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, SpikeQuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def find_layers(module, layers=[SpikeQuantLinear, SpikeQuantMatMul], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def static(layer, nsamples, inps, attention_mask, position_ids): 
    print("Starting ...")

    samples = nsamples
    subset = find_layers(layer)

    def add_batch(name):
        def tmp(_, inp, out):
            subset[name].add_batch(inp[0], out.data)
        return tmp

    handles = []
    res = None
    for name in subset:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    for j in range(samples):
        res = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
    for h in handles:
        h.remove()
    
    for name in subset:
        subset[name].static()
    del subset


def allocate_layer_quota(qlayer, args, is_llama, quant_inps, attention_mask, position_ids, logger=None):
    """Stage 2: Iso-MAC block-reconstruction-guided quota reallocation across linear projections."""
    linears = {
        name: m for name, m in qlayer.named_modules()
        if isinstance(m, SpikeQuantLinear) and hasattr(m, "saliency_score")
    }
    if len(linears) < 2:
        return

    max_O = max(m.out_features for m in linears.values())
    min_C = min(m.in_features for m in linears.values())
    unit_ch_at_maxO = max(1, int(round(0.02 * min_C)))
    unit_macs = unit_ch_at_maxO * max_O

    quotas = {}
    dk_map = {}
    min_low = {}
    max_low = {}
    init_high_macs = 0
    for name, m in linears.items():
        C = m.in_features
        O = m.out_features
        base_low = int(m.low_p * C)
        quotas[name] = base_low
        init_high_macs += (C - base_low) * O
        dk_map[name] = max(1, int(round(unit_macs / O)))
        min_low[name] = max(1, int(0.80 * C))
        max_low[name] = min(C - 1, int(0.98 * C))

    loss_func = torch.nn.MSELoss()
    n_eval = min(16, quant_inps.shape[0])

    set_quant_state(qlayer, weight_quant=False, act_quant=False)
    fp_targets = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for j in range(n_eval):
            fp_targets.append(
                qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0].detach()
            )
    set_quant_state(qlayer, weight_quant=False, act_quant=True)

    def apply_quota(name, low_q):
        m = linears[name]
        score = m.saliency_score
        thresh = torch.sort(score.flatten())[0][low_q - 1]
        m.mask_low_mse = (score <= thresh)

    def eval_loss():
        losses = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            smooth_and_quant_temporary(qlayer, args, is_llama)
            for j in range(n_eval):
                q_out = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                losses.append(loss_func(fp_targets[j], q_out).item())
        clear_temp_variable(qlayer)
        return float(sum(losses) / len(losses))

    cur_loss = eval_loss()
    init_loss = cur_loss

    for _ in range(6):
        d_more_high = {}
        d_less_high = {}
        for name, m in linears.items():
            dk = dk_map[name]
            q0 = quotas[name]
            if q0 - dk >= min_low[name]:
                apply_quota(name, q0 - dk)
                d_more_high[name] = (eval_loss() - cur_loss) / (dk * m.out_features)
            if q0 + dk <= max_low[name]:
                apply_quota(name, q0 + dk)
                d_less_high[name] = (eval_loss() - cur_loss) / (dk * m.out_features)
            apply_quota(name, q0)

        best_pair = None
        best_rate = 0.0
        for rec, rate_m in d_more_high.items():
            for don, rate_l in d_less_high.items():
                if rec == don:
                    continue
                if rate_m + rate_l < best_rate - 1e-12:
                    best_rate = rate_m + rate_l
                    best_pair = (rec, don)

        if best_pair is None:
            break
        rec, don = best_pair
        O_rec = linears[rec].out_features
        O_don = linears[don].out_features
        dk_don = dk_map[don]
        freed_macs = dk_don * O_don
        dk_rec = freed_macs // O_rec
        if dk_rec < 1 or quotas[rec] - dk_rec < min_low[rec]:
            break
        used_macs = dk_rec * O_rec
        actual_dk_don = int(math.ceil(used_macs / O_don))
        if quotas[don] + actual_dk_don > max_low[don]:
            break

        quotas[rec] -= dk_rec
        quotas[don] += actual_dk_don
        apply_quota(rec, quotas[rec])
        apply_quota(don, quotas[don])
        new_loss = eval_loss()
        if new_loss >= cur_loss - 1e-6:
            quotas[rec] += dk_rec
            quotas[don] -= actual_dk_don
            apply_quota(rec, quotas[rec])
            apply_quota(don, quotas[don])
            break
        cur_loss = new_loss

    for m in linears.values():
        del m.saliency_score

    final_high_macs = sum((linears[n].in_features - quotas[n]) * linears[n].out_features for n in linears)
    if logger is not None:
        q_str = ", ".join(
            f"{n.split('.')[-1]}:{quotas[n]}/{linears[n].in_features}({quotas[n]/linears[n].in_features:.3f})"
            for n in linears
        )
        logger.info(
            f"[adaptive_quota] block MSE {init_loss:.6f} -> {cur_loss:.6f} ({(cur_loss-init_loss)/init_loss*100:+.2f}%) | "
            f"high_MACs {init_high_macs} -> {final_high_macs} ({(final_high_macs-init_high_macs)/init_high_macs*100:+.2f}%) | {q_str}"
        )


def spike_omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # transformers>=4.42 keeps Llama rotary inv_freq on the model root.
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        has_saved_mask = args.resume and (i in omni_parameters) and any("mask" in k for k in omni_parameters[i].keys())
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

                    static(qlayer, args.nsamples, quant_inps, attention_mask, position_ids) ##############################
        elif not has_saved_mask:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    static(qlayer, args.nsamples, quant_inps, attention_mask, position_ids)


        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, SpikeQuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            for k, v in omni_parameters[i].items():
                if "mask_low_mse" in k:
                    mod_name = k.rsplit(".mask_low_mse", 1)[0]
                    submod = dict(qlayer.named_modules())[mod_name]
                    submod.mask_low_mse = v.to(dev)
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if getattr(args, "adaptive_quota", False) and not getattr(args, "orig_spikellm", False) and not has_saved_mask:
            with torch.no_grad():
                qlayer.float()
            allocate_layer_quota(qlayer, args, is_llama, quant_inps, attention_mask, position_ids, logger)

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    # try:
                        index = j * args.batch_size
                        # obtain output of quantization model
                        with traincast():
                            smooth_and_quant_temporary(qlayer, args, is_llama)
                            quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            fp_t1 = fp_inps[index:index+args.batch_size,]
                            loss = loss_func(fp_t1, quant_out)
                            if args.aug_loss:
                                fp_t2 = fp_inps_2[index:index+args.batch_size,]
                                if getattr(args, "orig_spikellm", False) or getattr(args, "ablate_orig_loss", False):
                                    loss += loss_func(fp_t2, quant_out)
                                else:
                                    cos_reg = (1.0 - torch.nn.functional.cosine_similarity(fp_t1.float(), quant_out.float(), dim=-1)).mean()
                                    w_main = 1.0 - (1.0 / (2 ** (2 * args.addbit)))
                                    w_cos = 1.0 / (2 ** (args.abits + args.addbit) - 1)
                                    loss = w_main * loss + (1.0 - w_main) * loss_func(fp_t2, quant_out) + w_cos * cos_reg
                        if not math.isfinite(loss.item()):
                            logger.info("Loss is NAN, stopping training")
                            pdb.set_trace()
                            
                        loss_list.append(loss.detach().cpu())
                        optimizer.zero_grad()
                        norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                        norm_list.append(norm.data)
                    # except:
                    #     print("########### one false ###########")
                    #     pass

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs > 0 or (args.resume and not has_saved_mask):
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                # with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            if args.epochs > 0:
                torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
            elif args.resume:
                torch.save(omni_parameters, args.resume)
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.SpikeQuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.SpikeQuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = False
    return model
