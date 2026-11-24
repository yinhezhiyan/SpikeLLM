import torch, math
import torch.nn as nn
import torch.nn.functional as F
from spike_driven_quant.quantizer import UniformAffineQuantizer


class SpikeQuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        
        self.use_weight_quant = False
        self.use_act_quant = False
        
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        
        self.orig_spikellm = act_quant_params.get("orig_spikellm", False)
        self.ablate_orig_saliency = act_quant_params.get("ablate_orig_saliency", False)
        self.ablate_no_linear_shield = act_quant_params.get("ablate_no_linear_shield", False)
        self.legacy_multibit_bug = act_quant_params.get("legacy_multibit_bug", False)
        _extra_keys = {"orig_spikellm", "ablate_orig_saliency", "ablate_no_linear_shield", "ablate_no_kv_shield", "legacy_multibit_bug"}
        act_quant_params_clean = {k: v for k, v in act_quant_params.items() if k not in _extra_keys}
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params_clean)
            self.act_quantizer_high = UniformAffineQuantizer(**act_quant_params_clean, high=True)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        self.low_p = act_quant_params["low_p"]

        self.sum = None
        self.nsamples = 0

        mask_low_mse = torch.ones(self.weight.shape[1], dtype=torch.bool)
        self.register_buffer('mask_low_mse', mask_low_mse)

        self.mode = "fake_binary_simulate"
        self.steps = int(2**act_quant_params["addbit"])
        self.step_levels = 2**act_quant_params["n_bits"]
        
    @torch.no_grad()
    def add_batch(self, inp, out):
        
        self.nsamples += 1

        inp = inp.to(dtype=torch.float32)
        weight = self.weight.data.to(dtype=torch.float32)

        grad = torch.mm(inp.squeeze(0), weight.transpose(0,1))
        grad = torch.mm(grad, weight).unsqueeze(0)
        inp = grad * (inp)

        if self.sum==None:
            self.sum = inp
        else:
            self.sum += inp

    @torch.no_grad()
    def static(self,):
        self.sum /= self.nsamples
        _, mask_low_mse = self.get_channel_mask(self.sum)
        self.mask_low_mse = mask_low_mse
        del self.sum

    @torch.no_grad()
    def get_channel_mask(self, input):
        if self.orig_spikellm or self.ablate_orig_saliency:
            mean = input.mean(1).unsqueeze(1)
        else:
            # Weight temporal token-wise standard deviation by the relative quantization step size 1/(2^b_a - 1)
            delta_rel = 1.0 / (self.step_levels - 1)
            mean = (input.mean(1) + delta_rel * input.std(1)).unsqueeze(1)
        
        low_quota = int(self.low_p*input.shape[2])
        thresh = torch.sort(mean.flatten())[0][low_quota - 1]
        mask_low = mean <= thresh

        return mean, mask_low

    def _quant_low(self, inputs: torch.Tensor, mask_low: torch.Tensor):
        if self.orig_spikellm or self.ablate_no_linear_shield:
            return self.act_quantizer(inputs) * mask_low
        # Mean-anchored outlier shielding: replaces high-saliency outlier channels (~mask_low)
        # with the exact mean of the low-saliency subset (mask_low) prior to dynamic min-max calibration.
        # Because min(S) <= mean(S) <= max(S) for any non-empty set S, this preserves the exact
        # [xmin, xmax] of the low-saliency channels without zero-padding bias or inf/nan gradients.
        with torch.no_grad():
            low_count = mask_low.sum(dim=-1, keepdim=True).clamp(min=1)
            mean_low = ((inputs.float() * mask_low).sum(dim=-1, keepdim=True) / low_count).to(inputs.dtype)
        shielded_inputs = torch.where(mask_low, inputs, mean_low)
        return self.act_quantizer(shielded_inputs) * mask_low

    def forward(self, inputs: torch.Tensor):
        
        mask_low = self.mask_low_mse
        if len(inputs.shape)==2:
            mask_low = mask_low.squeeze(0)

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            '''The equivalence of quantization-SNN conversion is addressed in Appendix A.3. Binary simulate is also proved equal to quantization 
            recently by Spike-Driven Transformer-V3: Yao M, Qiu X, Hu T, et al. Scaling spike-driven transformer with efficient spike firing approximation training[J]. 
            IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.'''
            if self.training or self.mode == "fake_binary_simulate" or self.mode == "fake_quant":
                inputs = self._quant_low(inputs, mask_low) + self.act_quantizer_high(inputs) * (~mask_low)
                out = self.fwd_func(inputs, weight, bias, **self.fwd_kwargs)
                return out
            elif self.mode == "fake_multibit_simulate":
                self.act_quantizer_high.per_token_dynamic_calibration(inputs)
                inputs_int, scale, round_zero_point = self.act_quantizer_high.fake_quant_int(inputs, self.act_quantizer_high.scale, self.act_quantizer_high.round_zero_point)
                inputs_int = inputs_int * (~mask_low)
                spikes = []
                mem = torch.zeros_like(inputs_int)
                for i in range(self.steps):
                    if i == 0:
                        mem = inputs_int
                    else:
                        mem = mem - spikes[i - 1]
                    
                    spike = torch.where(mem>=self.step_levels, self.step_levels-1, mem).to(mem.dtype).to(mem.device)
                    spikes.append(spike)
                
                spikes = torch.stack(spikes,dim=0)

                x_dequant = self.act_quantizer_high.fake_dequant_int(spikes.sum(0), self.act_quantizer_high.scale, self.act_quantizer_high.round_zero_point)
                inputs_quant = x_dequant * (~mask_low) + self._quant_low(inputs, mask_low)
                out = self.fwd_func(inputs_quant, weight, bias, **self.fwd_kwargs)
                return out

            elif self.mode == "real_multibit_simulate":
                self.act_quantizer_high.per_token_dynamic_calibration(inputs)
                inputs_int, scale, round_zero_point = self.act_quantizer_high.fake_quant_int(inputs, self.act_quantizer_high.scale, self.act_quantizer_high.round_zero_point)
                inputs_int = inputs_int * (~mask_low)
                spikes = []
                mem = torch.zeros_like(inputs_int)
                for i in range(self.steps):
                    if i == 0:
                        mem = inputs_int
                    else:
                        mem = mem - spikes[i - 1]
                    
                    spike = torch.where(mem>=self.step_levels, self.step_levels-1, mem).to(mem.dtype).to(mem.device)
                    spikes.append(spike)
                
                spikes = torch.stack(spikes,dim=0)
                
                if self.legacy_multibit_bug:
                    w_sum = weight.sum(1)
                else:
                    w_sum = (weight * (~mask_low.view(1, -1))).sum(1)
                out = ((torch.matmul(spikes, weight.T).sum(0).float() - round_zero_point.float() * w_sum.float()) * scale.float()).to(inputs.dtype)
                
                out += self.fwd_func(self._quant_low(inputs, mask_low), weight, bias, **self.fwd_kwargs)
                return out
            else:
                raise NotImplementedError
        
        out = self.fwd_func(inputs, weight, bias, **self.fwd_kwargs)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
