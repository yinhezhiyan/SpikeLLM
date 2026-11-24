"""Dual-branch GIF activation wrapper on top of official spcl/QuaRot ActQuantWrapper.

Decouples:
  - shield_mode: {'off', 'mean', 'zero'} (Stage 1 Subspace-Isolated Calibration)
  - saliency_mode: {'mean', 'mean_std'} (controlled independently during calibration)
  - enabled_dual: False -> exact pure official QuaRot ActQuantWrapper behavior
"""

import math
import torch
import torch.nn as nn

import hadamard_utils
import quant_utils


class DualBranchQuaRotWrapper(nn.Module):
    """Wraps an official QuaRot ActQuantWrapper with dual-branch GIF + Stage 1/2 controls."""

    def __init__(
        self,
        base_wrapper: quant_utils.ActQuantWrapper,
        low_p: float = 0.9,
        addbit: int = 1,
        shield_mode: str = "mean",
        a_bits: int = 4,
        a_sym: bool = False,
        clip_ratio: float = 1.0,
    ):
        super().__init__()
        assert shield_mode in ("off", "mean", "zero"), f"Unknown shield_mode: {shield_mode}"
        self.base = base_wrapper
        self.in_features = base_wrapper.module.in_features
        self.out_features = base_wrapper.module.out_features
        self.low_p = low_p
        self.addbit = addbit
        self.shield_mode = shield_mode
        self.a_bits = a_bits
        self.a_sym = a_sym
        self.clip_ratio = clip_ratio

        self.q_low = quant_utils.ActQuantizer()
        self.q_low.configure(bits=a_bits, groupsize=-1, sym=a_sym, clip_ratio=clip_ratio)
        self.q_high = quant_utils.ActQuantizer()
        self.q_high.configure(bits=a_bits + addbit, groupsize=-1, sym=a_sym, clip_ratio=clip_ratio)

        self.register_buffer("mask_low", torch.ones(self.in_features, dtype=torch.bool))
        self.saliency_score = None
        self.enabled_dual = False

    def apply_online_had(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base
        x_dtype = x.dtype
        if w.online_full_had:
            if w.fp32_had:
                x = hadamard_utils.matmul_hadU_cuda(x.float(), w.had_K, w.K).to(x_dtype)
            else:
                x = hadamard_utils.matmul_hadU_cuda(x, w.had_K, w.K)
        elif w.online_partial_had:
            if w.fp32_had:
                x = x.float()
            init_shape = x.shape
            if w.K == 1:
                import fast_hadamard_transform

                x = fast_hadamard_transform.hadamard_transform(
                    x.reshape(-1, init_shape[-1] // w.had_dim, w.had_dim).transpose(1, 2),
                    scale=1.0 / math.sqrt(init_shape[-1] // w.had_dim),
                ).transpose(1, 2)
            else:
                x = (
                    w.had_K.to(x.device).to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // w.had_dim, w.had_dim)
                ) / math.sqrt(init_shape[-1] // w.had_dim)
            if w.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = self.apply_online_had(x)

        if self.enabled_dual:
            mask = self.mask_low.to(x.device)
            # Low-saliency branch (1-step 4-bit GIF)
            if self.shield_mode == "mean":
                with torch.no_grad():
                    low_cnt = mask.sum().clamp(min=1)
                    mean_low = ((x.float() * mask).sum(dim=-1, keepdim=True) / low_cnt).to(x.dtype)
                x_low_in = torch.where(mask, x, mean_low)
            elif self.shield_mode == "zero":
                x_low_in = torch.where(mask, x, torch.zeros_like(x))
            else:
                x_low_in = x

            self.q_low.find_params(x_low_in)
            x_low = self.q_low(x_low_in).to(x_dtype) * mask
            self.q_low.free()

            # High-saliency branch (2^addbit-step GIF, e.g., 2-step 5-bit at addbit=1)
            self.q_high.find_params(x)
            x_high = self.q_high(x).to(x_dtype) * (~mask)
            self.q_high.free()

            x = x_low + x_high
        elif self.base.quantizer.bits < 16:
            self.base.quantizer.find_params(x)
            x = self.base.quantizer(x).to(x_dtype)
            self.base.quantizer.free()

        x = self.base.module(x).to(x_dtype)

        if self.base.out_quantizer.bits < 16:
            self.base.out_quantizer.find_params(x)
            x = self.base.out_quantizer(x).to(x_dtype)
            self.base.out_quantizer.free()
        return x
