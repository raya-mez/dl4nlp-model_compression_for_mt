import logging
import math
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from safetensors.torch import save_file, load_file

from utils import KBitConfig, smallest_int_dtype_for_k, kbit_range
from constants import model_id_map, max_allowed_torch_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

torch.backends.cuda.matmul.allow_tf32 = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_groupwise_symmetric(W, cfg):
    out_features, in_features = W.shape
    group_size = cfg.group_size
    if in_features % group_size != 0:
        # Use Custom Group size that is compatible with in_features
        group_size = math.gcd(in_features, cfg.group_size)
        if group_size == 0:
            group_size = in_features

    groups = in_features // group_size

    Wv = W.view(out_features, groups, group_size)  # [O, groups, group_size]

    int_dtype_to_use = smallest_int_dtype_for_k(cfg.k)

    if cfg.k == 1:
        scale = Wv.abs().mean(dim=2, keepdim=True).clamp_min(torch.finfo(W.dtype).eps)
        Q = torch.sign(Wv)
        Q[Q == 0] = 1 # Convert 0 to 1 as we only consider -1 and 1
        Qc = Q.to(int_dtype_to_use).view_as(W)
        scales = scale.squeeze(2).to(cfg.f_dtype)
    else:
        qmin, qmax = kbit_range(cfg.k)

        max_abs = Wv.abs().amax(dim=2, keepdim=True)                          # [O, groups, 1]
        scale = (max_abs / max(qmax, 1)).clamp_min(torch.finfo(W.dtype).eps)
        Q = torch.round(Wv / scale).clamp(qmin, qmax)
        Qc = Q.to(int_dtype_to_use).view_as(W)
        scales = scale.squeeze(2).to(cfg.f_dtype)
    return Qc, scales


def replace_linear_with_quant(model, cfg):
    """
    Recursively replace nn.Linear layers of model with QuantLinear and quantize their float weights.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            qlin = QuantLinear(child.in_features, child.out_features, bias=(child.bias is not None), cfg=cfg)
            with torch.no_grad():
                qlin.quantize_from_float(child.weight.data)
                if child.bias is not None:
                    qlin.bias.copy_(child.bias.data.to(cfg.f_dtype))
            setattr(model, name, qlin)
        else:
            replace_linear_with_quant(child, cfg)
    return model


def save_quant_to_safetensors(model, save_dir, cast_scales_to_fp16=True, cpu_offload=True):
    save_dir = Path(save_dir)
    precision = save_dir.parts[-1].split('_')[0]

    tensors = {}
    i = 0

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            Q = module.Q_w
            S = module.scales

            if cpu_offload:
                Q = Q.detach().to("cpu", non_blocking=True)
                S = S.detach().to("cpu", non_blocking=True)
            else:
                Q = Q.detach()
                S = S.detach()

            if cast_scales_to_fp16 and S.dtype != torch.float16:
                S = S.half()

            if hasattr(module, "group_size") and module.group_size is not None:
                G = int(module.group_size)
            elif hasattr(module, "groups") and module.groups is not None:
                G = int(module.groups)
            else:
                raise AttributeError(
                    f"QuantLinear '{name}' missing group_size/groups attribute."
                )

            # Save per-layer tensors
            tensors[f"Q.{i}"] = Q
            tensors[f"S.{i}"] = S
            tensors[f"G.{i}"] = torch.tensor(G, dtype=torch.int32)
            tensors[f"in.{i}"] = torch.tensor(int(module.in_features), dtype=torch.int32)
            tensors[f"out.{i}"] = torch.tensor(int(module.out_features), dtype=torch.int32)

            i += 1

    if i == 0:
        raise RuntimeError("No QuantLinear layers found in model.")

    # Build metadata (strings only)
    meta = {"num_layers": str(i)}

    save_file(tensors, str(save_dir / f'{precision}_bit_model.safetensors'), metadata=meta)
    print(f"Saved {i} QuantLinear layers -> {save_dir}")


def replace_linear_with_quant_loading(module, cfg):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            qlin = QuantLinear(child.in_features, child.out_features, bias=(child.bias is not None), cfg=cfg)
            if child.bias is not None:
                with torch.no_grad():
                    qlin.bias.copy_(child.bias.data.to(cfg.f_dtype))
            setattr(module, name, qlin)
        else:
            replace_linear_with_quant_loading(child, cfg)
    return module


def fast_load_k_into_skeleton(model, safetensors_path):
    state_dict = load_file(safetensors_path, device="cpu")  # mmap
    num_layers = max([int(x.split('.')[-1]) for x in list(state_dict.keys())])

    i = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            Q = state_dict[f"Q.{i}"]
            S = state_dict[f"S.{i}"]
            G = int(state_dict[f"G.{i}"].item())
            module.set_quant_state(Q, S, G)
            i += 1
            if i > num_layers:
                break

    model = model.to(device, non_blocking=True)
    return model


class QuantLinear(nn.Module):
    """
    Linear Layer to hold k-bit quantized weights and dequantizes on-the-fly during forward to measure performance drop.
    """
    def __init__(self, in_features, out_features, cfg, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.cfg = cfg
        self.int_dtype_to_use = smallest_int_dtype_for_k(self.cfg.k)
        self.register_buffer("Q_w", torch.empty(out_features, in_features, dtype=self.int_dtype_to_use),
                             persistent=True)
        self.register_buffer("scales", torch.empty(out_features, 0, dtype=self.cfg.f_dtype), persistent=True)

        self.groups = None
        self.group_size = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=self.cfg.f_dtype))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def quantize_from_float(self, W_float):
        Qc, scales = quantize_groupwise_symmetric(W_float.to(self.cfg.f_dtype), self.cfg)
        self.group_size = self.cfg.group_size if (self.in_features % self.cfg.group_size == 0) else math.gcd(self.in_features, self.cfg.group_size)
        if self.group_size == 0:
            self.group_size = self.in_features
        self.groups = self.in_features // self.group_size
        self.Q_w = Qc
        self.scales = scales

    @torch.no_grad()
    def set_quant_state(self, Q_w, scales, group_size):
        out_dim, in_dim = Q_w.shape

        self.Q_w = Q_w
        self.scales = scales
        self.group_size = int(group_size)
        self.groups = in_dim // group_size

    def forward(self, x):
        # Dequantize group-wise, then matmul (simple & consistent across k) to see performance drop
        out_dim, in_dim = self.out_features, self.in_features
        group_size = self.group_size
        groups = self.groups
        x = x.to(self.cfg.f_dtype)

        # reshape weights into [O, groups, group_size], apply scales per group, then bring back to [out_dim, in_dim]
        Qw = self.Q_w.view(out_dim, groups, group_size).to(self.cfg.f_dtype)
        Sc = self.scales.view(out_dim, groups, 1).to(self.cfg.f_dtype)
        Wdq = (Qw * Sc).view(out_dim, in_dim)      # dequantized weights

        y = torch.matmul(x, Wdq.t())
        if self.bias is not None:
            y = y + self.bias
        return y


def load_model_and_tokenizer(model_id_key, precision, kbit_group_size=64, kbit_load_dir=None):
    if model_id_key != 'pseudo_quant':
        if precision == 'int8':
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            logging.info("[LOAD] BitsAndBytesConfig: load_in_8bit=True")
            model = AutoModelForCausalLM.from_pretrained(
                model_id_map[model_id_key], device_map='auto', quantization_config=bnb
            )
        elif precision == 'int4':
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
            )
            logging.info(
                "[LOAD] BitsAndBytesConfig: load_in_4bit=True, quant_type=nf4, double_quant=True, compute_dtype=float16")
            model = AutoModelForCausalLM.from_pretrained(
                model_id_map[model_id_key], device_map='auto', quantization_config=bnb
            )
        elif precision == 'fp16':
            logging.info("[LOAD] Loading FP16 weights (torch.float16)")
            model = AutoModelForCausalLM.from_pretrained(
                model_id_map[model_id_key], device_map='auto', torch_dtype=torch.float16
            )
        else:
            logging.info("[LOAD] Loading BF16 weights (torch.bfloat16)")
            model = AutoModelForCausalLM.from_pretrained(
                model_id_map[model_id_key], device_map='auto', torch_dtype=torch.bfloat16
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id_map[model_id_key], use_fast=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        cfg = KBitConfig(k=precision, group_size=kbit_group_size, f_dtype=max_allowed_torch_dtype)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id_map['tower_instruct'], device_map='auto', torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id_map['tower_instruct'], use_fast=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_skeleton = replace_linear_with_quant_loading(base_model, cfg)
        model = fast_load_k_into_skeleton(model_skeleton, kbit_load_dir)
        del model_skeleton
    return model, tokenizer


def verify_model_precision(model, requested):
    logging.info("\n[VERIFY] ===== Model Load Verification =====")
    logging.info(f"[VERIFY] Requested precision: {requested}")
    logging.info(f"[VERIFY] torch.cuda.is_available(): {torch.cuda.is_available()}")
    logging.info(f"[VERIFY] hf_device_map: {getattr(model, 'hf_device_map', None)}")
    logging.info(f"[VERIFY] is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}")
    logging.info(f"[VERIFY] is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
    try:
        import bitsandbytes as bnb
        has_4 = any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
        has_8 = any(isinstance(m, bnb.nn.Linear8bitLt) for m in model.modules())
        logging.info(f"[VERIFY] Found bnb.nn.Linear4bit layers: {has_4}")
        logging.info(f"[VERIFY] Found bnb.nn.Linear8bitLt layers: {has_8}")
    except Exception as e:
        logging.info(f"[VERIFY] bitsandbytes introspection skipped: {e}")
    try:
        name, param = next(iter(model.named_parameters()))
        logging.info(f"[VERIFY] Example param: {name} | dtype={param.dtype} | device={param.device}")
    except Exception as e:
        logging.info(f"[VERIFY] Could not inspect a parameter: {e}")
    ok = True
    if requested == "int4" and not getattr(model, "is_loaded_in_4bit", False):
        ok = False
    if requested == "int8" and not getattr(model, "is_loaded_in_8bit", False):
        ok = False
    if requested in {"fp16", "bf16"}:
        cfg_dtype = getattr(model.config, "torch_dtype", None)
        logging.info(f"[VERIFY] model.config.torch_dtype: {cfg_dtype}")
        if requested == "fp16" and cfg_dtype not in (torch.float16, None):
            ok = False
        if requested == "bf16" and cfg_dtype not in (torch.bfloat16, None):
            ok = False
    logging.info(f"[VERIFY] RESULT: {'PASS' if ok else 'MISMATCH'}")
    logging.info("[VERIFY] ====================================\n")
