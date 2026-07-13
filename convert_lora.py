#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


SAT_PREFIX = "model.diffusion_model."


class ModuleParser:
    def __init__(self, key: str):
        self.key = key
        self.modules = key.split(".")
        self.idx = 0

    def match(self, pattern: str) -> bool:
        parts = pattern.split(".")
        if self.modules[self.idx : self.idx + len(parts)] != parts:
            return False
        self.idx += len(parts)
        return True

    def step(self) -> str:
        value = self.modules[self.idx]
        self.idx += 1
        return value

    def eof(self) -> bool:
        return self.idx == len(self.modules)


def _load_module_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for name in ("module", "state_dict", "model"):
            state_dict = checkpoint.get(name)
            if isinstance(state_dict, dict):
                return state_dict
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(f"Could not find a tensor state dict in {path}")


def _sat_module_to_wan_prefix(key: str) -> str:
    parser = ModuleParser(key)
    if not parser.match("model.diffusion_model.transformer.layers"):
        raise ValueError(f"Unsupported LoRA key: {key}")

    block_idx = parser.step()
    
    modules = ["diffusion_model", "blocks", block_idx]

    if parser.match("attention"):
        modules.append("self_attn")
        if parser.match("query_key_value.lora_layer"):
            qkv_idx = parser.step()
            qkv_map = {"0": "q", "1": "k", "2": "v"}
            if qkv_idx not in qkv_map:
                raise ValueError(f"Unsupported QKV LoRA index in {key}")
            modules.append(qkv_map[qkv_idx])
        elif parser.match("dense.lora_layer"):
            modules.append("o")
        else:
            raise ValueError(f"Unsupported self-attention LoRA key: {key}")
    elif parser.match("cross_attention"):
        modules.append("cross_attn")
        if parser.match("query.lora_layer"):
            modules.append("q")
        elif parser.match("key_value.lora_layer"):
            kv_idx = parser.step()
            kv_map = {"0": "k", "1": "v"}
            if kv_idx not in kv_map:
                raise ValueError(f"Unsupported cross-attention KV LoRA index in {key}")
            modules.append(kv_map[kv_idx])
        elif parser.match("dense.lora_layer"):
            modules.append("o")
        else:
            raise ValueError(f"Unsupported cross-attention LoRA key: {key}")
    elif parser.match("mlp"):
        modules.append("ffn")
        if parser.match("dense_h_to_4h.lora_layer"):
            modules.append("0")
        elif parser.match("dense_4h_to_h.lora_layer"):
            modules.append("2")
        else:
            raise ValueError(f"Unsupported MLP LoRA key: {key}")
    else:
        raise ValueError(f"Unsupported transformer LoRA key: {key}")

    if not parser.eof():
        raise ValueError(f"Unparsed LoRA key suffix in {key}")
    return ".".join(modules)


def convert_key(key: str) -> str:
    if key.endswith(".down.weight"):
        sat_module = key[: -len(".down.weight")]
        suffix = ".lora_down.weight"
    elif key.endswith(".up.weight"):
        sat_module = key[: -len(".up.weight")]
        suffix = ".lora_up.weight"
    elif key.endswith(".diff_b"):
        sat_module = key[: -len(".diff_b")]
        suffix = ".diff_b"
    else:
        raise ValueError(f"Unsupported LoRA tensor key: {key}")

    return _sat_module_to_wan_prefix(sat_module) + suffix


def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        if ".lora_layer" not in key:
            continue
        new_key = convert_key(key)
        if new_key in converted:
            raise ValueError(f"Duplicate converted key {new_key} from {key}")
        tensor = value.detach().cpu().contiguous()
        if dtype is not None:
            tensor = tensor.to(dtype)
        converted[new_key] = tensor
    if not converted:
        raise ValueError("No LoRA tensors were found in the input checkpoint")
    return converted


def _parse_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "keep":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a SAT-format SCAIL-2 LoRA checkpoint to Wan fuse_lora / ComfyUI safetensors."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="SAT LoRA checkpoint path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output safetensors path.",
    )
    parser.add_argument(
        "--dtype",
        default="keep",
        choices=["keep", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Optional dtype conversion for saved LoRA tensors. Defaults to keep.",
    )
    parser.add_argument(
        "--print-sample",
        type=int,
        default=8,
        help="Print this many converted keys for inspection.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    dtype = _parse_dtype(args.dtype)

    print(f"Loading SAT LoRA checkpoint from {input_path}...")
    state_dict = _load_module_state(input_path)
    converted = convert_state_dict(state_dict, dtype=dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(converted)} tensors to {output_path}...")
    save_file(converted, str(output_path))

    if args.print_sample > 0:
        print("Sample converted keys:")
        for key in list(converted.keys())[: args.print_sample]:
            value = converted[key]
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print("Done.")


if __name__ == "__main__":
    main()
