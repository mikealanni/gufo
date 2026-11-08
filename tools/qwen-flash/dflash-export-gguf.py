#!/usr/bin/env python3
"""Repack a fine-tuned DFlash2 drafter into a Gufo-loadable GGUF.

Reuses the original drafter GGUF verbatim for metadata, tensor order and dtype
(every matrix stays Q8_0, every vector stays F32) and only rewrites the tensor
data with the trained weights. Because types and counts are unchanged the data
offsets are identical, so this is an in-place data substitution.

  python dflash-export-gguf.py --template .../Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf \
      --weights draft-epoch2.safetensors --out new-drafter.gguf

The drafter shares the target embedding and output head, which the template
already omits; they are untouched here. Validate the result with dflash-tune.py
(greedy byte-equality is mandatory).
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from safetensors.torch import load_file

SCALARS = {0: "<f4", 30: "<u2"}


def quantize_q8(values: np.ndarray) -> bytes:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size % 32:
        raise ValueError("Q8_0 tensor is not a multiple of 32 elements")
    blocks = flat.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    scale = np.where(amax == 0.0, np.float32(1e-30), amax).astype(np.float32) / -128.0
    codes = np.clip(np.rint(blocks / scale[:, None]), -128, 127).astype(np.int8)
    scales = scale.astype(np.float16).view(np.uint8).reshape(-1, 2)
    return np.concatenate([scales, codes.view(np.uint8)], axis=1).reshape(-1).tobytes()


def gguf_name_map():
    mapping = {
        "fc.weight": "fc.weight",
        "enc.output_norm.weight": "hidden_norm.weight",
        "output_norm.weight": "norm.weight",
        "selector_predecessor.weight": "candidate_selector.predecessor_codebook.weight",
        "selector_successor.weight": "candidate_selector.successor_codebook.weight",
        "selector_hidden.weight": "candidate_selector.hidden_projection.weight",
    }
    suffix = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "attn_conv_proj.weight": "attention_conv.kernel_projection.weight",
        "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight",
        "attn_conv_base": "attention_conv.base_kernel",
        "ffn_conv_base": "mlp_conv.base_kernel",
    }
    for index in range(8):
        for gguf_key, torch_key in suffix.items():
            mapping[f"blk.{index}.{gguf_key}"] = f"layers.{index}.{torch_key}"
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True,
                        help="original drafter GGUF (metadata + layout source)")
    parser.add_argument("--weights", required=True,
                        help="fine-tuned drafter safetensors")
    parser.add_argument("--out", required=True, help="output GGUF path")
    parser.add_argument("--dry-run", action="store_true",
                        help="check sizes/coverage and report, write nothing")
    args = parser.parse_args()

    template = pathlib.Path(args.template)
    raw = bytearray(template.read_bytes())

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "codec", pathlib.Path(__file__).resolve().parents[1] / "gufo" / "gguf.py")
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    state = codec.parse_gguf(template)
    data_offset = state["data_offset"]

    weights = load_file(args.weights)
    weights = {
        (key + ".weight" if key.endswith("_codebook") else key):
            value.float().cpu().numpy()
        for key, value in weights.items()
    }
    names = gguf_name_map()

    replaced = 0
    carried = 0
    for info in state["tensors"]:
        name = info["name"]
        expected = int(np.prod(info["shape"]))
        source = weights.get(names.get(name, ""))
        if source is None:
            carried += 1
            continue
        if int(np.prod(source.shape)) != expected:
            raise SystemExit(f"{name}: trained shape {source.shape} != gguf "
                             f"{info['shape']} ({expected} elements)")
        start = data_offset + int(info["offset"])
        if info["type"] == 8:
            payload = quantize_q8(source)
        elif info["type"] in SCALARS:
            payload = np.ascontiguousarray(
                source.reshape(info["shape"]).astype(np.float32)).tobytes() \
                if info["type"] == 0 else \
                source.astype(np.float16).tobytes()
        else:
            raise SystemExit(f"{name}: unsupported target type {info['type_name']}")
        if len(payload) != codec.tensor_bytes(state, info):
            raise SystemExit(f"{name}: payload {len(payload)} != "
                             f"{codec.tensor_bytes(state, info)} bytes")
        if not args.dry_run:
            raw[start:start + len(payload)] = payload
        replaced += 1

    missing = [info["name"] for info in state["tensors"]
               if names.get(info["name"], "") not in weights]
    print(f"replaced {replaced}, carried {carried}, "
          f"unmapped tensors {len(missing)}")
    if missing:
        print("unmapped:", missing[:12], "..." if len(missing) > 12 else "")
    if args.dry_run:
        return
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(raw))
    print(f"wrote {out} ({len(raw)} bytes)")


if __name__ == "__main__":
    main()
