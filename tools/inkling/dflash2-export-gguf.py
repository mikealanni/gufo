#!/usr/bin/env python3
"""Convert an Inkling-2-Mini DFlash2 drafter (safetensors) into a Gufo GGUF.

  python dflash2-export-gguf.py --weights model.safetensors --config config.json --out drafter.gguf

Matrices of the attention output and the MLP become Q8_0, the other matrices stay
BF16, norms and the convolution base kernels are F32. The drafter shares the
target's embedding and output head, so neither is stored.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFWriter
from gguf.quants import quantize

Q8_SUFFIXES = ("attn_q.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")

TOP = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
    "norm.weight": "output_norm.weight",
    "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
    "candidate_selector.successor_codebook": "selector_successor.weight",
    "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
}
LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "attention_conv.kernel_projection.weight": "attn_conv_proj.weight",
    "attention_conv.base_kernel": "attn_conv_base",
    "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight",
    "mlp_conv.base_kernel": "ffn_conv_base",
}


def read_safetensors(path: Path):
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__":
                continue
            assert info["dtype"] == "BF16", (name, info["dtype"])
            start, end = info["data_offsets"]
            f.seek(base + start)
            raw = np.frombuffer(f.read(end - start), dtype=np.uint16).reshape(info["shape"])
            yield name, raw


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def gguf_name(name: str) -> str:
    if name in TOP:
        return TOP[name]
    _, index, suffix = name.split(".", 2)
    return f"blk.{index}.{LAYER[suffix]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vocab-unpadded", type=int, default=200058)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    df = cfg["dflash_config"]
    writer = GGUFWriter(str(args.out), "dflash")
    writer.add_name("Inkling-2-Mini-DFlash2")
    p = "dflash."
    writer.add_uint32(p + "block_count", cfg["num_hidden_layers"])
    writer.add_uint32(p + "context_length", cfg["max_position_embeddings"])
    writer.add_uint32(p + "embedding_length", cfg["hidden_size"])
    writer.add_uint32(p + "feed_forward_length", cfg["intermediate_size"])
    writer.add_uint32(p + "attention.head_count", cfg["num_attention_heads"])
    writer.add_uint32(p + "attention.head_count_kv", cfg["num_key_value_heads"])
    writer.add_uint32(p + "attention.key_length", cfg["head_dim"])
    writer.add_uint32(p + "attention.value_length", cfg["head_dim"])
    writer.add_float32(p + "rope.freq_base", float(cfg["rope_parameters"]["rope_theta"]))
    writer.add_float32(p + "attention.layer_norm_rms_epsilon", float(cfg["rms_norm_eps"]))
    writer.add_uint32(p + "attention.sliding_window", cfg["sliding_window"])
    writer.add_bool(p + "attention.causal", bool(cfg["is_causal"]))
    writer.add_uint32(p + "vocab_size", cfg["vocab_size"])
    writer.add_uint32(p + "vocab_size_unpadded", args.vocab_unpadded)
    writer.add_uint32(p + "block_size", df["block_size"])
    writer.add_uint32(p + "conv_kernel_size", df["conv_kernel_size"])
    writer.add_uint32(p + "conv_group_size", df["conv_group_size"])
    writer.add_uint32(p + "selector_rank", df["selector_rank"])
    writer.add_uint32(p + "selector_top_k", df["selector_top_k"])
    writer.add_uint32(p + "mask_token_id", df["mask_token_id"])
    writer.add_array(p + "target_layers", [int(x) for x in df["target_layer_ids"]])
    writer.add_float32(p + "final_logit_softcapping", float(df["final_logit_softcapping"]))
    writer.add_string(p + "selector_softcap_mode", df["selector_softcap_mode"])

    count = 0
    for name, raw in read_safetensors(args.weights):
        target = gguf_name(name)
        norm = target.endswith("norm.weight") or target.endswith("_base")
        if norm or "conv_base" in target:
            writer.add_tensor(target, bf16_to_f32(raw))
        elif target.endswith(Q8_SUFFIXES):
            data = bf16_to_f32(raw)
            writer.add_tensor(target, quantize(data, Q.Q8_0), raw_dtype=Q.Q8_0)
        else:
            writer.add_tensor(target, np.ascontiguousarray(raw).view(np.uint8), raw_dtype=Q.BF16)
        count += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {count} tensors to {args.out}")


if __name__ == "__main__":
    main()
