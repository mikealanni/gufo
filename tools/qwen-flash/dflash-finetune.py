#!/usr/bin/env python3
"""Fine-tune the DFlash2 drafter for Qwen3.8 Flash-Next by target distillation.

Reads feature shards produced by tools/qwen-flash/dflash-capture.sh (the same
[taps][hidden] residual-mean features the live drafter consumes) plus the frozen
target embedding/output head from the target GGUF, then teacher-forces the
drafter's diffusion block on captured greedy continuations with a masked
cross-entropy loss. That is what raises acceptance: the drafter learns this
target's next-token distribution instead of the upstream one.

The drafter is ~420 MB in bf16 and the frozen head adds ~1.3 GB, so one process
fits with room for the agent running alongside. Bound peak memory with
--context and --blocks. Verify with dflash-tune.py afterward and ship only a
drafter whose greedy output stays byte-identical on the regression prompts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gufo"))
import gguf as codec  # noqa: E402


def build_model(model_py: pathlib.Path, config_path: pathlib.Path, weights_path,
                device, dtype=torch.float32):
    spec = importlib.util.spec_from_file_location("dflash2", model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.Qwen3Config(**json.loads(config_path.read_text()))
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = module.DFlash2DraftModel(config)
    state = load_file(str(weights_path))
    cleaned = {
        (key + ".weight" if key.startswith("candidate_selector.") and
         key.endswith("codebook") else key): value.to(dtype)
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False, assign=True)
    try:
        model.rotary_emb = module.Qwen3RotaryEmbedding(config, device=device)
    except TypeError:
        model.rotary_emb = module.Qwen3RotaryEmbedding(config)
        model.rotary_emb = model.rotary_emb.to(device)
    return model, config, missing, unexpected


def read_tensor(path: pathlib.Path, name: str) -> torch.Tensor:
    state = codec.parse_gguf(path)
    tensors = {t["name"]: t for t in state["tensors"]}
    if name not in tensors:
        raise SystemExit(f"{name} is not in {path}")
    info = tensors[name]
    if info["type"] not in codec.DEQUANT:
        raise SystemExit(f"dequant not supported for {name} ({info['type']})")
    nbytes = codec.tensor_bytes(state, info)
    raw = np.memmap(path, mode="r", dtype=np.uint8,
                   offset=state["data_offset"] + info["offset"], shape=(nbytes,))
    values = codec.DEQUANT[info["type"]](raw).reshape(info["shape"])
    return torch.from_numpy(np.array(values, dtype=np.float32))


def bf16_to_f32(raw):
    return (np.asarray(raw, dtype=np.uint32) << 16).view(np.float32)


def shards_from(directory: pathlib.Path):
    for meta in sorted(directory.glob("*.meta.json")):
        stem = meta.name[: -len(".meta.json")]
        token_path = meta.with_name(stem + ".tokens.i32")
        feature_path = meta.with_name(stem + ".features.f16")
        header = json.loads(meta.read_text())
        rows = int(header["rows"])
        taps = len(header["taps"])
        hidden = int(header["hidden"])
        if rows < 16 or not token_path.exists() or not feature_path.exists():
            continue
        tokens = np.memmap(token_path, dtype="<i4", mode="r", shape=(rows,))
        train_from = 0
        if not header.get("chat", True):
            starts = np.nonzero(np.asarray(tokens) == int(tokens[0]))[0]
            train_from = int(starts[-1]) + 3
        features = np.memmap(feature_path, dtype="<u2", mode="r",
                             shape=(rows, taps * hidden))
        yield str(meta), tokens, features, rows, train_from


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="pinned dflash model.py")
    parser.add_argument("--config", required=True, help="drafter config.json")
    parser.add_argument("--weights", help="initial safetensors (default: config dir)")
    parser.add_argument("--head-gguf", required=True,
                        help="target shard with output.weight and token_embd.weight")
    parser.add_argument("--data", required=True, help="capture output directory")
    parser.add_argument("--out", required=True, help="checkpoint output directory")
    parser.add_argument("--resume", help="continue from a safetensors checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context", type=int, default=512,
                        help="feature rows of history per block (bounds memory)")
    parser.add_argument("--blocks", type=int, default=4,
                        help="draft blocks per optimizer step (gradient accumulation)")
    parser.add_argument("--steps", type=int, default=0,
                        help="stop after N optimizer steps (0 = all epochs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draft-vocab", type=int, default=163840)
    parser.add_argument("--tail-base", type=int, default=248044)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--force", action="store_true",
                        help="run even if a gufo serve is detected")
    parser.add_argument("--cpu", action="store_true",
                        help="run on CPU")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default=None,
                        help="model/optimizer dtype (default fp32; bf16 parameters round "
                             "small updates away)")
    parser.add_argument("--only", default="",
                        help="comma list of substrings; only matching parameters train "
                             "(low-memory subset fine-tune, e.g. selector_hidden,fc)")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--dry-run", action="store_true",
                        help="load everything, run one step on CPU-free shape check")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu" if (args.dry_run or args.cpu) else "cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if device.type == "cuda" and not args.force:
        import subprocess
        running = subprocess.run(["pgrep", "-af", "gufo"],
                                 capture_output=True, text=True)
        if running.returncode == 0 and any("gufo serve" in line and "bench" not in line
                                           for line in running.stdout.splitlines()):
            raise SystemExit("refusing to train: a gufo serve is running and holds the "
                             "GPU model. Stop it first (or pass --force).")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_path = pathlib.Path(args.config)
    weights = pathlib.Path(args.weights) if args.weights else \
        config_path.with_name("model.safetensors")

    model, config, missing, unexpected = build_model(
        pathlib.Path(args.upstream), config_path, weights, device, dtype)
    if missing:
        print("missing drafter tensors:", missing)
    if unexpected:
        print("unexpected drafter tensors:", unexpected)
    if args.resume:
        model.load_state_dict(load_file(args.resume), strict=False, assign=True)
    model.to(device=device, dtype=dtype)
    model.train()

    block = int(config.dflash_config["block_size"])
    mask_id = int(config.dflash_config["mask_token_id"])
    vocab = int(config.vocab_size)

    def load_head(name):
        tensor = read_tensor(pathlib.Path(args.head_gguf), name)
        tensor = tensor.to(device=device, dtype=dtype)
        tensor.requires_grad_(False)
        return tensor

    head = load_head("output.weight")
    embed = load_head("token_embd.weight")

    if args.only:
        wanted = [part.strip() for part in args.only.split(",") if part.strip()]
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(any(part in name for part in wanted))
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(dtype)

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"training {len(trainable)} tensors: "
          f"{trainable[:8]}{' ...' if len(trainable) > 8 else ''}")

    blocked_columns = torch.zeros(vocab, device=device, dtype=torch.float32)
    blocked_columns[args.draft_vocab:] = -1e9
    if args.tail_base > args.draft_vocab:
        blocked_columns[args.tail_base:] = 0.0

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                                   weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr,
                                     betas=(args.beta1, args.beta2),
                                     weight_decay=args.weight_decay)

    shards = list(shards_from(pathlib.Path(args.data)))
    if not shards:
        raise SystemExit(f"no capture shards under {args.data}")
    print(f"{len(shards)} shards; context={args.context} blocks/step={args.blocks} "
          f"block={block} vocab={vocab}")

    def window(rows, tokens, features, generator, train_from):
        low = max(train_from, 8)
        if rows <= low + block + 1:
            return None
        start = int(generator.integers(low, rows - block))
        history = min(args.context, start)
        raw = features[start - history:start]
        ctx = torch.from_numpy(bf16_to_f32(raw).copy()).to(
            device=device, dtype=head.dtype).unsqueeze(0)
        anchor = int(tokens[start])
        noise_ids = torch.tensor([anchor] + [mask_id] * (block - 1),
                                 dtype=torch.long, device=device)
        labels = torch.tensor(
            [int(tokens[start + offset]) for offset in range(1, block)],
            dtype=torch.long, device=device)
        first = start - history
        positions = torch.arange(first, first + history + block,
                                 device=device).unsqueeze(0)
        return ctx, noise_ids, labels, positions

    step = 0
    started = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(shards)))
        random.shuffle(order)
        generator = np.random.default_rng(args.seed + epoch)
        running_loss = running_accept = 0.0
        running_count = 0
        optimizer.zero_grad(set_to_none=True)
        for index in order:
            name, tokens, features, rows, train_from = shards[index]
            for _ in range(max(1, args.blocks)):
                sample = window(rows, tokens, features, generator, train_from)
                if sample is None:
                    continue
                ctx, noise_ids, labels, positions = sample
                noise = embed[noise_ids].unsqueeze(0)
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hidden = model(target_hidden=ctx, noise_embedding=noise,
                                       position_ids=positions)
                else:
                    hidden = model(target_hidden=ctx, noise_embedding=noise,
                                   position_ids=positions)
                logits = (hidden[:, 1:, :].to(head.dtype) @ head.t()).to(torch.float32)
                logits = logits[0] + blocked_columns
                valid = (labels < args.draft_vocab) | (labels >= args.tail_base)
                if not bool(valid.any()):
                    continue
                loss = torch.nn.functional.cross_entropy(
                    logits, labels.masked_fill(~valid, -100), ignore_index=-100)
                (loss / args.blocks).backward()
                running_loss += float(loss.detach())
                with torch.no_grad():
                    accepted = logits.argmax(-1) == labels
                    running_accept += float(
                        accepted.to(torch.int32).cumprod(0).sum().item())
                running_count += 1
                if running_count >= args.blocks:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    mean = running_loss / running_count
                    accept = running_accept / running_count
                    if step % args.log_every == 0:
                        print(f"epoch {epoch} step {step} loss {mean:.4f} "
                              f"accept-len {accept:.2f}/{block - 1} "
                              f"grad {float(grad_norm):.2f} "
                              f"{step / (time.time() - started):.2f} step/s",
                              flush=True)
                    running_loss = running_accept = 0.0
                    running_count = 0
                    if args.steps and step >= args.steps:
                        break
            if args.steps and step >= args.steps:
                break
        checkpoint = {key: value.detach().to(torch.bfloat16).contiguous().cpu()
                      for key, value in model.state_dict().items()}
        save_file(checkpoint, str(out / f"draft-epoch{epoch}.safetensors"))
        print(f"saved {out / f'draft-epoch{epoch}.safetensors'} at step {step}",
              flush=True)
        if args.dry_run:
            break
    print(f"done: {step} steps in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
