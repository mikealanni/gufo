import argparse, importlib.util, pathlib, random, sys, time, json
import numpy as np, torch, torch.nn.functional as F
from safetensors.torch import save_file
ROOT = pathlib.Path.home() / "gufo/tools/qwen-flash"
spec = importlib.util.spec_from_file_location("ft", ROOT / "dflash-finetune.py")
ft = importlib.util.module_from_spec(spec); sys.path.insert(0, str(ROOT)); spec.loader.exec_module(ft)
ap = argparse.ArgumentParser()
ap.add_argument("--train"); ap.add_argument("--held"); ap.add_argument("--out")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--sel-lr", type=float, default=5e-5)
ap.add_argument("--context", type=int, default=512); ap.add_argument("--wps", type=int, default=32)
ap.add_argument("--resume"); ap.add_argument("--held2"); ap.add_argument("--ocw", type=int, default=4)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--eval-per", type=int, default=24)
a = ap.parse_args()
H = pathlib.Path.home()
dev, dt = torch.device("cuda"), torch.float32
torch.set_float32_matmul_precision("high")
model, cfg, miss, unexp = ft.build_model(ROOT / "vendor/dflash2_model_pinned.py", H / "models/flash-next-dflash2/config.json", H / "models/flash-next-dflash2/model.safetensors", dev, dt)
print("missing", miss, "unexpected", unexp)
if a.resume:
    from safetensors.torch import load_file
    print("resume", model.load_state_dict({k: v.float() for k, v in load_file(a.resume).items()}, strict=False))
model.to(device=dev, dtype=dt)
head_gguf = pathlib.Path("/mnt/raid0/user_cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/83cadfda58d30be06c110518208d1bb918b33f10/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf")
head = ft.read_tensor(head_gguf, "output.weight").to(dev, dt); embed = ft.read_tensor(head_gguf, "token_embd.weight").to(dev, dt)
block = int(cfg.dflash_config["block_size"]); mask_id = int(cfg.dflash_config["mask_token_id"]); V = int(cfg.vocab_size)
DV, TB = 163840, 248044
blocked = torch.zeros(V, device=dev); blocked[DV:] = -1e9; blocked[TB:] = 0.0
sel = model.candidate_selector
import os
A_, L_, C_ = int(os.environ.get("AOFF", 0)), int(os.environ.get("LOFF", 1)), int(os.environ.get("COFF", 0))
def window(tokens, feats, start, history):
    e = start + C_
    raw = feats[e - history:e]
    ctx = torch.from_numpy(ft.bf16_to_f32(raw).copy()).to(dev, dt).unsqueeze(0)
    noise = torch.tensor([int(tokens[start + A_])] + [mask_id] * (block - 1), device=dev)
    labels = torch.tensor([int(tokens[start + L_ + o - 1]) for o in range(1, block)], device=dev)
    pos = torch.arange(e - history, e + block, device=dev).unsqueeze(0)
    return ctx, noise, labels, pos, int(tokens[start + A_])
def run(ctx, noise, pos):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(target_hidden=ctx, noise_embedding=embed[noise].unsqueeze(0), position_ids=pos)
    return h[:, 1:, :].to(dt)
def unary(h):
    return model.compute_logits(h, lambda x: x @ head.t()) + blocked
def train_loss(h, labels, anchor):
    u = unary(h)[0]
    prev = torch.cat([torch.tensor([anchor], device=dev), labels[:-1]])
    pair = (sel.predecessor_codebook(prev) * sel.hidden_projection(h[0])) @ sel.successor_codebook.weight.t()
    valid = (labels < DV) | (labels >= TB)
    return F.cross_entropy(u + pair, labels.masked_fill(~valid, -100), ignore_index=-100), u
@torch.no_grad()
def evaluate(shards):
    model.eval(); rng = np.random.default_rng(123); tot = []; unary_tot = []
    for _, tokens, feats, rows, tf in shards:
        for _ in range(a.eval_per):
            lo = max(tf, 64)
            if rows <= lo + block + 1: break
            s = int(rng.integers(lo, rows - block)); hist = min(a.context, s)
            ctx, noise, labels, pos, anc = window(tokens, feats, s, hist)
            h = run(ctx, noise, pos)
            path, _, _ = sel.select(h, unary(h), torch.tensor([anc], device=dev), 0.0)
            ok = (path[0] == labels).to(torch.int32).cumprod(0).sum().item()
            tot.append(ok + 1)
            ok2 = (unary(h)[0].argmax(-1) == labels).to(torch.int32).cumprod(0).sum().item()
            unary_tot.append(ok2 + 1)
    model.train()
    return float(np.mean(tot)), float(np.mean(unary_tot)), len(tot)
held = [(n,t,f,r,64) for n,t,f,r,_ in ft.shards_from(pathlib.Path(a.held))]; train = [(n,t,f,r,64) for n,t,f,r,_ in ft.shards_from(pathlib.Path(a.train))]
held2 = [(n,t,f,r,64) for n,t,f,r,_ in ft.shards_from(pathlib.Path(a.held2))] if a.held2 else []
print(len(train), "train shards", len(held), "held shards", len(held2), "held2", flush=True)
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
m, mu, n = evaluate(held); best = m
print(f"BASELINE held-out accepted/cycle (selector path) {m:.3f}  unary-only {mu:.3f}  n={n}", flush=True)
if held2:
    m2, _, n2 = evaluate(held2); print(f"BASELINE2 held2 {m2:.3f} n={n2}", flush=True)
selp = [p for n_, p in model.named_parameters() if n_.startswith("candidate_selector")]
selids = {id(p) for p in selp}
rest = [p for p in model.parameters() if id(p) not in selids]
opt = torch.optim.AdamW([{"params": rest, "lr": a.lr}, {"params": selp, "lr": a.sel_lr}], betas=(0.9, 0.98), weight_decay=0.0)
model.train(); step = 0; t0 = time.time(); rl = 0.0; ra = 0.0; rc = 0
for ep in range(a.epochs):
    order = list(range(len(train))); random.shuffle(order); rng = np.random.default_rng(ep)
    for i in order:
        _, tokens, feats, rows, tf = train[i]; lo = max(tf, 8)
        if rows <= lo + block + 1: continue
        for _ in range(a.wps * (a.ocw if "/oc" in train[i][0] else 1)):
            s = int(rng.integers(lo, rows - block)); hist = min(a.context, s)
            ctx, noise, labels, pos, anc = window(tokens, feats, s, hist)
            h = run(ctx, noise, pos)
            loss, u = train_loss(h, labels, anc)
            (loss / a.bs).backward(); rl += float(loss.detach()); rc += 1
            if rc >= a.bs:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 100 == 0: print(f"ep {ep} step {step} loss {rl/rc:.3f} {step/(time.time()-t0):.2f} step/s", flush=True)
                rl = 0.0; rc = 0
    m, mu, n = evaluate(held)
    if held2:
        m2, _, n2 = evaluate(held2); print(f"EPOCH {ep} held2 {m2:.3f} n={n2}", flush=True)
    print(f"EPOCH {ep} held-out accepted/cycle {m:.3f} unary-only {mu:.3f} (baseline {best if ep==0 else best:.3f}) n={n}", flush=True)
    save_file({k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in model.state_dict().items()}, str(out / f"draft-ep{ep}.safetensors"))
print("done", time.time() - t0)
