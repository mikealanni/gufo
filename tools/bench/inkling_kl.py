#!/usr/bin/env python3
"""KL and top-1 agreement between two `gufo_inkling_gpu_check --windows` logit files.

usage: inkling_kl.py base.bin test.bin

Each file holds the last-token logits (float32, vocabulary 200058) of every window.
Reports the mean and maximum KL(base || test) in nats and how many windows keep
the same top-1 token. Use 48 windows of 1,024 tokens: six are too noisy to rank
settings.
"""
import sys

import numpy as np

VOCAB = 200058


def main() -> None:
    a = np.fromfile(sys.argv[1], dtype=np.float32)
    b = np.fromfile(sys.argv[2], dtype=np.float32)
    n = len(a) // VOCAB
    a = a[: n * VOCAB].reshape(n, VOCAB).astype(np.float64)
    b = b[: n * VOCAB].reshape(n, VOCAB).astype(np.float64)
    la = a - a.max(1, keepdims=True)
    la -= np.log(np.exp(la).sum(1, keepdims=True))
    lb = b - b.max(1, keepdims=True)
    lb -= np.log(np.exp(lb).sum(1, keepdims=True))
    kl = (np.exp(la) * (la - lb)).sum(1)
    top = int((a.argmax(1) == b.argmax(1)).sum())
    print(f"windows {n}  mean KL {kl.mean():.3e}  max KL {kl.max():.3e}  top1 {top}/{n}")


if __name__ == "__main__":
    main()
