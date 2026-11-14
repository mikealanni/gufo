import argparse
import os
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith, memref
from aie.extras import types as T
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_

COLS, ROWS = 8, 4
SE = 4608  # stripe object: two 128-column head stripes
TE = 5120  # table object: rope table + norm weights + eps
PIECE = 1536
TILE = 18432  # one 128x128 C-wire tile
NT = int(os.environ.get("PREP_NT", "8"))  # 128-token tiles per dispatch (<= 8, one QKV GEMM result)
PASS = int(os.environ.get("PREP_PASS", str(8 * 98304)))  # byte stride between 128-token pass buffers of the Q output
PCOL = 98304
QKV_BYTES = 8 * 3 * 32 * TILE  # 1024 rows x 3072 cols, kM = 1024
WIN = 4
MODE = os.environ.get("PREP_MODE", "kv")
KV_WIN_PAD = 2


def main():
    ap = argparse.ArgumentParser()
    for a in ("-M", "-K", "-N", "-m", "-k", "-n"):
        ap.add_argument(a, type=int, default=0)
    args = ap.parse_args()
    with mlir_mod_ctx() as ctx:
        design(args.m, args.k, args.n)
        print(ctx.module)


def design(m, k, n):
    @device(AIEDevice.npu2)
    def device_body():
        S_l2_ty = np.ndarray[(ROWS * SE,), np.dtype[np.uint8]]
        S_l1_ty = np.ndarray[(SE,), np.dtype[np.uint8]]
        T_ty = np.ndarray[(TE,), np.dtype[np.uint8]]
        C_l1_ty = np.ndarray[(PIECE,), np.dtype[np.uint8]]
        C_l2_ty = np.ndarray[(ROWS * PIECE,), np.dtype[np.uint8]]
        lw = f"mm_{m}x{k}x{n}.o"
        prep_q2 = external_func("prep_q2", inputs=[S_l1_ty, T_ty, np.int32], link_with=lw)
        prep_kv2 = external_func("prep_kv2", inputs=[S_l1_ty, T_ty], link_with=lw)
        prep_emit = external_func("prep_emit", inputs=[C_l1_ty, np.int32, np.int32], link_with=lw)
        prep_mbegin = external_func("prep_mbegin", inputs=[], link_with=lw)
        prep_kh = external_func("prep_kh", inputs=[S_l1_ty, T_ty, np.int32], link_with=lw)
        prep_mfin = external_func("prep_mfin", inputs=[T_ty, np.int32], link_with=lw)

        tiles = [[tile(col, row) for col in range(COLS)] for row in range(ROWS + 2)]
        shim_tiles, mem_tiles, core_tiles = tiles[0], tiles[1], tiles[2:]

        S_l3l2 = [None] * COLS
        S_l2l1 = [[None] * COLS for _ in range(ROWS)]
        T_f = [None] * COLS
        C_l1l2 = [[None] * COLS for _ in range(ROWS)]
        C_l2l3 = [None] * COLS
        for col in range(COLS):
            S_l3l2[col] = object_fifo(f"S_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, S_l2_ty)
            for row in range(ROWS):
                S_l2l1[row][col] = object_fifo(f"S_L2L1_{col}_{row}", mem_tiles[col], core_tiles[row][col], 2, S_l1_ty)
            object_fifo_link(S_l3l2[col], [S_l2l1[row][col] for row in range(ROWS)], [], [SE * row for row in range(ROWS)])
            T_f[col] = object_fifo(f"T_{col}", shim_tiles[col], [core_tiles[row][col] for row in range(ROWS)], 2, T_ty)
            for row in range(ROWS):
                C_l1l2[row][col] = object_fifo(f"C_L1L2_{col}_{row}", core_tiles[row][col], mem_tiles[col], 2, C_l1_ty)
            C_l2l3[col] = object_fifo(f"C_L2L3_{col}", mem_tiles[col], shim_tiles[col], 2, C_l2_ty)
            object_fifo_link([C_l1l2[row][col] for row in range(ROWS)], C_l2l3[col], [PIECE * row for row in range(ROWS)], [])

        rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]
        rtps = [[None] * COLS for _ in range(ROWS)]
        rtp_locks = [[None] * COLS for _ in range(ROWS)]
        for row in range(ROWS):
            for col in range(COLS):
                rtps[row][col] = buffer(core_tiles[row][col], rtp_ty, name=f"rtp_{row}_{col}", use_write_rtp=True)
                rtp_locks[row][col] = lock(core_tiles[row][col], init=0, sym_name=f"rtp_lock_{row}_{col}")

        for row in range(ROWS):
            for col in range(COLS):

                @core(core_tiles[row][col], stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        use_lock(rtp_locks[row][col], LockAction.Acquire, value=1)
                        nt_i32 = memref.load(rtps[row][col], [arith.constant(T.index(), 0)])
                        nq_i32 = memref.load(rtps[row][col], [arith.constant(T.index(), 1)])
                        use_lock(rtp_locks[row][col], LockAction.Release, value=1)
                        nt = arith.index_cast(T.index(), nt_i32)
                        nq = arith.index_cast(T.index(), nq_i32)
                        for _ in range_(nt):
                            t = T_f[col].acquire(ObjectFifoPort.Consume, 1)
                            s = S_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                            prep_kv2(s, t)
                            S_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            for pr in range_(2):
                                s = S_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                                prep_q2(s, t, arith.index_cast(T.i32(), pr))
                                S_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            T_f[col].release(ObjectFifoPort.Consume, 1)
                            for kind, cnt in ((0, 3), (1, 6), (2, 6), (3, 1)):
                                for u in range_(cnt):
                                    o = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                                    prep_emit(o, arith.constant(T.i32(), kind), arith.index_cast(T.i32(), u))
                                    C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)
                        for _ in range_(nq):
                            t = T_f[col].acquire(ObjectFifoPort.Consume, 1)
                            prep_mbegin()
                            for pr in range_(2):
                                s = S_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                                prep_q2(s, t, arith.index_cast(T.i32(), pr))
                                S_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            for hi in range_(4):
                                s = S_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                                prep_kh(s, t, arith.index_cast(T.i32(), hi))
                                S_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            prep_mfin(t, arith.constant(T.i32(), row))
                            T_f[col].release(ObjectFifoPort.Consume, 1)
                            for kind, cnt in ((4, 1), (1, 6)):
                                for u in range_(cnt):
                                    o = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                                    prep_emit(o, arith.constant(T.i32(), kind), arith.index_cast(T.i32(), u))
                                    C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        TAB_BYTES = NT * 8 * TE
        PASS_BYTES = (NT - 1) * PASS + 8 * PCOL
        KVW = (NT * 8 + KV_WIN_PAD) * 18432
        KN_BYTES = NT * 8 * 6144
        KV0 = 18432

        @runtime_sequence(
            np.ndarray[(QKV_BYTES,), np.dtype[np.uint8]],
            np.ndarray[(TAB_BYTES,), np.dtype[np.uint8]],
            np.ndarray[(PASS_BYTES,), np.dtype[np.uint8]],
            np.ndarray[(KVW,), np.dtype[np.uint8]],
            np.ndarray[(max(KN_BYTES, KV0),), np.dtype[np.uint8]],
        )
        def sequence(QKV, TAB, PASSB, KVB, AUX):
            def lin(full, offset, sizes, strides):
                return TensorAccessPattern((1, full), offset, sizes, strides)

            kvmode = MODE == "kv"
            for row in range(ROWS):
                for col in range(COLS):
                    npu_rtp_write(f"rtp_{row}_{col}", 0, NT if kvmode else 0)
                    npu_rtp_write(f"rtp_{row}_{col}", 1, 0 if kvmode else NT)
                    set_lock_value(rtp_locks[row][col], 1)
            bufs = {"qkv": QKV, "tab": TAB, "pass": PASSB, "kv": KVB, "aux": AUX}
            full = {"qkv": QKV_BYTES, "tab": TAB_BYTES, "pass": PASS_BYTES, "kv": KVW, "aux": max(KN_BYTES, KV0)}
            for c in range(COLS):
                sin, tin, outs = [], [], []
                for mt in range(NT):
                    w = mt * 8 + c
                    base = ((mt // 4) * 96 + mt % 4) * TILE + c * 2304
                    kvq = ("qkv", base + 64 * TILE, [4, 2, 9, 256], [4 * TILE, 16 * TILE, 256, 1])
                    qa = ("qkv", base, [4, 2, 9, 256], [16 * TILE, 4 * TILE, 256, 1])
                    qb = ("qkv", base + 8 * TILE, [4, 2, 9, 256], [16 * TILE, 4 * TILE, 256, 1])
                    if kvmode:
                        sin += [kvq, qa, qb]
                        outs.append(("kv", w * 18432, [1, 1, 18, 1024], [0, 0, 1024, 1]))
                        outs.append(("pass", mt * PASS + c * PCOL + 6144, [1, 1, 72, 1024], [0, 0, 1024, 1]))
                        outs.append(("aux", w * 6144, [1, 1, 6, 1024], [0, 0, 1024, 1]))
                    else:
                        sin += [qa, qb]
                        for hi in range(3):
                            sin.append(("kv", (w + hi) * 18432, [4, 3, 6, 256], [1536, 6144, 256, 1]))
                        sin.append(("aux", 0, [4, 3, 6, 256], [1536, 6144, 256, 1]))
                        outs.append(("pass", mt * PASS + c * PCOL, [1, 1, 42, 1024], [0, 0, 1024, 1]))
                    tin.append(("tab", w * TE, [1, 1, 5, 1024], [0, 0, 1024, 1]))
                nS = len(sin) // NT
                nO = len(outs) // NT
                chans = [sin, tin, outs]
                fifos = [S_l3l2[c], T_f[c], C_l2l3[c]]
                tasks = [dict() for _ in chans]

                def start(ci, k):
                    name, off, sz, st = chans[ci][k]
                    tasks[ci][k] = shim_dma_single_bd_task(fifos[ci], bufs[name], tap=lin(full[name], off, sz, st), issue_token=True)
                    dma_start_task(tasks[ci][k])

                nxt = [0, 0, 0]
                for ci in range(3):
                    for k in range(min(WIN, len(chans[ci]))):
                        start(ci, k)
                        nxt[ci] = k + 1
                for mt in range(NT):
                    need = [(mt + 1) * nS, (mt + 1), (mt + 1) * nO]
                    for ci in range(3):
                        while nxt[ci] < len(chans[ci]) and nxt[ci] - WIN < need[ci]:
                            kk = nxt[ci]
                            dma_await_task(tasks[ci][kk - WIN])
                            dma_free_task(tasks[ci][kk - WIN])
                            start(ci, kk)
                            nxt[ci] += 1
                for ci in range(3):
                    for k in range(max(0, len(chans[ci]) - WIN), len(chans[ci])):
                        dma_await_task(tasks[ci][k])
                        dma_free_task(tasks[ci][k])


main()
