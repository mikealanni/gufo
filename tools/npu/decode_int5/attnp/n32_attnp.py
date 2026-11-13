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
BQ = 1536
CB = 1024
CHUNK = 3 * ROWS * BQ
PREFIX_UNITS = 16
PREFIX_COL = PREFIX_UNITS * ROWS * BQ
OUT_ELEMS = 34
MAX_BD = 1023


def main():
    ap = argparse.ArgumentParser()
    for a in ("-M", "-K", "-N", "-m", "-k", "-n"):
        ap.add_argument(a, type=int, default=0)
    args = ap.parse_args()
    mc = int(os.environ.get("ATTNP_MC", "8"))
    with mlir_mod_ctx() as ctx:
        design(args.m, args.k, args.n, mc)
        print(ctx.module)


def design(m, k, n, mc):
    @device(AIEDevice.npu2)
    def device_body():
        B_l2_ty = np.ndarray[(ROWS * BQ,), np.dtype[np.uint8]]
        B_l1_ty = np.ndarray[(BQ,), np.dtype[np.uint8]]
        B_chunk_ty = np.ndarray[(3 * BQ,), np.dtype[np.uint8]]
        C_l1_ty = np.ndarray[(CB,), np.dtype[np.uint8]]
        C_l2_ty = np.ndarray[(ROWS * CB,), np.dtype[np.uint8]]
        lw = f"mm_{m}x{k}x{n}.o"
        attn_begin = external_func("attn_begin_p", inputs=[B_l1_ty], link_with=lw)
        attn_q = external_func("attn_q_p", inputs=[B_l1_ty, np.int32], link_with=lw)
        attn_tail = external_func("attn_tail_p", inputs=[B_l1_ty, np.int32], link_with=lw)
        attn_feed = external_func("attn_feed_p", inputs=[B_chunk_ty], link_with=lw)
        attn_out = external_func("attn_out", inputs=[C_l1_ty, np.int32], link_with=lw)

        tiles = [[tile(col, row) for col in range(COLS)] for row in range(ROWS + 2)]
        shim_tiles, mem_tiles, core_tiles = tiles[0], tiles[1], tiles[2:]

        B_l3l2 = [None] * COLS
        B_l2l1 = [[None] * COLS for _ in range(ROWS)]
        C_l1l2 = [[None] * COLS for _ in range(ROWS)]
        C_l2l3 = [None] * COLS
        for col in range(COLS):
            B_l3l2[col] = object_fifo(f"B_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, B_l2_ty)
            for row in range(ROWS):
                B_l2l1[row][col] = object_fifo(
                    f"B_L2L1_{col}_{row}", mem_tiles[col], core_tiles[row][col], 1, B_l1_ty
                )
            object_fifo_link(
                B_l3l2[col], [B_l2l1[row][col] for row in range(ROWS)], [], [BQ * row for row in range(ROWS)]
            )
            for row in range(ROWS):
                C_l1l2[row][col] = object_fifo(
                    f"C_L1L2_{col}_{row}", core_tiles[row][col], mem_tiles[col], 1, C_l1_ty
                )
            C_l2l3[col] = object_fifo(f"C_L2L3_{col}", mem_tiles[col], shim_tiles[col], 2, C_l2_ty)
            object_fifo_link(
                [C_l1l2[row][col] for row in range(ROWS)], C_l2l3[col], [CB * row for row in range(ROWS)], []
            )

        BC = [
            object_fifo(f"BC_{r}", shim_tiles[r], [core_tiles[r][c] for c in range(COLS)], 2, B_chunk_ty)
            for r in range(ROWS)
        ]
        rtp_ty = np.ndarray[(1,), np.dtype[np.int32]]
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
                        i0 = arith.constant(T.index(), 0)
                        feeds_i32 = memref.load(rtps[row][col], [i0])
                        use_lock(rtp_locks[row][col], LockAction.Release, value=1)
                        feeds = arith.index_cast(T.index(), feeds_i32)
                        e = B_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                        attn_begin(e)
                        B_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                        for qi in range_(6):
                            e = B_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                            attn_q(e, arith.index_cast(T.i32(), qi))
                            B_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                        for ti in range_(9):
                            e = B_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                            attn_tail(e, arith.index_cast(T.i32(), ti))
                            B_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                        for _ in range_(feeds):
                            e = BC[row].acquire(ObjectFifoPort.Consume, 1)
                            attn_feed(e)
                            BC[row].release(ObjectFifoPort.Consume, 1)
                        for oi in range_(OUT_ELEMS):
                            o = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                            attn_out(o, arith.index_cast(T.i32(), oi))
                            C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        b_total = mc * CHUNK
        c_per_col = OUT_ELEMS * ROWS * CB
        c_total = COLS * c_per_col
        n_bd = (mc + MAX_BD - 1) // MAX_BD
        assert n_bd <= 3, "context beyond 3 BDs of 1023 chunks unsupported"
        per_bd = (mc + n_bd - 1) // n_bd

        @runtime_sequence(
            np.ndarray[(COLS * PREFIX_COL,), np.dtype[np.uint8]],
            np.ndarray[(b_total,), np.dtype[np.uint8]],
            np.ndarray[(c_total,), np.dtype[np.uint8]],
        )
        def sequence(A, B, C):
            for row in range(ROWS):
                for col in range(COLS):
                    npu_rtp_write(f"rtp_{row}_{col}", 0, mc)
                    set_lock_value(rtp_locks[row][col], 1)
            in_tasks, c_tasks = [], []
            for col in range(COLS):
                p_tap = TensorAccessPattern(
                    (1, COLS * PREFIX_COL), col * PREFIX_COL, [1, 1, PREFIX_COL // 1024, 1024], [0, 0, 1024, 1]
                )
                in_tasks.append(shim_dma_single_bd_task(B_l3l2[col], A, tap=p_tap, issue_token=False))
                dma_start_task(in_tasks[-1])
            for r in range(ROWS):
                for bi in range(n_bd):
                    inner = min(per_bd, mc - bi * per_bd)
                    b_tap = TensorAccessPattern(
                        (1, b_total), bi * per_bd * CHUNK + r * BQ, [1, inner, 3, BQ], [0, CHUNK, ROWS * BQ, 1]
                    )
                    in_tasks.append(shim_dma_single_bd_task(BC[r], B, tap=b_tap, issue_token=False))
                    dma_start_task(in_tasks[-1])
            for col in range(COLS):
                c_tap = TensorAccessPattern((1, c_total), col * c_per_col, [1, 1, 1, c_per_col], [0, 0, 0, 1])
                c_tasks.append(shim_dma_single_bd_task(C_l2l3[col], C, tap=c_tap, issue_token=True))
                dma_start_task(c_tasks[-1])
            dma_await_task(*c_tasks)
            dma_free_task(*in_tasks)


main()
