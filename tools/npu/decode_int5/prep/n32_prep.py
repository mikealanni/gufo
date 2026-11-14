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
QE = 13824
KE = QE
PIECE = 1536
NQ = int(os.environ.get("PREP_NQ", "8"))
NKV = int(os.environ.get("PREP_NKV", "8"))


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
        Q_l2_ty = np.ndarray[(ROWS * QE,), np.dtype[np.uint8]]
        Q_l1_ty = np.ndarray[(QE,), np.dtype[np.uint8]]
        C_l1_ty = np.ndarray[(PIECE,), np.dtype[np.uint8]]
        C_l2_ty = np.ndarray[(ROWS * PIECE,), np.dtype[np.uint8]]
        lw = f"mm_{m}x{k}x{n}.o"
        prep_q = external_func("prep_q", inputs=[Q_l1_ty], link_with=lw)
        prep_kv = external_func("prep_kv", inputs=[Q_l1_ty], link_with=lw)
        prep_emit = external_func("prep_emit", inputs=[C_l1_ty, np.int32], link_with=lw)

        tiles = [[tile(col, row) for col in range(COLS)] for row in range(ROWS + 2)]
        shim_tiles, mem_tiles, core_tiles = tiles[0], tiles[1], tiles[2:]

        Q_l3l2 = [None] * COLS
        Q_l2l1 = [[None] * COLS for _ in range(ROWS)]
        C_l1l2 = [[None] * COLS for _ in range(ROWS)]
        C_l2l3 = [None] * COLS
        for col in range(COLS):
            Q_l3l2[col] = object_fifo(f"Q_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, Q_l2_ty)
            for row in range(ROWS):
                Q_l2l1[row][col] = object_fifo(f"Q_L2L1_{col}_{row}", mem_tiles[col], core_tiles[row][col], 2, Q_l1_ty)
            object_fifo_link(Q_l3l2[col], [Q_l2l1[row][col] for row in range(ROWS)], [], [QE * row for row in range(ROWS)])
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
                        nq_i32 = memref.load(rtps[row][col], [arith.constant(T.index(), 0)])
                        nk_i32 = memref.load(rtps[row][col], [arith.constant(T.index(), 1)])
                        use_lock(rtp_locks[row][col], LockAction.Release, value=1)
                        nq = arith.index_cast(T.index(), nq_i32)
                        nk = arith.index_cast(T.index(), nk_i32)
                        for _ in range_(nq):
                            e = Q_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                            prep_q(e)
                            Q_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            for u in range_(6):
                                o = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                                prep_emit(o, arith.index_cast(T.i32(), u))
                                C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)
                        for _ in range_(nk):
                            e = Q_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                            prep_kv(e)
                            Q_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            for u in range_(3):
                                o = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                                prep_emit(o, arith.index_cast(T.i32(), u))
                                C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        q_col_in, k_col_in = NQ * ROWS * QE, NKV * ROWS * KE
        col_in = q_col_in + k_col_in
        q_col_out, k_col_out = NQ * 6 * ROWS * PIECE, NKV * 3 * ROWS * PIECE

        @runtime_sequence(
            np.ndarray[(COLS * col_in,), np.dtype[np.uint8]],
            np.ndarray[(4,), np.dtype[np.uint8]],
            np.ndarray[(COLS * (q_col_out + k_col_out),), np.dtype[np.uint8]],
        )
        def sequence(QI, KI, OUT):
            for row in range(ROWS):
                for col in range(COLS):
                    npu_rtp_write(f"rtp_{row}_{col}", 0, NQ)
                    npu_rtp_write(f"rtp_{row}_{col}", 1, NKV)
                    set_lock_value(rtp_locks[row][col], 1)
            in_tasks, c_tasks = [], []
            for col in range(COLS):
                q_tap = TensorAccessPattern(
                    (1, COLS * col_in), col * col_in, [1, NQ + NKV, ROWS * QE // 1024, 1024], [0, ROWS * QE, 1024, 1]
                )
                in_tasks.append(shim_dma_single_bd_task(Q_l3l2[col], QI, tap=q_tap, issue_token=False))
                dma_start_task(in_tasks[-1])
            tot_k = (q_col_out + k_col_out) // 1024
            o_in = next(d for d in range(min(tot_k, 1023), 0, -1) if tot_k % d == 0)
            o_out = tot_k // o_in
            assert o_out <= 1023
            for col in range(COLS):
                o_tap = TensorAccessPattern(
                    (1, COLS * (q_col_out + k_col_out)), col * (q_col_out + k_col_out),
                    [1, o_out, o_in, 1024], [0, o_in * 1024, 1024, 1]
                )
                c_tasks.append(shim_dma_single_bd_task(C_l2l3[col], OUT, tap=o_tap, issue_token=True))
                dma_start_task(c_tasks[-1])
            dma_await_task(*c_tasks)
            dma_free_task(*in_tasks)


main()
