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
ELEM = 6912
CELEM = 2304
NE = int(os.environ.get("PREP_NE", "8"))


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
        B_l2_ty = np.ndarray[(ROWS * ELEM,), np.dtype[np.uint8]]
        B_l1_ty = np.ndarray[(ELEM,), np.dtype[np.uint8]]
        C_l1_ty = np.ndarray[(CELEM,), np.dtype[np.uint8]]
        C_l2_ty = np.ndarray[(ROWS * CELEM,), np.dtype[np.uint8]]
        lw = f"mm_{m}x{k}x{n}.o"
        prep = external_func("prep_elem", inputs=[B_l1_ty, C_l1_ty], link_with=lw)

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
                    f"B_L2L1_{col}_{row}", mem_tiles[col], core_tiles[row][col], 2, B_l1_ty
                )
            object_fifo_link(
                B_l3l2[col], [B_l2l1[row][col] for row in range(ROWS)], [], [ELEM * row for row in range(ROWS)]
            )
            for row in range(ROWS):
                C_l1l2[row][col] = object_fifo(
                    f"C_L1L2_{col}_{row}", core_tiles[row][col], mem_tiles[col], 2, C_l1_ty
                )
            C_l2l3[col] = object_fifo(f"C_L2L3_{col}", mem_tiles[col], shim_tiles[col], 2, C_l2_ty)
            object_fifo_link(
                [C_l1l2[row][col] for row in range(ROWS)], C_l2l3[col], [CELEM * row for row in range(ROWS)], []
            )

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
                        ne_i32 = memref.load(rtps[row][col], [i0])
                        use_lock(rtp_locks[row][col], LockAction.Release, value=1)
                        ne = arith.index_cast(T.index(), ne_i32)
                        for _ in range_(ne):
                            e = B_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                            o = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                            prep(e, o)
                            C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)
                            B_l2l1[row][col].release(ObjectFifoPort.Consume, 1)

        @runtime_sequence(
            np.ndarray[(COLS * NE * ROWS * ELEM,), np.dtype[np.uint8]],
            np.ndarray[(4,), np.dtype[np.uint8]],
            np.ndarray[(COLS * NE * ROWS * CELEM,), np.dtype[np.uint8]],
        )
        def sequence(A, B, C):
            ne = NE
            for row in range(ROWS):
                for col in range(COLS):
                    npu_rtp_write(f"rtp_{row}_{col}", 0, ne)
                    set_lock_value(rtp_locks[row][col], 1)
            in_tasks, c_tasks = [], []
            for col in range(COLS):
                a_tap = TensorAccessPattern(
                    (1, COLS * ne * ROWS * ELEM), col * ne * ROWS * ELEM, [1, ne, ROWS * ELEM // 1024, 1024], [0, ROWS * ELEM, 1024, 1]
                )
                in_tasks.append(shim_dma_single_bd_task(B_l3l2[col], A, tap=a_tap, issue_token=False))
                dma_start_task(in_tasks[-1])
            for col in range(COLS):
                c_tap = TensorAccessPattern(
                    (1, COLS * ne * ROWS * CELEM), col * ne * ROWS * CELEM, [1, ne, ROWS * CELEM // 1024, 1024], [0, ROWS * CELEM, 1024, 1]
                )
                c_tasks.append(shim_dma_single_bd_task(C_l2l3[col], C, tap=c_tap, issue_token=True))
                dma_start_task(c_tasks[-1])
            dma_await_task(*c_tasks)
            dma_free_task(*in_tasks)


main()
