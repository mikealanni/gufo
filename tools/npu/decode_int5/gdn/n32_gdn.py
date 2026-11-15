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
TPE = 4
REC_IN, REC_OUT = 1024, 512
IN_E, OUT_E = TPE * REC_IN, TPE * REC_OUT
NE = int(os.environ.get("GDN_NE", "64"))  # elements per core per dispatch (NE * 4 tokens per head)


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
        I2_ty = np.ndarray[(ROWS * IN_E,), np.dtype[np.uint8]]
        I1_ty = np.ndarray[(IN_E,), np.dtype[np.uint8]]
        O1_ty = np.ndarray[(OUT_E,), np.dtype[np.uint8]]
        O2_ty = np.ndarray[(ROWS * OUT_E,), np.dtype[np.uint8]]
        lw = f"mm_{m}x{k}x{n}.o"
        gdn_begin = external_func("gdn_begin", inputs=[], link_with=lw)
        gdn_chunk = external_func("gdn_chunk", inputs=[I1_ty, O1_ty], link_with=lw)

        tiles = [[tile(col, row) for col in range(COLS)] for row in range(ROWS + 2)]
        shim_tiles, mem_tiles, core_tiles = tiles[0], tiles[1], tiles[2:]
        I_l3l2 = [None] * COLS
        I_l2l1 = [[None] * COLS for _ in range(ROWS)]
        O_l1l2 = [[None] * COLS for _ in range(ROWS)]
        O_l2l3 = [None] * COLS
        for col in range(COLS):
            I_l3l2[col] = object_fifo(f"I_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, I2_ty)
            for row in range(ROWS):
                I_l2l1[row][col] = object_fifo(f"I_L2L1_{col}_{row}", mem_tiles[col], core_tiles[row][col], 2, I1_ty)
            object_fifo_link(I_l3l2[col], [I_l2l1[row][col] for row in range(ROWS)], [], [IN_E * row for row in range(ROWS)])
            for row in range(ROWS):
                O_l1l2[row][col] = object_fifo(f"O_L1L2_{col}_{row}", core_tiles[row][col], mem_tiles[col], 2, O1_ty)
            O_l2l3[col] = object_fifo(f"O_L2L3_{col}", mem_tiles[col], shim_tiles[col], 2, O2_ty)
            object_fifo_link([O_l1l2[row][col] for row in range(ROWS)], O_l2l3[col], [OUT_E * row for row in range(ROWS)], [])

        for row in range(ROWS):
            for col in range(COLS):

                @core(core_tiles[row][col], stack_size=0x2000)
                def core_body():
                    gdn_begin()
                    for _ in range_(0xFFFFFFFF):
                        i = I_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                        o = O_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                        gdn_chunk(i, o)
                        I_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                        O_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        in_bytes = COLS * NE * ROWS * IN_E
        out_bytes = COLS * NE * ROWS * OUT_E

        @runtime_sequence(np.ndarray[(in_bytes,), np.dtype[np.uint8]], np.ndarray[(4,), np.dtype[np.uint8]], np.ndarray[(out_bytes,), np.dtype[np.uint8]])
        def sequence(IN, X, OUT):
            per_in = NE * ROWS * IN_E
            per_out = NE * ROWS * OUT_E
            tasks = []
            for col in range(COLS):
                ni = per_in // 1024
                d2 = next(d for d in range(min(ni, 1023), 0, -1) if ni % d == 0)
                d1 = ni // d2
                ti = shim_dma_single_bd_task(I_l3l2[col], IN, tap=TensorAccessPattern((1, in_bytes), col * per_in, [1, d1, d2, 1024], [0, d2 * 1024, 1024, 1]), issue_token=True)
                dma_start_task(ti)
                no = per_out // 1024
                e2 = next(d for d in range(min(no, 1023), 0, -1) if no % d == 0)
                e1 = no // e2
                to = shim_dma_single_bd_task(O_l2l3[col], OUT, tap=TensorAccessPattern((1, out_bytes), col * per_out, [1, e1, e2, 1024], [0, e2 * 1024, 1024, 1]), issue_token=True)
                dma_start_task(to)
                tasks += [ti, to]
            dma_await_task(*tasks)
            dma_free_task(*tasks)


main()
