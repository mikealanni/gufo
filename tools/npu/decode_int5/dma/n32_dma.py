# Pure DMA stream bandwidth test: shim -> mem tile -> 4 core rows per column, cores only acquire/release.
# env: CH = shim channels per column feeding the weight stream (1 or 2), NI = elements per core, ELEM = bytes per element
import argparse
import os
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_

ROWS = 4
COLS = int(os.environ.get("COLS", "8"))
CH = int(os.environ.get("CH", "1"))
NI = int(os.environ.get("NI", "256"))
ELEM = int(os.environ.get("ELEM", "4096"))
AELEM = int(os.environ.get("AELEM", "0"))  # >0: also multicast an activation tile of this many bytes per iteration to every core
ADEPTH = int(os.environ.get("ADEPTH", "2"))


def main():
    ap = argparse.ArgumentParser()
    for a in ("-M", "-K", "-N", "-m", "-k", "-n"):
        ap.add_argument(a, type=int, default=0)
    ap.parse_args()
    with mlir_mod_ctx() as ctx:
        design()
        print(ctx.module)


def design():
    rows_per = ROWS // CH

    @device(AIEDevice.npu2)
    def device_body():
        L2_ty = np.ndarray[(rows_per * ELEM,), np.dtype[np.uint8]]
        L1_ty = np.ndarray[(ELEM,), np.dtype[np.uint8]]
        tiles = [[tile(col, row) for col in range(COLS)] for row in range(ROWS + 2)]
        shim_tiles, mem_tiles, core_tiles = tiles[0], tiles[1], tiles[2:]
        l3l2 = [[None] * CH for _ in range(COLS)]
        l2l1 = [[None] * COLS for _ in range(ROWS)]
        for col in range(COLS):
            for h in range(CH):
                l3l2[col][h] = object_fifo(f"B_L3L2_{col}_{h}", shim_tiles[col], mem_tiles[col], 2, L2_ty)
                outs = []
                for rr in range(rows_per):
                    row = h * rows_per + rr
                    l2l1[row][col] = object_fifo(f"B_L2L1_{col}_{row}", mem_tiles[col], core_tiles[row][col], 2, L1_ty)
                    outs.append(l2l1[row][col])
                object_fifo_link(l3l2[col][h], outs, [], [ELEM * rr for rr in range(rows_per)])
        if AELEM:
            A_ty = np.ndarray[(AELEM,), np.dtype[np.uint8]]
            A_l3l2 = object_fifo("A_L3L2", shim_tiles[0], mem_tiles[0], 2, A_ty)
            A_l2l1 = object_fifo("A_L2L1", mem_tiles[0], [core_tiles[r][c] for r in range(ROWS) for c in range(COLS)], ADEPTH, A_ty)
            object_fifo_link(A_l3l2, A_l2l1)
        for row in range(ROWS):
            for col in range(COLS):

                @core(core_tiles[row][col])
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range_(NI):
                            if AELEM:
                                ea = A_l2l1.acquire(ObjectFifoPort.Consume, 1)
                            e = l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                            l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            if AELEM:
                                A_l2l1.release(ObjectFifoPort.Consume, 1)

        per_col = NI * ROWS * ELEM
        per_ch = per_col // CH
        a_bytes = NI * AELEM
        total = COLS * per_col + a_bytes
        nk = per_ch // 1024
        d2 = next(d for d in range(min(nk, 1023), 0, -1) if nk % d == 0)
        d1 = nk // d2
        assert d1 <= 1023

        @runtime_sequence(np.ndarray[(total,), np.dtype[np.uint8]], np.ndarray[(4,), np.dtype[np.uint8]], np.ndarray[(4,), np.dtype[np.uint8]])
        def sequence(B, X, Y):
            tasks = []
            for col in range(COLS):
                for h in range(CH):
                    tap = TensorAccessPattern((1, total), col * per_col + h * per_ch, [1, d1, d2, 1024], [0, d2 * 1024, 1024, 1])
                    t = shim_dma_single_bd_task(l3l2[col][h], B, tap=tap, issue_token=True)
                    dma_start_task(t)
                    tasks.append(t)
            if AELEM:
                nka = a_bytes // 1024
                da2 = next(d for d in range(min(nka, 1023), 0, -1) if nka % d == 0)
                da1 = nka // da2
                ta = shim_dma_single_bd_task(A_l3l2, B, tap=TensorAccessPattern((1, total), COLS * per_col, [1, da1, da2, 1024], [0, da2 * 1024, 1024, 1]), issue_token=True)
                dma_start_task(ta)
                tasks.append(ta)
            dma_await_task(*tasks)
            dma_free_task(*tasks)


main()
