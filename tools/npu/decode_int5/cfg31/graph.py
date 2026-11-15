#
# cfg30: swiglu-wire -> dn-A reshape pass (AGENT-8).
#
# Device-side link between the two FULLC gu GEMMs and the dn GEMM:
#   args = (A = pa.C wire, B = pb.C wire, C = dn A stream)
# Every shim BD is a SINGLE LINEAR run (this stack only runs 64B-aligned
# linear shim transfers; tile-mode BDs with 72B runs crash at runtime):
#   read  = one 2048-block wire tile (18432 B, aligned),
#   write = one 1024-block dn A tile (9216 B = 64*144, aligned).
# Wire tile -> dn A tile bijection (verified against the mode-7 host map):
#   pa (kt 0..63):  T = ((mt>>2)*8 + (kt>>3))*32 + (kt&7)*4 + (mt&3)
#   pb (kt 64..):   T = ((mt>>2)*7 + (kt'>>3))*32 + (kt'&7)*4 + (mt&3)
# Core just permutes vecs inside the tile (no scalar arg needed).
#
import argparse
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_


def main():
    argparser = argparse.ArgumentParser(prog="cfg30 swiglu reshape")
    argparser.add_argument("-M", type=int, default=0)
    argparser.add_argument("-K", type=int, default=0)
    argparser.add_argument("-N", type=int, default=0)
    argparser.add_argument("-m", type=int, default=0)
    argparser.add_argument("-k", type=int, default=0)
    argparser.add_argument("-n", type=int, default=0)
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_design(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)


import os
BATCH = int(os.environ.get("SWI31_BATCH", "4"))
AWAIT_IN = os.environ.get("SWI31_AWAIT_IN", "0") == "1"


def lin(full, offset, nblk):
    return TensorAccessPattern((1, full), offset, [1, 1, 1, nblk], [0, 0, 0, 1])


def my_design(M, K, N, m, k, n):
    n_aie_rows = 4
    dtype = v8bfp16ebs8
    dev_ty = AIEDevice.npu2

    WIRE = np.ndarray[(2048,), np.dtype[dtype]]  # one input tile
    OUT = np.ndarray[(1024,), np.dtype[dtype]]  # one dn A tile

    @device(dev_ty)
    def device_body():
        resh = external_func(
            "swi_swiglu_permute", inputs=[WIRE, OUT], link_with=f"mm_{m}x{k}x{n}.o"
        )

        tiles = [
            [tile(col, row) for col in range(0, 8)] for row in range(0, n_aie_rows + 2)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        A_fifos = [None] * n_aie_rows  # pa wire  -> core[g][0]
        B_fifos = [None] * n_aie_rows  # pb wire  -> core[g][0]
        Cs_fifos = [None] * n_aie_rows  # mem[g] -> shim (L2L3 side)

        for g in range(n_aie_rows):
            A_fifos[g] = object_fifo(
                f"A_L3L2_{g}", shim_tiles[2 * g], mem_tiles[2 * g], 2, WIRE
            )
            B_fifos[g] = object_fifo(
                f"B_L3L2_{g}", shim_tiles[2 * g + 1], mem_tiles[2 * g + 1], 2, WIRE
            )
            A_c = object_fifo(f"A_L2L1_{g}", mem_tiles[2 * g], core_tiles[g][0], 1, WIRE)
            B_c = object_fifo(f"B_L2L1_{g}", mem_tiles[2 * g + 1], core_tiles[g][0], 1, WIRE)
            C_c = object_fifo(f"C_L1L2_{g}", core_tiles[g][0], mem_tiles[2 * g], 2, OUT)
            Cs_fifos[g] = object_fifo(
                f"C_L2L3_{g}", mem_tiles[2 * g], shim_tiles[2 * g], 2, OUT
            )
            object_fifo_link(A_fifos[g], A_c)
            object_fifo_link(B_fifos[g], B_c)
            object_fifo_link(C_c, Cs_fifos[g])

            ac, bc, cc = A_c, B_c, C_c

            @core(core_tiles[g][0], stack_size=0x400)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for partA in range(2):
                        for _ in range_(64):
                            out = cc.acquire(ObjectFifoPort.Produce, 1)
                            elem = ac.acquire(ObjectFifoPort.Consume, 1)
                            resh(elem, out)
                            ac.release(ObjectFifoPort.Consume, 1)
                            cc.release(ObjectFifoPort.Produce, 1)
                    for partB in range(2):
                        for _ in range_(56):
                            out = cc.acquire(ObjectFifoPort.Produce, 1)
                            elem = bc.acquire(ObjectFifoPort.Consume, 1)
                            resh(elem, out)
                            bc.release(ObjectFifoPort.Consume, 1)
                            cc.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * 1024,), np.dtype[dtype]],
            np.ndarray[(M * 896,), np.dtype[dtype]],
            np.ndarray[(M * 960,), np.dtype[dtype]],
        )
        def sequence(A, B, C):
            for g in range(n_aie_rows):
                outs, ins = [], []
                for mt2 in range(2):  # A: mt0 kt 0..63, mt1 kt 0..63
                    mt = 2 * g + mt2
                    for kt in range(64):
                        T = ((mt >> 2) * 8 + (kt >> 3)) * 32 + (kt & 7) * 4 + (mt & 3)
                        it = shim_dma_single_bd_task(
                            A_fifos[g], A, tap=lin(M * 1024, T * 2048, 2048),
                            issue_token=AWAIT_IN,
                        )
                        ct = shim_dma_single_bd_task(
                            Cs_fifos[g], C,
                            tap=lin(M * 960, (mt * 120 + kt) * 1024, 1024),
                            issue_token=True,
                        )
                        dma_start_task(it)
                        dma_start_task(ct)
                        ins.append(it)
                        outs.append(ct)
                        if len(outs) == BATCH:
                            for i in range(len(outs)):
                                dma_await_task(outs[i])
                                (dma_await_task(ins[i]) if AWAIT_IN else None)
                                dma_free_task(ins[i])
                            outs, ins = [], []
                for mt2 in range(2):  # B: mt0, mt1: kt' 0..55
                    mt = 2 * g + mt2
                    for k2 in range(56):
                        T = ((mt >> 2) * 7 + (k2 >> 3)) * 32 + (k2 & 7) * 4 + (mt & 3)
                        it = shim_dma_single_bd_task(
                            B_fifos[g], B, tap=lin(M * 896, T * 2048, 2048),
                            issue_token=AWAIT_IN,
                        )
                        ct = shim_dma_single_bd_task(
                            Cs_fifos[g], C,
                            tap=lin(M * 960, (mt * 120 + 64 + k2) * 1024, 1024),
                            issue_token=True,
                        )
                        dma_start_task(it)
                        dma_start_task(ct)
                        ins.append(it)
                        outs.append(ct)
                        if len(outs) == BATCH:
                            for i in range(len(outs)):
                                dma_await_task(outs[i])
                                (dma_await_task(ins[i]) if AWAIT_IN else None)
                                dma_free_task(ins[i])
                            outs, ins = [], []
                for i in range(len(outs)):
                    dma_await_task(outs[i])
                    (dma_await_task(ins[i]) if AWAIT_IN else None)
                    dma_free_task(ins[i])


main()
