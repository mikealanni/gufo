#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
import argparse
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.aiex import v8bfp16ebs8
from aie.dialects import arith, memref
from aie.extras import types as T
import aie.utils.trace as trace_utils
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_


def ceildiv(a, b):
    return (a + b - 1) // b


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (32 Cores, 8x4 grid) with bfp16ebs8 weights and bf16 input/output",
        description="Emits MLIR code for a matrix multiplication design of the given input size using 32 cores (8 columns x 4 rows). Only supported in NPU2 devices.",
    )
    argparser.add_argument("-M", type=int, default=0)
    argparser.add_argument("-K", type=int, default=0)
    argparser.add_argument("-N", type=int, default=0)
    argparser.add_argument("-m", type=int, default=0)
    argparser.add_argument("-k", type=int, default=0)
    argparser.add_argument("-n", type=int, default=0)
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)


def my_matmul(M, K, N, m, k, n):
    n_aie_cols = 8
    n_aie_rows = 4
    nq = n // n_aie_rows
    dtype = v8bfp16ebs8
    dev_ty = AIEDevice.npu2

    @device(dev_ty)
    def device_body():
        A_ty = np.ndarray[(m, k // 8), np.dtype[dtype]]
        BQ = 1536
        B_l2_ty = np.ndarray[(n_aie_rows * BQ,), np.dtype[np.uint8]]
        B_l1_ty = np.ndarray[(BQ,), np.dtype[np.uint8]]
        C_l1_ty = np.ndarray[(m, nq // 8), np.dtype[dtype]]
        C_l2_ty = np.ndarray[(m, n // 8), np.dtype[dtype]]

        zero = external_func("zero_kernel", inputs=[C_l1_ty], link_with=f"mm_{m}x{k}x{n}.o")
        matmul = external_func(
            "matmul_vectorized_bfp16",
            inputs=[A_ty, B_l1_ty, C_l1_ty, np.int32],
            link_with=f"mm_{m}x{k}x{n}.o",
        )

        tiles = [[tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, n_aie_rows + 2)]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        A_l3l2 = object_fifo("A_L3L2", shim_tiles[0], mem_tiles[0], 2, A_ty)
        A_l2l1 = object_fifo(
            "A_L2L1", mem_tiles[0],
            [core_tiles[r][c] for r in range(n_aie_rows) for c in range(n_aie_cols)],
            2, A_ty,
        )
        object_fifo_link(A_l3l2, A_l2l1)

        B_l3l2 = [None] * n_aie_cols
        B_l2l1 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3 = [None] * n_aie_cols
        for col in range(n_aie_cols):
            B_l3l2[col] = object_fifo(f"B_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, B_l2_ty)
            for row in range(n_aie_rows):
                B_l2l1[row][col] = object_fifo(
                    f"B_L2L1_{col}_{row}", mem_tiles[col], core_tiles[row][col], 2, B_l1_ty
                )
            object_fifo_link(
                B_l3l2[col],
                [B_l2l1[row][col] for row in range(n_aie_rows)],
                [],
                [BQ * row for row in range(n_aie_rows)],
            )
            for row in range(n_aie_rows):
                C_l1l2[row][col] = object_fifo(
                    f"C_L1L2_{col}_{row}", core_tiles[row][col], mem_tiles[col], 1, C_l1_ty
                )
            C_l2l3[col] = object_fifo(f"C_L2L3_{col}", mem_tiles[col], shim_tiles[col], 2, C_l2_ty)
            object_fifo_link(
                [C_l1l2[row][col] for row in range(n_aie_rows)],
                C_l2l3[col],
                [m * nq // 8 * row for row in range(n_aie_rows)],
                [],
            )

        rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]
        rtps = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        rtp_locks = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rtps[row][col] = buffer(core_tiles[row][col], rtp_ty, name=f"rtp_{row}_{col}", use_write_rtp=True)
                rtp_locks[row][col] = lock(core_tiles[row][col], init=0, sym_name=f"rtp_lock_{row}_{col}")

        for row in range(n_aie_rows):
            for col in range(n_aie_cols):

                @core(core_tiles[row][col], stack_size=0xF00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        use_lock(rtp_locks[row][col], LockAction.Acquire, value=1)
                        zero_idx = arith.constant(T.index(), 0)
                        k_iters_i32 = memref.load(rtps[row][col], [zero_idx])
                        one_idx = arith.constant(T.index(), 1)
                        tiles_i32 = memref.load(rtps[row][col], [one_idx])
                        use_lock(rtp_locks[row][col], LockAction.Release, value=1)
                        k_iters = arith.index_cast(T.index(), k_iters_i32)
                        tile_iters = arith.index_cast(T.index(), tiles_i32)
                        for _ in range_(tile_iters):
                            elem_out = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                            zero(elem_out)
                            for _ in range_(k_iters):
                                elem_b = B_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                                elem_a = A_l2l1.acquire(ObjectFifoPort.Consume, 1)
                                matmul(elem_a, elem_b, elem_out, k_iters_i32)
                                A_l2l1.release(ObjectFifoPort.Consume, 1)
                                B_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K // 8,), np.dtype[dtype]],
            np.ndarray[((K // k) * (N // n) * n_aie_rows * BQ,), np.dtype[np.uint8]],
            np.ndarray[(M * N // 8,), np.dtype[dtype]],
        )
        def sequence(A, B, C):
            for row in range(n_aie_rows):
                for col in range(n_aie_cols):
                    npu_rtp_write(f"rtp_{row}_{col}", 0, K // k)
                    npu_rtp_write(f"rtp_{row}_{col}", 1, N // n // n_aie_cols)
                    set_lock_value(rtp_locks[row][col], 1)
            A_taps = TensorTiler2D.group_tiler((1, M * K // 8), (1, m * K // 8), (1, 1))
            B_taps = TensorTiler2D.group_tiler((1, (K // k) * (N // n) * n_aie_rows * BQ), (1, (K // k) * n_aie_rows * BQ), (1, 1))
            C_taps = TensorTiler2D.group_tiler((1, M * N // 8), (1, m * n // 8), (1, 1))
            num_groups = N // n // n_aie_cols
            tb_max = 4
            in_groups = [[] for _ in range(tb_max)]
            out_groups = [[] for _ in range(tb_max)]
            for g in range(num_groups):
                a_task = shim_dma_single_bd_task(A_l3l2, A, tap=A_taps[0], issue_token=False)
                dma_start_task(a_task)
                in_groups[g % tb_max].append(a_task)
                for col in range(n_aie_cols):
                    b_task = shim_dma_single_bd_task(
                        B_l3l2[col], B, tap=B_taps[g * n_aie_cols + col], issue_token=False
                    )
                    dma_start_task(b_task)
                    in_groups[g % tb_max].append(b_task)
                for col in range(n_aie_cols):
                    c_task = shim_dma_single_bd_task(
                        C_l2l3[col], C, tap=C_taps[g * n_aie_cols + col], issue_token=True
                    )
                    dma_start_task(c_task)
                    out_groups[g % tb_max].append(c_task)
                if (g % tb_max == 1) and (g != 1):
                    for slot in (2, 3):
                        dma_await_task(*out_groups[slot])
                        out_groups[slot] = []
                        dma_free_task(*in_groups[slot])
                        in_groups[slot] = []
                if g % tb_max == 3:
                    for slot in (0, 1):
                        dma_await_task(*out_groups[slot])
                        out_groups[slot] = []
                        dma_free_task(*in_groups[slot])
                        in_groups[slot] = []
            for slot in range(tb_max):
                if out_groups[slot]:
                    dma_await_task(*out_groups[slot])
                    dma_free_task(*in_groups[slot])


main()
