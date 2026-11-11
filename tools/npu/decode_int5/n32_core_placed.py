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
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_


def ceildiv(a, b):
    return (a + b - 1) // b


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (32 Cores, 8x4 grid) with bfp16ebs8 weights and bf16 input/output",
        description="Emits MLIR code for a matrix multiplication design of the given input size using 32 cores (8 columns x 4 rows). Only supported in NPU2 devices.",
    )
    argparser.add_argument("-M", type=int, default=16)
    argparser.add_argument("-K", type=int, default=0)
    argparser.add_argument("-N", type=int, default=0)
    argparser.add_argument("-m", type=int, default=16)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=128)
    argparser.add_argument("--mode", default="ffn", help="ffn | plain | conv")
    argparser.add_argument("--k1", type=int, default=None, help="Phase 1 K (ffn: gate_up K, conv: hidden)")
    argparser.add_argument("--n1", type=int, default=None, help="Phase 1 N (ffn: gate_up N, conv: in_proj N)")
    argparser.add_argument("--k2", type=int, default=None, help="Phase 2 K (ffn: down K, plain: K, conv: hidden)")
    argparser.add_argument("--n2", type=int, default=None, help="Phase 2 N (ffn: down N, plain: N, conv: hidden)")
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n, args.mode,
                  args.k1, args.n1, args.k2, args.n2)
        print(ctx.module)


def my_matmul(M, K, N, m, k, n, mode="ffn", k1=None, n1=None, k2=None, n2=None):
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
        CB = 1024
        C_l1_ty = np.ndarray[(CB,), np.dtype[np.uint8]]
        C_l2_ty = np.ndarray[(n_aie_rows * CB,), np.dtype[np.uint8]]

        zero = external_func("zero_kernel", inputs=[C_l1_ty], link_with=f"mm_{m}x{k}x{n}.o")
        matmul_gu = external_func(
            "matmul_p1",
            inputs=[A_ty, B_l1_ty, C_l1_ty, np.int32, np.int32],
            link_with=f"mm_{m}x{k}x{n}.o",
        )
        matmul_dn = external_func(
            "matmul_dn",
            inputs=[A_ty, B_l1_ty, C_l1_ty, np.int32],
            link_with=f"mm_{m}x{k}x{n}.o",
        )

        attn_begin = external_func("attn_begin", inputs=[A_ty, np.int32], link_with=f"mm_{m}x{k}x{n}.o")
        attn_q = external_func("attn_q", inputs=[A_ty, np.int32, np.int32], link_with=f"mm_{m}x{k}x{n}.o")
        attn_tail = external_func("attn_tail", inputs=[A_ty, np.int32, np.int32, np.int32], link_with=f"mm_{m}x{k}x{n}.o")
        attn_feed = external_func("attn_feed", inputs=[B_l1_ty, np.int32], link_with=f"mm_{m}x{k}x{n}.o")
        attn_out = external_func("attn_out", inputs=[C_l1_ty, np.int32], link_with=f"mm_{m}x{k}x{n}.o")

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
                [CB * row for row in range(n_aie_rows)],
                [],
            )

        rtp_ty = np.ndarray[(8,), np.dtype[np.int32]]
        ATTN_Q_ELEMS = 32
        ATTN_TAIL_ELEMS = 36
        ATTN_OUT_ELEMS = 34
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
                        i0 = arith.constant(T.index(), 0)
                        i1 = arith.constant(T.index(), 1)
                        i2 = arith.constant(T.index(), 2)
                        i3 = arith.constant(T.index(), 3)
                        i4 = arith.constant(T.index(), 4)
                        i5 = arith.constant(T.index(), 5)
                        i6 = arith.constant(T.index(), 6)
                        k1_i32 = memref.load(rtps[row][col], [i0])
                        t1_i32 = memref.load(rtps[row][col], [i1])
                        k2_i32 = memref.load(rtps[row][col], [i2])
                        t2_i32 = memref.load(rtps[row][col], [i3])
                        mode_i32 = memref.load(rtps[row][col], [i4])
                        attn_on_i32 = memref.load(rtps[row][col], [i5])
                        attn_feeds_i32 = memref.load(rtps[row][col], [i6])
                        use_lock(rtp_locks[row][col], LockAction.Release, value=1)
                        k1 = arith.index_cast(T.index(), k1_i32)
                        t1 = arith.index_cast(T.index(), t1_i32)
                        k2 = arith.index_cast(T.index(), k2_i32)
                        t2 = arith.index_cast(T.index(), t2_i32)
                        for _ in range_(t1):
                            elem_out = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                            zero(elem_out)
                            for _ in range_(k1):
                                elem_b = B_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                                elem_a = A_l2l1.acquire(ObjectFifoPort.Consume, 1)
                                matmul_gu(elem_a, elem_b, elem_out, k1_i32, mode_i32)
                                A_l2l1.release(ObjectFifoPort.Consume, 1)
                                B_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)
                        for _ in range_(t2):
                            elem_out = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                            zero(elem_out)
                            for _ in range_(k2):
                                elem_b = B_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                                elem_a = A_l2l1.acquire(ObjectFifoPort.Consume, 1)
                                matmul_dn(elem_a, elem_b, elem_out, k2_i32)
                                A_l2l1.release(ObjectFifoPort.Consume, 1)
                                B_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)
                        attn_on = arith.index_cast(T.index(), attn_on_i32)
                        attn_feeds = arith.index_cast(T.index(), attn_feeds_i32)
                        row_c = arith.constant(T.i32(), row)
                        col_c = arith.constant(T.i32(), col)
                        for _ in range_(attn_on):
                            elem_a = A_l2l1.acquire(ObjectFifoPort.Consume, 1)
                            attn_begin(elem_a, row_c)
                            A_l2l1.release(ObjectFifoPort.Consume, 1)
                            for qi in range_(ATTN_Q_ELEMS):
                                elem_a = A_l2l1.acquire(ObjectFifoPort.Consume, 1)
                                attn_q(elem_a, arith.index_cast(T.i32(), qi), row_c)
                                A_l2l1.release(ObjectFifoPort.Consume, 1)
                            for ti in range_(ATTN_TAIL_ELEMS):
                                elem_a = A_l2l1.acquire(ObjectFifoPort.Consume, 1)
                                attn_tail(elem_a, arith.index_cast(T.i32(), ti), row_c, col_c)
                                A_l2l1.release(ObjectFifoPort.Consume, 1)
                            for _ in range_(attn_feeds):
                                elem_b = B_l2l1[row][col].acquire(ObjectFifoPort.Consume, 1)
                                attn_feed(elem_b, col_c)
                                B_l2l1[row][col].release(ObjectFifoPort.Consume, 1)
                            for oi in range_(ATTN_OUT_ELEMS):
                                elem_out = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                                attn_out(elem_out, arith.index_cast(T.i32(), oi))
                                C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        if mode == "attn":
            mchunks = k1 if k1 is not None else 4
            a_groups = (1 + ATTN_Q_ELEMS + ATTN_TAIL_ELEMS) * 128
            chunk_bytes = 3 * n_aie_rows * BQ
            b_total = n_aie_cols * mchunks * chunk_bytes
            c_per_col = ATTN_OUT_ELEMS * n_aie_rows * CB
            c_total = n_aie_cols * c_per_col

            @runtime_sequence(
                np.ndarray[(a_groups,), np.dtype[dtype]],
                np.ndarray[(b_total,), np.dtype[np.uint8]],
                np.ndarray[(c_total,), np.dtype[np.uint8]],
            )
            def attn_sequence(A, B, C):
                for row in range(n_aie_rows):
                    for col in range(n_aie_cols):
                        for wi, wv in enumerate((0, 0, 0, 0, 0, 1, 3 * mchunks)):
                            npu_rtp_write(f"rtp_{row}_{col}", wi, wv)
                        set_lock_value(rtp_locks[row][col], 1)
                a_tap = TensorAccessPattern((1, a_groups), 0, [1, 1, 1, a_groups], [0, 0, 0, 1])
                a_task = shim_dma_single_bd_task(A_l3l2, A, tap=a_tap, issue_token=False)
                dma_start_task(a_task)
                b_tasks, c_tasks = [], []
                for col in range(n_aie_cols):
                    inner = min(mchunks, 512)
                    b_tap = TensorAccessPattern(
                        (1, b_total), col * chunk_bytes,
                        [mchunks // inner, inner, chunk_bytes // 1024, 1024],
                        [inner * n_aie_cols * chunk_bytes, n_aie_cols * chunk_bytes, 1024, 1])
                    b_tasks.append(shim_dma_single_bd_task(B_l3l2[col], B, tap=b_tap, issue_token=False))
                    dma_start_task(b_tasks[-1])
                for col in range(n_aie_cols):
                    c_tap = TensorAccessPattern((1, c_total), col * c_per_col, [1, 1, 1, c_per_col], [0, 0, 0, 1])
                    c_tasks.append(shim_dma_single_bd_task(C_l2l3[col], C, tap=c_tap, issue_token=True))
                    dma_start_task(c_tasks[-1])
                dma_await_task(*c_tasks)
                dma_free_task(a_task, *b_tasks)
            return

        if mode == "conv":
            K1 = k1 if k1 is not None else K
            K2 = k2 if k2 is not None else K
            N1 = n1 if n1 is not None else 3 * K
            N2 = n2 if n2 is not None else N
            k1t = K1 // k + 1
            KA, rtp_mode = K1 + k, 1
        elif mode == "ffn":
            K1 = k1 if k1 is not None else K
            N1 = n1 if n1 is not None else 15360
            K2 = k2 if k2 is not None else 7680
            N2 = n2 if n2 is not None else 2048
            k1t = K1 // k
            KA, rtp_mode = K1, 0
        else:
            K1, K2, N1, N2 = k, (k2 if k2 is not None else K), 0, (n2 if n2 is not None else N)
            k1t = 0
            KA, rtp_mode = K2, 0
        BQ_ = BQ
        gu_bytes = k1t * (N1 // n) * n_aie_rows * BQ_ if N1 else 0
        dn_bytes = (K2 // k) * (N2 // n) * n_aie_rows * BQ_
        SB = n_aie_rows * CB
        c_total = (N1 // n + N2 // n) * SB
        a_groups = m * KA // 8

        @runtime_sequence(
            np.ndarray[(a_groups,), np.dtype[dtype]],
            np.ndarray[(gu_bytes + dn_bytes,), np.dtype[np.uint8]],
            np.ndarray[(c_total,), np.dtype[np.uint8]],
        )
        def sequence(A, B, C):
            for row in range(n_aie_rows):
                for col in range(n_aie_cols):
                    npu_rtp_write(f"rtp_{row}_{col}", 0, k1t)
                    npu_rtp_write(f"rtp_{row}_{col}", 1, N1 // n // n_aie_cols if N1 else 0)
                    npu_rtp_write(f"rtp_{row}_{col}", 2, K2 // k)
                    npu_rtp_write(f"rtp_{row}_{col}", 3, N2 // n // n_aie_cols)
                    npu_rtp_write(f"rtp_{row}_{col}", 4, rtp_mode)
                    npu_rtp_write(f"rtp_{row}_{col}", 5, 0)
                    npu_rtp_write(f"rtp_{row}_{col}", 6, 0)
                    set_lock_value(rtp_locks[row][col], 1)
            A_tap = TensorAccessPattern((1, a_groups), 0, [1, 1, 1, a_groups], [0, 0, 0, 1])
            tb_max = 4

            def phase(num_groups, a_tap_fn, b_off_fn, b_len, c_off_fn):
                in_groups = [[] for _ in range(tb_max)]
                out_groups = [[] for _ in range(tb_max)]
                for g in range(num_groups):
                    a_task = shim_dma_single_bd_task(A_l3l2, a_tap_fn[0], tap=a_tap_fn[1], issue_token=False)
                    dma_start_task(a_task)
                    in_groups[g % tb_max].append(a_task)
                    for col in range(n_aie_cols):
                        b_tap = TensorAccessPattern(
                            (1, gu_bytes + dn_bytes), b_off_fn(g * n_aie_cols + col), [1, 1, 1, b_len], [0, 0, 0, 1]
                        )
                        b_task = shim_dma_single_bd_task(B_l3l2[col], B, tap=b_tap, issue_token=False)
                        dma_start_task(b_task)
                        in_groups[g % tb_max].append(b_task)
                    for col in range(n_aie_cols):
                        c_tap = TensorAccessPattern(
                            (1, c_total), c_off_fn(g * n_aie_cols + col),
                            [1, 1, 1, SB], [0, 0, 0, 1],
                        )
                        c_task = shim_dma_single_bd_task(C_l2l3[col], C, tap=c_tap, issue_token=True)
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

            if mode == "conv":
                phase(
                    N1 // n // n_aie_cols, (A, A_tap),
                    lambda s: s * k1t * n_aie_rows * BQ_, k1t * n_aie_rows * BQ_,
                    lambda s: ({2: 0, 1: 16, 0: 32}[(s // 8) % 3] + (s // 8 // 3) * 8 + s % 8) * SB,
                )
                a2 = (C, TensorAccessPattern((1, c_total), 0, [K2 // k // 2, 2, 2, 576], [SB, 2 * CB, CB, 1]))
            elif mode == "ffn":
                phase(
                    N1 // n // n_aie_cols, (A, A_tap),
                    lambda s: s * k1t * n_aie_rows * BQ_, k1t * n_aie_rows * BQ_,
                    lambda s: s * SB,
                )
                a2 = (C, TensorAccessPattern((1, c_total), 0, [1, K2 // k, n_aie_rows, 288], [0, SB, CB, 1]))
            else:
                a2 = (A, A_tap)
            phase(
                N2 // n // n_aie_cols, a2,
                lambda s: gu_bytes + s * (K2 // k) * n_aie_rows * BQ_,
                (K2 // k) * n_aie_rows * BQ_,
                lambda s: (N1 // n + s) * SB,
            )


main()
