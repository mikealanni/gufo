module {
  aie.device(npu2) {
    func.func private @zero_kernel(memref<1024xui8>, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @zero_kernel_c2(memref<4096xui8>, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @matmul_p1(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @matmul_dn(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @matmul_p1_c2(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @matmul_dn_c2(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_begin(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_q(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_tail(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_feed(memref<1536xui8>, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_out(memref<1024xui8>, i32) attributes {link_with = "mm_16x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_3 = aie.tile(4, 3)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_3 = aie.tile(7, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_4_4 = aie.tile(4, 4)
    %tile_5_4 = aie.tile(5, 4)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_4 = aie.tile(7, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_5 = aie.tile(7, 5)
    aie.objectfifo @A_L3L2(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo @A_L2L1(%mem_tile_0_1, {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2, %tile_4_2, %tile_5_2, %tile_6_2, %tile_7_2, %tile_0_3, %tile_1_3, %tile_2_3, %tile_3_3, %tile_4_3, %tile_5_3, %tile_6_3, %tile_7_3, %tile_0_4, %tile_1_4, %tile_2_4, %tile_3_4, %tile_4_4, %tile_5_4, %tile_6_4, %tile_7_4, %tile_0_5, %tile_1_5, %tile_2_5, %tile_3_5, %tile_4_5, %tile_5_5, %tile_6_5, %tile_7_5}, 2 : i32) : !aie.objectfifo<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo.link [@A_L3L2] -> [@A_L2L1]([] [])
    aie.objectfifo @B_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<6144xui8>> 
    aie.objectfifo @B_L2L1_0_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_0_1(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_0_2(%mem_tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_0_3(%mem_tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0_0, @B_L2L1_0_1, @B_L2L1_0_2, @B_L2L1_0_3]([] [0, 1536, 3072, 4608])
    aie.objectfifo @C_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L2L3_0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C2_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L2L3_0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16384xui8>> 
    aie.objectfifo.link [@C2_L1L2_0_0, @C2_L1L2_0_1, @C2_L1L2_0_2, @C2_L1L2_0_3] -> [@C2_L2L3_0]([0, 4096, 8192, 12288] [])
    aie.objectfifo @B_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<6144xui8>> 
    aie.objectfifo @B_L2L1_1_0(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_1_1(%mem_tile_1_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_1_2(%mem_tile_1_1, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_1_3(%mem_tile_1_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1_0, @B_L2L1_1_1, @B_L2L1_1_2, @B_L2L1_1_3]([] [0, 1536, 3072, 4608])
    aie.objectfifo @C_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_1_1(%tile_1_3, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_1_2(%tile_1_4, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_1_3(%tile_1_5, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L2L3_1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C2_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_1_1(%tile_1_3, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_1_2(%tile_1_4, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_1_3(%tile_1_5, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L2L3_1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<16384xui8>> 
    aie.objectfifo.link [@C2_L1L2_1_0, @C2_L1L2_1_1, @C2_L1L2_1_2, @C2_L1L2_1_3] -> [@C2_L2L3_1]([0, 4096, 8192, 12288] [])
    aie.objectfifo @B_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<6144xui8>> 
    aie.objectfifo @B_L2L1_2_0(%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_2_1(%mem_tile_2_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_2_2(%mem_tile_2_1, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_2_3(%mem_tile_2_1, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2_0, @B_L2L1_2_1, @B_L2L1_2_2, @B_L2L1_2_3]([] [0, 1536, 3072, 4608])
    aie.objectfifo @C_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_2_1(%tile_2_3, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_2_2(%tile_2_4, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_2_3(%tile_2_5, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L2L3_2(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C2_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_2_1(%tile_2_3, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_2_2(%tile_2_4, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_2_3(%tile_2_5, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L2L3_2(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<16384xui8>> 
    aie.objectfifo.link [@C2_L1L2_2_0, @C2_L1L2_2_1, @C2_L1L2_2_2, @C2_L1L2_2_3] -> [@C2_L2L3_2]([0, 4096, 8192, 12288] [])
    aie.objectfifo @B_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<6144xui8>> 
    aie.objectfifo @B_L2L1_3_0(%mem_tile_3_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_3_1(%mem_tile_3_1, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_3_2(%mem_tile_3_1, {%tile_3_4}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_3_3(%mem_tile_3_1, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3_0, @B_L2L1_3_1, @B_L2L1_3_2, @B_L2L1_3_3]([] [0, 1536, 3072, 4608])
    aie.objectfifo @C_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_3_1(%tile_3_3, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_3_2(%tile_3_4, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_3_3(%tile_3_5, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L2L3_3(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C2_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_3_1(%tile_3_3, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_3_2(%tile_3_4, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_3_3(%tile_3_5, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L2L3_3(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<16384xui8>> 
    aie.objectfifo.link [@C2_L1L2_3_0, @C2_L1L2_3_1, @C2_L1L2_3_2, @C2_L1L2_3_3] -> [@C2_L2L3_3]([0, 4096, 8192, 12288] [])
    aie.objectfifo @B_L3L2_4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<6144xui8>> 
    aie.objectfifo @B_L2L1_4_0(%mem_tile_4_1, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_4_1(%mem_tile_4_1, {%tile_4_3}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_4_2(%mem_tile_4_1, {%tile_4_4}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_4_3(%mem_tile_4_1, {%tile_4_5}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo.link [@B_L3L2_4] -> [@B_L2L1_4_0, @B_L2L1_4_1, @B_L2L1_4_2, @B_L2L1_4_3]([] [0, 1536, 3072, 4608])
    aie.objectfifo @C_L1L2_4_0(%tile_4_2, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_4_1(%tile_4_3, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_4_2(%tile_4_4, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_4_3(%tile_4_5, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L2L3_4(%mem_tile_4_1, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo.link [@C_L1L2_4_0, @C_L1L2_4_1, @C_L1L2_4_2, @C_L1L2_4_3] -> [@C_L2L3_4]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C2_L1L2_4_0(%tile_4_2, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_4_1(%tile_4_3, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_4_2(%tile_4_4, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_4_3(%tile_4_5, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L2L3_4(%mem_tile_4_1, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<16384xui8>> 
    aie.objectfifo.link [@C2_L1L2_4_0, @C2_L1L2_4_1, @C2_L1L2_4_2, @C2_L1L2_4_3] -> [@C2_L2L3_4]([0, 4096, 8192, 12288] [])
    aie.objectfifo @B_L3L2_5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<6144xui8>> 
    aie.objectfifo @B_L2L1_5_0(%mem_tile_5_1, {%tile_5_2}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_5_1(%mem_tile_5_1, {%tile_5_3}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_5_2(%mem_tile_5_1, {%tile_5_4}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_5_3(%mem_tile_5_1, {%tile_5_5}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo.link [@B_L3L2_5] -> [@B_L2L1_5_0, @B_L2L1_5_1, @B_L2L1_5_2, @B_L2L1_5_3]([] [0, 1536, 3072, 4608])
    aie.objectfifo @C_L1L2_5_0(%tile_5_2, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_5_1(%tile_5_3, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_5_2(%tile_5_4, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_5_3(%tile_5_5, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L2L3_5(%mem_tile_5_1, {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo.link [@C_L1L2_5_0, @C_L1L2_5_1, @C_L1L2_5_2, @C_L1L2_5_3] -> [@C_L2L3_5]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C2_L1L2_5_0(%tile_5_2, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_5_1(%tile_5_3, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_5_2(%tile_5_4, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_5_3(%tile_5_5, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L2L3_5(%mem_tile_5_1, {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<16384xui8>> 
    aie.objectfifo.link [@C2_L1L2_5_0, @C2_L1L2_5_1, @C2_L1L2_5_2, @C2_L1L2_5_3] -> [@C2_L2L3_5]([0, 4096, 8192, 12288] [])
    aie.objectfifo @B_L3L2_6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<6144xui8>> 
    aie.objectfifo @B_L2L1_6_0(%mem_tile_6_1, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_6_1(%mem_tile_6_1, {%tile_6_3}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_6_2(%mem_tile_6_1, {%tile_6_4}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_6_3(%mem_tile_6_1, {%tile_6_5}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo.link [@B_L3L2_6] -> [@B_L2L1_6_0, @B_L2L1_6_1, @B_L2L1_6_2, @B_L2L1_6_3]([] [0, 1536, 3072, 4608])
    aie.objectfifo @C_L1L2_6_0(%tile_6_2, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_6_1(%tile_6_3, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_6_2(%tile_6_4, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_6_3(%tile_6_5, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L2L3_6(%mem_tile_6_1, {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo.link [@C_L1L2_6_0, @C_L1L2_6_1, @C_L1L2_6_2, @C_L1L2_6_3] -> [@C_L2L3_6]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C2_L1L2_6_0(%tile_6_2, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_6_1(%tile_6_3, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_6_2(%tile_6_4, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_6_3(%tile_6_5, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L2L3_6(%mem_tile_6_1, {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<16384xui8>> 
    aie.objectfifo.link [@C2_L1L2_6_0, @C2_L1L2_6_1, @C2_L1L2_6_2, @C2_L1L2_6_3] -> [@C2_L2L3_6]([0, 4096, 8192, 12288] [])
    aie.objectfifo @B_L3L2_7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<6144xui8>> 
    aie.objectfifo @B_L2L1_7_0(%mem_tile_7_1, {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_7_1(%mem_tile_7_1, {%tile_7_3}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_7_2(%mem_tile_7_1, {%tile_7_4}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo @B_L2L1_7_3(%mem_tile_7_1, {%tile_7_5}, 2 : i32) : !aie.objectfifo<memref<1536xui8>> 
    aie.objectfifo.link [@B_L3L2_7] -> [@B_L2L1_7_0, @B_L2L1_7_1, @B_L2L1_7_2, @B_L2L1_7_3]([] [0, 1536, 3072, 4608])
    aie.objectfifo @C_L1L2_7_0(%tile_7_2, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_7_1(%tile_7_3, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_7_2(%tile_7_4, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L1L2_7_3(%tile_7_5, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @C_L2L3_7(%mem_tile_7_1, {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo.link [@C_L1L2_7_0, @C_L1L2_7_1, @C_L1L2_7_2, @C_L1L2_7_3] -> [@C_L2L3_7]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C2_L1L2_7_0(%tile_7_2, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_7_1(%tile_7_3, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_7_2(%tile_7_4, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L1L2_7_3(%tile_7_5, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<4096xui8>> 
    aie.objectfifo @C2_L2L3_7(%mem_tile_7_1, {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<16384xui8>> 
    aie.objectfifo.link [@C2_L1L2_7_0, @C2_L1L2_7_1, @C2_L1L2_7_2, @C2_L1L2_7_3] -> [@C2_L2L3_7]([0, 4096, 8192, 12288] [])
    %rtp_0_0 = aie.buffer(%tile_0_2) {sym_name = "rtp_0_0"} : memref<9xi32> 
    %rtp_lock_0_0 = aie.lock(%tile_0_2) {init = 0 : i32, sym_name = "rtp_lock_0_0"}
    %rtp_0_1 = aie.buffer(%tile_1_2) {sym_name = "rtp_0_1"} : memref<9xi32> 
    %rtp_lock_0_1 = aie.lock(%tile_1_2) {init = 0 : i32, sym_name = "rtp_lock_0_1"}
    %rtp_0_2 = aie.buffer(%tile_2_2) {sym_name = "rtp_0_2"} : memref<9xi32> 
    %rtp_lock_0_2 = aie.lock(%tile_2_2) {init = 0 : i32, sym_name = "rtp_lock_0_2"}
    %rtp_0_3 = aie.buffer(%tile_3_2) {sym_name = "rtp_0_3"} : memref<9xi32> 
    %rtp_lock_0_3 = aie.lock(%tile_3_2) {init = 0 : i32, sym_name = "rtp_lock_0_3"}
    %rtp_0_4 = aie.buffer(%tile_4_2) {sym_name = "rtp_0_4"} : memref<9xi32> 
    %rtp_lock_0_4 = aie.lock(%tile_4_2) {init = 0 : i32, sym_name = "rtp_lock_0_4"}
    %rtp_0_5 = aie.buffer(%tile_5_2) {sym_name = "rtp_0_5"} : memref<9xi32> 
    %rtp_lock_0_5 = aie.lock(%tile_5_2) {init = 0 : i32, sym_name = "rtp_lock_0_5"}
    %rtp_0_6 = aie.buffer(%tile_6_2) {sym_name = "rtp_0_6"} : memref<9xi32> 
    %rtp_lock_0_6 = aie.lock(%tile_6_2) {init = 0 : i32, sym_name = "rtp_lock_0_6"}
    %rtp_0_7 = aie.buffer(%tile_7_2) {sym_name = "rtp_0_7"} : memref<9xi32> 
    %rtp_lock_0_7 = aie.lock(%tile_7_2) {init = 0 : i32, sym_name = "rtp_lock_0_7"}
    %rtp_1_0 = aie.buffer(%tile_0_3) {sym_name = "rtp_1_0"} : memref<9xi32> 
    %rtp_lock_1_0 = aie.lock(%tile_0_3) {init = 0 : i32, sym_name = "rtp_lock_1_0"}
    %rtp_1_1 = aie.buffer(%tile_1_3) {sym_name = "rtp_1_1"} : memref<9xi32> 
    %rtp_lock_1_1 = aie.lock(%tile_1_3) {init = 0 : i32, sym_name = "rtp_lock_1_1"}
    %rtp_1_2 = aie.buffer(%tile_2_3) {sym_name = "rtp_1_2"} : memref<9xi32> 
    %rtp_lock_1_2 = aie.lock(%tile_2_3) {init = 0 : i32, sym_name = "rtp_lock_1_2"}
    %rtp_1_3 = aie.buffer(%tile_3_3) {sym_name = "rtp_1_3"} : memref<9xi32> 
    %rtp_lock_1_3 = aie.lock(%tile_3_3) {init = 0 : i32, sym_name = "rtp_lock_1_3"}
    %rtp_1_4 = aie.buffer(%tile_4_3) {sym_name = "rtp_1_4"} : memref<9xi32> 
    %rtp_lock_1_4 = aie.lock(%tile_4_3) {init = 0 : i32, sym_name = "rtp_lock_1_4"}
    %rtp_1_5 = aie.buffer(%tile_5_3) {sym_name = "rtp_1_5"} : memref<9xi32> 
    %rtp_lock_1_5 = aie.lock(%tile_5_3) {init = 0 : i32, sym_name = "rtp_lock_1_5"}
    %rtp_1_6 = aie.buffer(%tile_6_3) {sym_name = "rtp_1_6"} : memref<9xi32> 
    %rtp_lock_1_6 = aie.lock(%tile_6_3) {init = 0 : i32, sym_name = "rtp_lock_1_6"}
    %rtp_1_7 = aie.buffer(%tile_7_3) {sym_name = "rtp_1_7"} : memref<9xi32> 
    %rtp_lock_1_7 = aie.lock(%tile_7_3) {init = 0 : i32, sym_name = "rtp_lock_1_7"}
    %rtp_2_0 = aie.buffer(%tile_0_4) {sym_name = "rtp_2_0"} : memref<9xi32> 
    %rtp_lock_2_0 = aie.lock(%tile_0_4) {init = 0 : i32, sym_name = "rtp_lock_2_0"}
    %rtp_2_1 = aie.buffer(%tile_1_4) {sym_name = "rtp_2_1"} : memref<9xi32> 
    %rtp_lock_2_1 = aie.lock(%tile_1_4) {init = 0 : i32, sym_name = "rtp_lock_2_1"}
    %rtp_2_2 = aie.buffer(%tile_2_4) {sym_name = "rtp_2_2"} : memref<9xi32> 
    %rtp_lock_2_2 = aie.lock(%tile_2_4) {init = 0 : i32, sym_name = "rtp_lock_2_2"}
    %rtp_2_3 = aie.buffer(%tile_3_4) {sym_name = "rtp_2_3"} : memref<9xi32> 
    %rtp_lock_2_3 = aie.lock(%tile_3_4) {init = 0 : i32, sym_name = "rtp_lock_2_3"}
    %rtp_2_4 = aie.buffer(%tile_4_4) {sym_name = "rtp_2_4"} : memref<9xi32> 
    %rtp_lock_2_4 = aie.lock(%tile_4_4) {init = 0 : i32, sym_name = "rtp_lock_2_4"}
    %rtp_2_5 = aie.buffer(%tile_5_4) {sym_name = "rtp_2_5"} : memref<9xi32> 
    %rtp_lock_2_5 = aie.lock(%tile_5_4) {init = 0 : i32, sym_name = "rtp_lock_2_5"}
    %rtp_2_6 = aie.buffer(%tile_6_4) {sym_name = "rtp_2_6"} : memref<9xi32> 
    %rtp_lock_2_6 = aie.lock(%tile_6_4) {init = 0 : i32, sym_name = "rtp_lock_2_6"}
    %rtp_2_7 = aie.buffer(%tile_7_4) {sym_name = "rtp_2_7"} : memref<9xi32> 
    %rtp_lock_2_7 = aie.lock(%tile_7_4) {init = 0 : i32, sym_name = "rtp_lock_2_7"}
    %rtp_3_0 = aie.buffer(%tile_0_5) {sym_name = "rtp_3_0"} : memref<9xi32> 
    %rtp_lock_3_0 = aie.lock(%tile_0_5) {init = 0 : i32, sym_name = "rtp_lock_3_0"}
    %rtp_3_1 = aie.buffer(%tile_1_5) {sym_name = "rtp_3_1"} : memref<9xi32> 
    %rtp_lock_3_1 = aie.lock(%tile_1_5) {init = 0 : i32, sym_name = "rtp_lock_3_1"}
    %rtp_3_2 = aie.buffer(%tile_2_5) {sym_name = "rtp_3_2"} : memref<9xi32> 
    %rtp_lock_3_2 = aie.lock(%tile_2_5) {init = 0 : i32, sym_name = "rtp_lock_3_2"}
    %rtp_3_3 = aie.buffer(%tile_3_5) {sym_name = "rtp_3_3"} : memref<9xi32> 
    %rtp_lock_3_3 = aie.lock(%tile_3_5) {init = 0 : i32, sym_name = "rtp_lock_3_3"}
    %rtp_3_4 = aie.buffer(%tile_4_5) {sym_name = "rtp_3_4"} : memref<9xi32> 
    %rtp_lock_3_4 = aie.lock(%tile_4_5) {init = 0 : i32, sym_name = "rtp_lock_3_4"}
    %rtp_3_5 = aie.buffer(%tile_5_5) {sym_name = "rtp_3_5"} : memref<9xi32> 
    %rtp_lock_3_5 = aie.lock(%tile_5_5) {init = 0 : i32, sym_name = "rtp_lock_3_5"}
    %rtp_3_6 = aie.buffer(%tile_6_5) {sym_name = "rtp_3_6"} : memref<9xi32> 
    %rtp_lock_3_6 = aie.lock(%tile_6_5) {init = 0 : i32, sym_name = "rtp_lock_3_6"}
    %rtp_3_7 = aie.buffer(%tile_7_5) {sym_name = "rtp_3_7"} : memref<9xi32> 
    %rtp_lock_3_7 = aie.lock(%tile_7_5) {init = 0 : i32, sym_name = "rtp_lock_3_7"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_0_0, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_0_0[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_0_0[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_0_0[%c2] : memref<9xi32>
        %3 = memref.load %rtp_0_0[%c3] : memref<9xi32>
        %4 = memref.load %rtp_0_0[%c4] : memref<9xi32>
        %5 = memref.load %rtp_0_0[%c5] : memref<9xi32>
        %6 = memref.load %rtp_0_0[%c6] : memref<9xi32>
        %7 = memref.load %rtp_0_0[%c7] : memref<9xi32>
        %8 = memref.load %rtp_0_0[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_0_0, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %9 step %c1_10 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_11 = arith.constant 0 : index
            %c1_12 = arith.constant 1 : index
            scf.for %arg3 = %c0_11 to %13 step %c1_12 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_0_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %11 step %c1_10 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_11 = arith.constant 0 : index
            %c1_12 = arith.constant 1 : index
            scf.for %arg3 = %c0_11 to %13 step %c1_12 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_0_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_6 = arith.constant 0 : i32
        %c0_7 = arith.constant 0 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %14 step %c1_8 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_9 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c32 step %c1_10 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_11 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_12 = arith.constant 1 : index
          scf.for %arg2 = %c0_11 to %c36 step %c1_12 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c0_i32, %c0_i32_6) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          scf.for %arg2 = %c0_13 to %15 step %c1_14 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c0_i32_6) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_0_0(Consume, 1)
          }
          %c0_15 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_16 = arith.constant 1 : index
          scf.for %arg2 = %c0_15 to %c34 step %c1_16 {
            %18 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_0_1, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_0_1[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_0_1[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_0_1[%c2] : memref<9xi32>
        %3 = memref.load %rtp_0_1[%c3] : memref<9xi32>
        %4 = memref.load %rtp_0_1[%c4] : memref<9xi32>
        %5 = memref.load %rtp_0_1[%c5] : memref<9xi32>
        %6 = memref.load %rtp_0_1[%c6] : memref<9xi32>
        %7 = memref.load %rtp_0_1[%c7] : memref<9xi32>
        %8 = memref.load %rtp_0_1[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_0_1, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c0_i32, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c1_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_1_0(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_0_2, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_0_2[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_0_2[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_0_2[%c2] : memref<9xi32>
        %3 = memref.load %rtp_0_2[%c3] : memref<9xi32>
        %4 = memref.load %rtp_0_2[%c4] : memref<9xi32>
        %5 = memref.load %rtp_0_2[%c5] : memref<9xi32>
        %6 = memref.load %rtp_0_2[%c6] : memref<9xi32>
        %7 = memref.load %rtp_0_2[%c7] : memref<9xi32>
        %8 = memref.load %rtp_0_2[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_0_2, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_2_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_2_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c0_i32 = arith.constant 0 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c0_i32, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c2_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_2_0(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_0_3, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_0_3[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_0_3[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_0_3[%c2] : memref<9xi32>
        %3 = memref.load %rtp_0_3[%c3] : memref<9xi32>
        %4 = memref.load %rtp_0_3[%c4] : memref<9xi32>
        %5 = memref.load %rtp_0_3[%c5] : memref<9xi32>
        %6 = memref.load %rtp_0_3[%c6] : memref<9xi32>
        %7 = memref.load %rtp_0_3[%c7] : memref<9xi32>
        %8 = memref.load %rtp_0_3[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_0_3, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_3_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_3_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c0_i32 = arith.constant 0 : i32
        %c3_i32 = arith.constant 3 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c0_i32, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c3_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_3_0(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_0_4, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_0_4[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_0_4[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_0_4[%c2] : memref<9xi32>
        %3 = memref.load %rtp_0_4[%c3] : memref<9xi32>
        %4 = memref.load %rtp_0_4[%c4] : memref<9xi32>
        %5 = memref.load %rtp_0_4[%c5] : memref<9xi32>
        %6 = memref.load %rtp_0_4[%c6] : memref<9xi32>
        %7 = memref.load %rtp_0_4[%c7] : memref<9xi32>
        %8 = memref.load %rtp_0_4[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_0_4, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_4_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_4_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c0_i32, %c4_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c4_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_4_0(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_0_5, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_0_5[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_0_5[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_0_5[%c2] : memref<9xi32>
        %3 = memref.load %rtp_0_5[%c3] : memref<9xi32>
        %4 = memref.load %rtp_0_5[%c4] : memref<9xi32>
        %5 = memref.load %rtp_0_5[%c5] : memref<9xi32>
        %6 = memref.load %rtp_0_5[%c6] : memref<9xi32>
        %7 = memref.load %rtp_0_5[%c7] : memref<9xi32>
        %8 = memref.load %rtp_0_5[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_0_5, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_5_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_5_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c0_i32 = arith.constant 0 : i32
        %c5_i32 = arith.constant 5 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c0_i32, %c5_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c5_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_5_0(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_0_6, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_0_6[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_0_6[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_0_6[%c2] : memref<9xi32>
        %3 = memref.load %rtp_0_6[%c3] : memref<9xi32>
        %4 = memref.load %rtp_0_6[%c4] : memref<9xi32>
        %5 = memref.load %rtp_0_6[%c5] : memref<9xi32>
        %6 = memref.load %rtp_0_6[%c6] : memref<9xi32>
        %7 = memref.load %rtp_0_6[%c7] : memref<9xi32>
        %8 = memref.load %rtp_0_6[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_0_6, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_6_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_6_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c0_i32 = arith.constant 0 : i32
        %c6_i32 = arith.constant 6 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c0_i32, %c6_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c6_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_6_0(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_0_7, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_0_7[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_0_7[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_0_7[%c2] : memref<9xi32>
        %3 = memref.load %rtp_0_7[%c3] : memref<9xi32>
        %4 = memref.load %rtp_0_7[%c4] : memref<9xi32>
        %5 = memref.load %rtp_0_7[%c5] : memref<9xi32>
        %6 = memref.load %rtp_0_7[%c6] : memref<9xi32>
        %7 = memref.load %rtp_0_7[%c7] : memref<9xi32>
        %8 = memref.load %rtp_0_7[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_0_7, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_7_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_7_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c0_i32 = arith.constant 0 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c0_i32, %c7_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_0(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c7_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_7_0(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_1_0, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_1_0[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_1_0[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_1_0[%c2] : memref<9xi32>
        %3 = memref.load %rtp_1_0[%c3] : memref<9xi32>
        %4 = memref.load %rtp_1_0[%c4] : memref<9xi32>
        %5 = memref.load %rtp_1_0[%c5] : memref<9xi32>
        %6 = memref.load %rtp_1_0[%c6] : memref<9xi32>
        %7 = memref.load %rtp_1_0[%c7] : memref<9xi32>
        %8 = memref.load %rtp_1_0[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_1_0, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_0_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_0_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c1_i32 = arith.constant 1 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c1_i32, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c0_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_0_1(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_1_1, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_1_1[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_1_1[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_1_1[%c2] : memref<9xi32>
        %3 = memref.load %rtp_1_1[%c3] : memref<9xi32>
        %4 = memref.load %rtp_1_1[%c4] : memref<9xi32>
        %5 = memref.load %rtp_1_1[%c5] : memref<9xi32>
        %6 = memref.load %rtp_1_1[%c6] : memref<9xi32>
        %7 = memref.load %rtp_1_1[%c7] : memref<9xi32>
        %8 = memref.load %rtp_1_1[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_1_1, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %9 step %c1_10 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_11 = arith.constant 0 : index
            %c1_12 = arith.constant 1 : index
            scf.for %arg3 = %c0_11 to %13 step %c1_12 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %11 step %c1_10 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_11 = arith.constant 0 : index
            %c1_12 = arith.constant 1 : index
            scf.for %arg3 = %c0_11 to %13 step %c1_12 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c1_i32 = arith.constant 1 : i32
        %c1_i32_6 = arith.constant 1 : i32
        %c0_7 = arith.constant 0 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %14 step %c1_8 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_9 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c32 step %c1_10 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_11 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_12 = arith.constant 1 : index
          scf.for %arg2 = %c0_11 to %c36 step %c1_12 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c1_i32, %c1_i32_6) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          scf.for %arg2 = %c0_13 to %15 step %c1_14 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c1_i32_6) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_1_1(Consume, 1)
          }
          %c0_15 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_16 = arith.constant 1 : index
          scf.for %arg2 = %c0_15 to %c34 step %c1_16 {
            %18 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_1_2, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_1_2[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_1_2[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_1_2[%c2] : memref<9xi32>
        %3 = memref.load %rtp_1_2[%c3] : memref<9xi32>
        %4 = memref.load %rtp_1_2[%c4] : memref<9xi32>
        %5 = memref.load %rtp_1_2[%c5] : memref<9xi32>
        %6 = memref.load %rtp_1_2[%c6] : memref<9xi32>
        %7 = memref.load %rtp_1_2[%c7] : memref<9xi32>
        %8 = memref.load %rtp_1_2[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_1_2, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_2_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_2_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c1_i32, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c2_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_2_1(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_1_3, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_1_3[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_1_3[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_1_3[%c2] : memref<9xi32>
        %3 = memref.load %rtp_1_3[%c3] : memref<9xi32>
        %4 = memref.load %rtp_1_3[%c4] : memref<9xi32>
        %5 = memref.load %rtp_1_3[%c5] : memref<9xi32>
        %6 = memref.load %rtp_1_3[%c6] : memref<9xi32>
        %7 = memref.load %rtp_1_3[%c7] : memref<9xi32>
        %8 = memref.load %rtp_1_3[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_1_3, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_3_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_3_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c1_i32 = arith.constant 1 : i32
        %c3_i32 = arith.constant 3 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c1_i32, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c3_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_3_1(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_1_4, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_1_4[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_1_4[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_1_4[%c2] : memref<9xi32>
        %3 = memref.load %rtp_1_4[%c3] : memref<9xi32>
        %4 = memref.load %rtp_1_4[%c4] : memref<9xi32>
        %5 = memref.load %rtp_1_4[%c5] : memref<9xi32>
        %6 = memref.load %rtp_1_4[%c6] : memref<9xi32>
        %7 = memref.load %rtp_1_4[%c7] : memref<9xi32>
        %8 = memref.load %rtp_1_4[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_1_4, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_4_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_4_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c1_i32 = arith.constant 1 : i32
        %c4_i32 = arith.constant 4 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c1_i32, %c4_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c4_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_4_1(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_1_5, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_1_5[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_1_5[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_1_5[%c2] : memref<9xi32>
        %3 = memref.load %rtp_1_5[%c3] : memref<9xi32>
        %4 = memref.load %rtp_1_5[%c4] : memref<9xi32>
        %5 = memref.load %rtp_1_5[%c5] : memref<9xi32>
        %6 = memref.load %rtp_1_5[%c6] : memref<9xi32>
        %7 = memref.load %rtp_1_5[%c7] : memref<9xi32>
        %8 = memref.load %rtp_1_5[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_1_5, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_5_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_5_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c1_i32 = arith.constant 1 : i32
        %c5_i32 = arith.constant 5 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c1_i32, %c5_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c5_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_5_1(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_1_6, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_1_6[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_1_6[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_1_6[%c2] : memref<9xi32>
        %3 = memref.load %rtp_1_6[%c3] : memref<9xi32>
        %4 = memref.load %rtp_1_6[%c4] : memref<9xi32>
        %5 = memref.load %rtp_1_6[%c5] : memref<9xi32>
        %6 = memref.load %rtp_1_6[%c6] : memref<9xi32>
        %7 = memref.load %rtp_1_6[%c7] : memref<9xi32>
        %8 = memref.load %rtp_1_6[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_1_6, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_6_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_6_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c1_i32 = arith.constant 1 : i32
        %c6_i32 = arith.constant 6 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c1_i32, %c6_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c6_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_6_1(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_1_7, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_1_7[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_1_7[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_1_7[%c2] : memref<9xi32>
        %3 = memref.load %rtp_1_7[%c3] : memref<9xi32>
        %4 = memref.load %rtp_1_7[%c4] : memref<9xi32>
        %5 = memref.load %rtp_1_7[%c5] : memref<9xi32>
        %6 = memref.load %rtp_1_7[%c6] : memref<9xi32>
        %7 = memref.load %rtp_1_7[%c7] : memref<9xi32>
        %8 = memref.load %rtp_1_7[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_1_7, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_7_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_7_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c1_i32 = arith.constant 1 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c1_i32, %c7_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_1(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c7_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_7_1(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_2_0, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_2_0[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_2_0[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_2_0[%c2] : memref<9xi32>
        %3 = memref.load %rtp_2_0[%c3] : memref<9xi32>
        %4 = memref.load %rtp_2_0[%c4] : memref<9xi32>
        %5 = memref.load %rtp_2_0[%c5] : memref<9xi32>
        %6 = memref.load %rtp_2_0[%c6] : memref<9xi32>
        %7 = memref.load %rtp_2_0[%c7] : memref<9xi32>
        %8 = memref.load %rtp_2_0[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_2_0, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_0_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_0_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c2_i32 = arith.constant 2 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c2_i32, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c0_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_0_2(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_2_1, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_2_1[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_2_1[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_2_1[%c2] : memref<9xi32>
        %3 = memref.load %rtp_2_1[%c3] : memref<9xi32>
        %4 = memref.load %rtp_2_1[%c4] : memref<9xi32>
        %5 = memref.load %rtp_2_1[%c5] : memref<9xi32>
        %6 = memref.load %rtp_2_1[%c6] : memref<9xi32>
        %7 = memref.load %rtp_2_1[%c7] : memref<9xi32>
        %8 = memref.load %rtp_2_1[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_2_1, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c2_i32 = arith.constant 2 : i32
        %c1_i32 = arith.constant 1 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c2_i32, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c1_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_1_2(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_2_2, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_2_2[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_2_2[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_2_2[%c2] : memref<9xi32>
        %3 = memref.load %rtp_2_2[%c3] : memref<9xi32>
        %4 = memref.load %rtp_2_2[%c4] : memref<9xi32>
        %5 = memref.load %rtp_2_2[%c5] : memref<9xi32>
        %6 = memref.load %rtp_2_2[%c6] : memref<9xi32>
        %7 = memref.load %rtp_2_2[%c7] : memref<9xi32>
        %8 = memref.load %rtp_2_2[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_2_2, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %9 step %c1_10 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_11 = arith.constant 0 : index
            %c1_12 = arith.constant 1 : index
            scf.for %arg3 = %c0_11 to %13 step %c1_12 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_2_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %11 step %c1_10 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_11 = arith.constant 0 : index
            %c1_12 = arith.constant 1 : index
            scf.for %arg3 = %c0_11 to %13 step %c1_12 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_2_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_6 = arith.constant 2 : i32
        %c0_7 = arith.constant 0 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %14 step %c1_8 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_9 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c32 step %c1_10 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_11 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_12 = arith.constant 1 : index
          scf.for %arg2 = %c0_11 to %c36 step %c1_12 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c2_i32, %c2_i32_6) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          scf.for %arg2 = %c0_13 to %15 step %c1_14 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c2_i32_6) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_2_2(Consume, 1)
          }
          %c0_15 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_16 = arith.constant 1 : index
          scf.for %arg2 = %c0_15 to %c34 step %c1_16 {
            %18 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_2_3, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_2_3[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_2_3[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_2_3[%c2] : memref<9xi32>
        %3 = memref.load %rtp_2_3[%c3] : memref<9xi32>
        %4 = memref.load %rtp_2_3[%c4] : memref<9xi32>
        %5 = memref.load %rtp_2_3[%c5] : memref<9xi32>
        %6 = memref.load %rtp_2_3[%c6] : memref<9xi32>
        %7 = memref.load %rtp_2_3[%c7] : memref<9xi32>
        %8 = memref.load %rtp_2_3[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_2_3, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_3_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_3_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c2_i32 = arith.constant 2 : i32
        %c3_i32 = arith.constant 3 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c2_i32, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c3_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_3_2(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_4_4 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_2_4, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_2_4[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_2_4[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_2_4[%c2] : memref<9xi32>
        %3 = memref.load %rtp_2_4[%c3] : memref<9xi32>
        %4 = memref.load %rtp_2_4[%c4] : memref<9xi32>
        %5 = memref.load %rtp_2_4[%c5] : memref<9xi32>
        %6 = memref.load %rtp_2_4[%c6] : memref<9xi32>
        %7 = memref.load %rtp_2_4[%c7] : memref<9xi32>
        %8 = memref.load %rtp_2_4[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_2_4, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_4_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_4_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c2_i32 = arith.constant 2 : i32
        %c4_i32 = arith.constant 4 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c2_i32, %c4_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c4_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_4_2(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_5_4 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_2_5, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_2_5[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_2_5[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_2_5[%c2] : memref<9xi32>
        %3 = memref.load %rtp_2_5[%c3] : memref<9xi32>
        %4 = memref.load %rtp_2_5[%c4] : memref<9xi32>
        %5 = memref.load %rtp_2_5[%c5] : memref<9xi32>
        %6 = memref.load %rtp_2_5[%c6] : memref<9xi32>
        %7 = memref.load %rtp_2_5[%c7] : memref<9xi32>
        %8 = memref.load %rtp_2_5[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_2_5, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_5_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_5_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c2_i32 = arith.constant 2 : i32
        %c5_i32 = arith.constant 5 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c2_i32, %c5_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c5_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_5_2(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_2_6, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_2_6[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_2_6[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_2_6[%c2] : memref<9xi32>
        %3 = memref.load %rtp_2_6[%c3] : memref<9xi32>
        %4 = memref.load %rtp_2_6[%c4] : memref<9xi32>
        %5 = memref.load %rtp_2_6[%c5] : memref<9xi32>
        %6 = memref.load %rtp_2_6[%c6] : memref<9xi32>
        %7 = memref.load %rtp_2_6[%c7] : memref<9xi32>
        %8 = memref.load %rtp_2_6[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_2_6, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_6_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_6_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c2_i32 = arith.constant 2 : i32
        %c6_i32 = arith.constant 6 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c2_i32, %c6_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c6_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_6_2(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_2_7, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_2_7[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_2_7[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_2_7[%c2] : memref<9xi32>
        %3 = memref.load %rtp_2_7[%c3] : memref<9xi32>
        %4 = memref.load %rtp_2_7[%c4] : memref<9xi32>
        %5 = memref.load %rtp_2_7[%c5] : memref<9xi32>
        %6 = memref.load %rtp_2_7[%c6] : memref<9xi32>
        %7 = memref.load %rtp_2_7[%c7] : memref<9xi32>
        %8 = memref.load %rtp_2_7[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_2_7, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_7_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_7_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c2_i32 = arith.constant 2 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c2_i32, %c7_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_2(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c7_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_7_2(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_3_0, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_3_0[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_3_0[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_3_0[%c2] : memref<9xi32>
        %3 = memref.load %rtp_3_0[%c3] : memref<9xi32>
        %4 = memref.load %rtp_3_0[%c4] : memref<9xi32>
        %5 = memref.load %rtp_3_0[%c5] : memref<9xi32>
        %6 = memref.load %rtp_3_0[%c6] : memref<9xi32>
        %7 = memref.load %rtp_3_0[%c7] : memref<9xi32>
        %8 = memref.load %rtp_3_0[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_3_0, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_0_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_0_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c3_i32 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c3_i32, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_0_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c0_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_0_3(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_3_1, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_3_1[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_3_1[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_3_1[%c2] : memref<9xi32>
        %3 = memref.load %rtp_3_1[%c3] : memref<9xi32>
        %4 = memref.load %rtp_3_1[%c4] : memref<9xi32>
        %5 = memref.load %rtp_3_1[%c5] : memref<9xi32>
        %6 = memref.load %rtp_3_1[%c6] : memref<9xi32>
        %7 = memref.load %rtp_3_1[%c7] : memref<9xi32>
        %8 = memref.load %rtp_3_1[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_3_1, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c3_i32 = arith.constant 3 : i32
        %c1_i32 = arith.constant 1 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c3_i32, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_1_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c1_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_1_3(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_3_2, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_3_2[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_3_2[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_3_2[%c2] : memref<9xi32>
        %3 = memref.load %rtp_3_2[%c3] : memref<9xi32>
        %4 = memref.load %rtp_3_2[%c4] : memref<9xi32>
        %5 = memref.load %rtp_3_2[%c5] : memref<9xi32>
        %6 = memref.load %rtp_3_2[%c6] : memref<9xi32>
        %7 = memref.load %rtp_3_2[%c7] : memref<9xi32>
        %8 = memref.load %rtp_3_2[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_3_2, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_2_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_2_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c3_i32 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c3_i32, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_2_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c2_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_2_3(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_3_3, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_3_3[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_3_3[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_3_3[%c2] : memref<9xi32>
        %3 = memref.load %rtp_3_3[%c3] : memref<9xi32>
        %4 = memref.load %rtp_3_3[%c4] : memref<9xi32>
        %5 = memref.load %rtp_3_3[%c5] : memref<9xi32>
        %6 = memref.load %rtp_3_3[%c6] : memref<9xi32>
        %7 = memref.load %rtp_3_3[%c7] : memref<9xi32>
        %8 = memref.load %rtp_3_3[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_3_3, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %9 step %c1_10 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_11 = arith.constant 0 : index
            %c1_12 = arith.constant 1 : index
            scf.for %arg3 = %c0_11 to %13 step %c1_12 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_3_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %11 step %c1_10 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_11 = arith.constant 0 : index
            %c1_12 = arith.constant 1 : index
            scf.for %arg3 = %c0_11 to %13 step %c1_12 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_3_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_6 = arith.constant 3 : i32
        %c0_7 = arith.constant 0 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %14 step %c1_8 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_9 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c32 step %c1_10 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_11 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_12 = arith.constant 1 : index
          scf.for %arg2 = %c0_11 to %c36 step %c1_12 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c3_i32, %c3_i32_6) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          scf.for %arg2 = %c0_13 to %15 step %c1_14 {
            %18 = aie.objectfifo.acquire @B_L2L1_3_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c3_i32_6) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_3_3(Consume, 1)
          }
          %c0_15 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_16 = arith.constant 1 : index
          scf.for %arg2 = %c0_15 to %c34 step %c1_16 {
            %18 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_4_5 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_3_4, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_3_4[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_3_4[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_3_4[%c2] : memref<9xi32>
        %3 = memref.load %rtp_3_4[%c3] : memref<9xi32>
        %4 = memref.load %rtp_3_4[%c4] : memref<9xi32>
        %5 = memref.load %rtp_3_4[%c5] : memref<9xi32>
        %6 = memref.load %rtp_3_4[%c6] : memref<9xi32>
        %7 = memref.load %rtp_3_4[%c7] : memref<9xi32>
        %8 = memref.load %rtp_3_4[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_3_4, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_4_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_4_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c3_i32 = arith.constant 3 : i32
        %c4_i32 = arith.constant 4 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c3_i32, %c4_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_4_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c4_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_4_3(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_3_5, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_3_5[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_3_5[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_3_5[%c2] : memref<9xi32>
        %3 = memref.load %rtp_3_5[%c3] : memref<9xi32>
        %4 = memref.load %rtp_3_5[%c4] : memref<9xi32>
        %5 = memref.load %rtp_3_5[%c5] : memref<9xi32>
        %6 = memref.load %rtp_3_5[%c6] : memref<9xi32>
        %7 = memref.load %rtp_3_5[%c7] : memref<9xi32>
        %8 = memref.load %rtp_3_5[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_3_5, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_5_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_5_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c3_i32 = arith.constant 3 : i32
        %c5_i32 = arith.constant 5 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c3_i32, %c5_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_5_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c5_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_5_3(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_3_6, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_3_6[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_3_6[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_3_6[%c2] : memref<9xi32>
        %3 = memref.load %rtp_3_6[%c3] : memref<9xi32>
        %4 = memref.load %rtp_3_6[%c4] : memref<9xi32>
        %5 = memref.load %rtp_3_6[%c5] : memref<9xi32>
        %6 = memref.load %rtp_3_6[%c6] : memref<9xi32>
        %7 = memref.load %rtp_3_6[%c7] : memref<9xi32>
        %8 = memref.load %rtp_3_6[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_3_6, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_6_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_6_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c3_i32 = arith.constant 3 : i32
        %c6_i32 = arith.constant 6 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c3_i32, %c6_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_6_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c6_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_6_3(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%rtp_lock_3_7, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %rtp_3_7[%c0_0] : memref<9xi32>
        %1 = memref.load %rtp_3_7[%c1_1] : memref<9xi32>
        %2 = memref.load %rtp_3_7[%c2] : memref<9xi32>
        %3 = memref.load %rtp_3_7[%c3] : memref<9xi32>
        %4 = memref.load %rtp_3_7[%c4] : memref<9xi32>
        %5 = memref.load %rtp_3_7[%c5] : memref<9xi32>
        %6 = memref.load %rtp_3_7[%c6] : memref<9xi32>
        %7 = memref.load %rtp_3_7[%c7] : memref<9xi32>
        %8 = memref.load %rtp_3_7[%c8] : memref<9xi32>
        aie.use_lock(%rtp_lock_3_7, Release, 1)
        %9 = arith.index_cast %0 : i32 to index
        %10 = arith.index_cast %1 : i32 to index
        %11 = arith.index_cast %2 : i32 to index
        %12 = arith.index_cast %3 : i32 to index
        %13 = arith.index_cast %8 : i32 to index
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %10 step %c1_3 {
          %16 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %9 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_p1(%21, %19, %17, %0, %4, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_7_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        }
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %12 step %c1_5 {
          %16 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
          func.call @zero_kernel(%17, %8) : (memref<1024xui8>, i32) -> ()
          %c0_8 = arith.constant 0 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %11 step %c1_9 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            scf.for %arg3 = %c0_10 to %13 step %c1_11 {
              %20 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
              %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
              func.call @matmul_dn(%21, %19, %17, %2, %7, %8) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<1024xui8>, i32, i32, i32) -> ()
              aie.objectfifo.release @A_L2L1(Consume, 1)
            }
            aie.objectfifo.release @B_L2L1_7_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        }
        %14 = arith.index_cast %5 : i32 to index
        %15 = arith.index_cast %6 : i32 to index
        %c3_i32 = arith.constant 3 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %14 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
          func.call @attn_begin(%17, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          %c0_8 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c32 step %c1_9 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_q(%19, %20, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_10 = arith.constant 0 : index
          %c36 = arith.constant 36 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg2 = %c0_10 to %c36 step %c1_11 {
            %18 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x8x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x8x!aiex.bfp<"v8bfp16ebs8">>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_tail(%19, %20, %c3_i32, %c7_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg2 = %c0_12 to %15 step %c1_13 {
            %18 = aie.objectfifo.acquire @B_L2L1_7_3(Consume, 1) : !aie.objectfifosubview<memref<1536xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1536xui8>> -> memref<1536xui8>
            func.call @attn_feed(%19, %c7_i32) : (memref<1536xui8>, i32) -> ()
            aie.objectfifo.release @B_L2L1_7_3(Consume, 1)
          }
          %c0_14 = arith.constant 0 : index
          %c34 = arith.constant 34 : index
          %c1_15 = arith.constant 1 : index
          scf.for %arg2 = %c0_14 to %c34 step %c1_15 {
            %18 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
            %20 = arith.index_cast %arg2 : index to i32
            func.call @attn_out(%19, %20) : (memref<1024xui8>, i32) -> ()
            aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
          }
        }
      }
      aie.end
    } {stack_size = 3840 : i32}
    aie.runtime_sequence(%arg0: memref<4096x!aiex.bfp<"v8bfp16ebs8">>, %arg1: memref<3145728xui8>, %arg2: memref<65536xui8>) {
      aiex.npu.rtp_write(@rtp_0_0, 0, 0)
      aiex.npu.rtp_write(@rtp_0_0, 1, 0)
      aiex.npu.rtp_write(@rtp_0_0, 2, 32)
      aiex.npu.rtp_write(@rtp_0_0, 3, 2)
      aiex.npu.rtp_write(@rtp_0_0, 4, 0)
      aiex.npu.rtp_write(@rtp_0_0, 5, 0)
      aiex.npu.rtp_write(@rtp_0_0, 6, 0)
      aiex.npu.rtp_write(@rtp_0_0, 7, 0)
      aiex.npu.rtp_write(@rtp_0_0, 8, 1)
      aiex.set_lock(%rtp_lock_0_0, 1)
      aiex.npu.rtp_write(@rtp_0_1, 0, 0)
      aiex.npu.rtp_write(@rtp_0_1, 1, 0)
      aiex.npu.rtp_write(@rtp_0_1, 2, 32)
      aiex.npu.rtp_write(@rtp_0_1, 3, 2)
      aiex.npu.rtp_write(@rtp_0_1, 4, 0)
      aiex.npu.rtp_write(@rtp_0_1, 5, 0)
      aiex.npu.rtp_write(@rtp_0_1, 6, 0)
      aiex.npu.rtp_write(@rtp_0_1, 7, 0)
      aiex.npu.rtp_write(@rtp_0_1, 8, 1)
      aiex.set_lock(%rtp_lock_0_1, 1)
      aiex.npu.rtp_write(@rtp_0_2, 0, 0)
      aiex.npu.rtp_write(@rtp_0_2, 1, 0)
      aiex.npu.rtp_write(@rtp_0_2, 2, 32)
      aiex.npu.rtp_write(@rtp_0_2, 3, 2)
      aiex.npu.rtp_write(@rtp_0_2, 4, 0)
      aiex.npu.rtp_write(@rtp_0_2, 5, 0)
      aiex.npu.rtp_write(@rtp_0_2, 6, 0)
      aiex.npu.rtp_write(@rtp_0_2, 7, 0)
      aiex.npu.rtp_write(@rtp_0_2, 8, 1)
      aiex.set_lock(%rtp_lock_0_2, 1)
      aiex.npu.rtp_write(@rtp_0_3, 0, 0)
      aiex.npu.rtp_write(@rtp_0_3, 1, 0)
      aiex.npu.rtp_write(@rtp_0_3, 2, 32)
      aiex.npu.rtp_write(@rtp_0_3, 3, 2)
      aiex.npu.rtp_write(@rtp_0_3, 4, 0)
      aiex.npu.rtp_write(@rtp_0_3, 5, 0)
      aiex.npu.rtp_write(@rtp_0_3, 6, 0)
      aiex.npu.rtp_write(@rtp_0_3, 7, 0)
      aiex.npu.rtp_write(@rtp_0_3, 8, 1)
      aiex.set_lock(%rtp_lock_0_3, 1)
      aiex.npu.rtp_write(@rtp_0_4, 0, 0)
      aiex.npu.rtp_write(@rtp_0_4, 1, 0)
      aiex.npu.rtp_write(@rtp_0_4, 2, 32)
      aiex.npu.rtp_write(@rtp_0_4, 3, 2)
      aiex.npu.rtp_write(@rtp_0_4, 4, 0)
      aiex.npu.rtp_write(@rtp_0_4, 5, 0)
      aiex.npu.rtp_write(@rtp_0_4, 6, 0)
      aiex.npu.rtp_write(@rtp_0_4, 7, 0)
      aiex.npu.rtp_write(@rtp_0_4, 8, 1)
      aiex.set_lock(%rtp_lock_0_4, 1)
      aiex.npu.rtp_write(@rtp_0_5, 0, 0)
      aiex.npu.rtp_write(@rtp_0_5, 1, 0)
      aiex.npu.rtp_write(@rtp_0_5, 2, 32)
      aiex.npu.rtp_write(@rtp_0_5, 3, 2)
      aiex.npu.rtp_write(@rtp_0_5, 4, 0)
      aiex.npu.rtp_write(@rtp_0_5, 5, 0)
      aiex.npu.rtp_write(@rtp_0_5, 6, 0)
      aiex.npu.rtp_write(@rtp_0_5, 7, 0)
      aiex.npu.rtp_write(@rtp_0_5, 8, 1)
      aiex.set_lock(%rtp_lock_0_5, 1)
      aiex.npu.rtp_write(@rtp_0_6, 0, 0)
      aiex.npu.rtp_write(@rtp_0_6, 1, 0)
      aiex.npu.rtp_write(@rtp_0_6, 2, 32)
      aiex.npu.rtp_write(@rtp_0_6, 3, 2)
      aiex.npu.rtp_write(@rtp_0_6, 4, 0)
      aiex.npu.rtp_write(@rtp_0_6, 5, 0)
      aiex.npu.rtp_write(@rtp_0_6, 6, 0)
      aiex.npu.rtp_write(@rtp_0_6, 7, 0)
      aiex.npu.rtp_write(@rtp_0_6, 8, 1)
      aiex.set_lock(%rtp_lock_0_6, 1)
      aiex.npu.rtp_write(@rtp_0_7, 0, 0)
      aiex.npu.rtp_write(@rtp_0_7, 1, 0)
      aiex.npu.rtp_write(@rtp_0_7, 2, 32)
      aiex.npu.rtp_write(@rtp_0_7, 3, 2)
      aiex.npu.rtp_write(@rtp_0_7, 4, 0)
      aiex.npu.rtp_write(@rtp_0_7, 5, 0)
      aiex.npu.rtp_write(@rtp_0_7, 6, 0)
      aiex.npu.rtp_write(@rtp_0_7, 7, 0)
      aiex.npu.rtp_write(@rtp_0_7, 8, 1)
      aiex.set_lock(%rtp_lock_0_7, 1)
      aiex.npu.rtp_write(@rtp_1_0, 0, 0)
      aiex.npu.rtp_write(@rtp_1_0, 1, 0)
      aiex.npu.rtp_write(@rtp_1_0, 2, 32)
      aiex.npu.rtp_write(@rtp_1_0, 3, 2)
      aiex.npu.rtp_write(@rtp_1_0, 4, 0)
      aiex.npu.rtp_write(@rtp_1_0, 5, 0)
      aiex.npu.rtp_write(@rtp_1_0, 6, 0)
      aiex.npu.rtp_write(@rtp_1_0, 7, 0)
      aiex.npu.rtp_write(@rtp_1_0, 8, 1)
      aiex.set_lock(%rtp_lock_1_0, 1)
      aiex.npu.rtp_write(@rtp_1_1, 0, 0)
      aiex.npu.rtp_write(@rtp_1_1, 1, 0)
      aiex.npu.rtp_write(@rtp_1_1, 2, 32)
      aiex.npu.rtp_write(@rtp_1_1, 3, 2)
      aiex.npu.rtp_write(@rtp_1_1, 4, 0)
      aiex.npu.rtp_write(@rtp_1_1, 5, 0)
      aiex.npu.rtp_write(@rtp_1_1, 6, 0)
      aiex.npu.rtp_write(@rtp_1_1, 7, 0)
      aiex.npu.rtp_write(@rtp_1_1, 8, 1)
      aiex.set_lock(%rtp_lock_1_1, 1)
      aiex.npu.rtp_write(@rtp_1_2, 0, 0)
      aiex.npu.rtp_write(@rtp_1_2, 1, 0)
      aiex.npu.rtp_write(@rtp_1_2, 2, 32)
      aiex.npu.rtp_write(@rtp_1_2, 3, 2)
      aiex.npu.rtp_write(@rtp_1_2, 4, 0)
      aiex.npu.rtp_write(@rtp_1_2, 5, 0)
      aiex.npu.rtp_write(@rtp_1_2, 6, 0)
      aiex.npu.rtp_write(@rtp_1_2, 7, 0)
      aiex.npu.rtp_write(@rtp_1_2, 8, 1)
      aiex.set_lock(%rtp_lock_1_2, 1)
      aiex.npu.rtp_write(@rtp_1_3, 0, 0)
      aiex.npu.rtp_write(@rtp_1_3, 1, 0)
      aiex.npu.rtp_write(@rtp_1_3, 2, 32)
      aiex.npu.rtp_write(@rtp_1_3, 3, 2)
      aiex.npu.rtp_write(@rtp_1_3, 4, 0)
      aiex.npu.rtp_write(@rtp_1_3, 5, 0)
      aiex.npu.rtp_write(@rtp_1_3, 6, 0)
      aiex.npu.rtp_write(@rtp_1_3, 7, 0)
      aiex.npu.rtp_write(@rtp_1_3, 8, 1)
      aiex.set_lock(%rtp_lock_1_3, 1)
      aiex.npu.rtp_write(@rtp_1_4, 0, 0)
      aiex.npu.rtp_write(@rtp_1_4, 1, 0)
      aiex.npu.rtp_write(@rtp_1_4, 2, 32)
      aiex.npu.rtp_write(@rtp_1_4, 3, 2)
      aiex.npu.rtp_write(@rtp_1_4, 4, 0)
      aiex.npu.rtp_write(@rtp_1_4, 5, 0)
      aiex.npu.rtp_write(@rtp_1_4, 6, 0)
      aiex.npu.rtp_write(@rtp_1_4, 7, 0)
      aiex.npu.rtp_write(@rtp_1_4, 8, 1)
      aiex.set_lock(%rtp_lock_1_4, 1)
      aiex.npu.rtp_write(@rtp_1_5, 0, 0)
      aiex.npu.rtp_write(@rtp_1_5, 1, 0)
      aiex.npu.rtp_write(@rtp_1_5, 2, 32)
      aiex.npu.rtp_write(@rtp_1_5, 3, 2)
      aiex.npu.rtp_write(@rtp_1_5, 4, 0)
      aiex.npu.rtp_write(@rtp_1_5, 5, 0)
      aiex.npu.rtp_write(@rtp_1_5, 6, 0)
      aiex.npu.rtp_write(@rtp_1_5, 7, 0)
      aiex.npu.rtp_write(@rtp_1_5, 8, 1)
      aiex.set_lock(%rtp_lock_1_5, 1)
      aiex.npu.rtp_write(@rtp_1_6, 0, 0)
      aiex.npu.rtp_write(@rtp_1_6, 1, 0)
      aiex.npu.rtp_write(@rtp_1_6, 2, 32)
      aiex.npu.rtp_write(@rtp_1_6, 3, 2)
      aiex.npu.rtp_write(@rtp_1_6, 4, 0)
      aiex.npu.rtp_write(@rtp_1_6, 5, 0)
      aiex.npu.rtp_write(@rtp_1_6, 6, 0)
      aiex.npu.rtp_write(@rtp_1_6, 7, 0)
      aiex.npu.rtp_write(@rtp_1_6, 8, 1)
      aiex.set_lock(%rtp_lock_1_6, 1)
      aiex.npu.rtp_write(@rtp_1_7, 0, 0)
      aiex.npu.rtp_write(@rtp_1_7, 1, 0)
      aiex.npu.rtp_write(@rtp_1_7, 2, 32)
      aiex.npu.rtp_write(@rtp_1_7, 3, 2)
      aiex.npu.rtp_write(@rtp_1_7, 4, 0)
      aiex.npu.rtp_write(@rtp_1_7, 5, 0)
      aiex.npu.rtp_write(@rtp_1_7, 6, 0)
      aiex.npu.rtp_write(@rtp_1_7, 7, 0)
      aiex.npu.rtp_write(@rtp_1_7, 8, 1)
      aiex.set_lock(%rtp_lock_1_7, 1)
      aiex.npu.rtp_write(@rtp_2_0, 0, 0)
      aiex.npu.rtp_write(@rtp_2_0, 1, 0)
      aiex.npu.rtp_write(@rtp_2_0, 2, 32)
      aiex.npu.rtp_write(@rtp_2_0, 3, 2)
      aiex.npu.rtp_write(@rtp_2_0, 4, 0)
      aiex.npu.rtp_write(@rtp_2_0, 5, 0)
      aiex.npu.rtp_write(@rtp_2_0, 6, 0)
      aiex.npu.rtp_write(@rtp_2_0, 7, 0)
      aiex.npu.rtp_write(@rtp_2_0, 8, 1)
      aiex.set_lock(%rtp_lock_2_0, 1)
      aiex.npu.rtp_write(@rtp_2_1, 0, 0)
      aiex.npu.rtp_write(@rtp_2_1, 1, 0)
      aiex.npu.rtp_write(@rtp_2_1, 2, 32)
      aiex.npu.rtp_write(@rtp_2_1, 3, 2)
      aiex.npu.rtp_write(@rtp_2_1, 4, 0)
      aiex.npu.rtp_write(@rtp_2_1, 5, 0)
      aiex.npu.rtp_write(@rtp_2_1, 6, 0)
      aiex.npu.rtp_write(@rtp_2_1, 7, 0)
      aiex.npu.rtp_write(@rtp_2_1, 8, 1)
      aiex.set_lock(%rtp_lock_2_1, 1)
      aiex.npu.rtp_write(@rtp_2_2, 0, 0)
      aiex.npu.rtp_write(@rtp_2_2, 1, 0)
      aiex.npu.rtp_write(@rtp_2_2, 2, 32)
      aiex.npu.rtp_write(@rtp_2_2, 3, 2)
      aiex.npu.rtp_write(@rtp_2_2, 4, 0)
      aiex.npu.rtp_write(@rtp_2_2, 5, 0)
      aiex.npu.rtp_write(@rtp_2_2, 6, 0)
      aiex.npu.rtp_write(@rtp_2_2, 7, 0)
      aiex.npu.rtp_write(@rtp_2_2, 8, 1)
      aiex.set_lock(%rtp_lock_2_2, 1)
      aiex.npu.rtp_write(@rtp_2_3, 0, 0)
      aiex.npu.rtp_write(@rtp_2_3, 1, 0)
      aiex.npu.rtp_write(@rtp_2_3, 2, 32)
      aiex.npu.rtp_write(@rtp_2_3, 3, 2)
      aiex.npu.rtp_write(@rtp_2_3, 4, 0)
      aiex.npu.rtp_write(@rtp_2_3, 5, 0)
      aiex.npu.rtp_write(@rtp_2_3, 6, 0)
      aiex.npu.rtp_write(@rtp_2_3, 7, 0)
      aiex.npu.rtp_write(@rtp_2_3, 8, 1)
      aiex.set_lock(%rtp_lock_2_3, 1)
      aiex.npu.rtp_write(@rtp_2_4, 0, 0)
      aiex.npu.rtp_write(@rtp_2_4, 1, 0)
      aiex.npu.rtp_write(@rtp_2_4, 2, 32)
      aiex.npu.rtp_write(@rtp_2_4, 3, 2)
      aiex.npu.rtp_write(@rtp_2_4, 4, 0)
      aiex.npu.rtp_write(@rtp_2_4, 5, 0)
      aiex.npu.rtp_write(@rtp_2_4, 6, 0)
      aiex.npu.rtp_write(@rtp_2_4, 7, 0)
      aiex.npu.rtp_write(@rtp_2_4, 8, 1)
      aiex.set_lock(%rtp_lock_2_4, 1)
      aiex.npu.rtp_write(@rtp_2_5, 0, 0)
      aiex.npu.rtp_write(@rtp_2_5, 1, 0)
      aiex.npu.rtp_write(@rtp_2_5, 2, 32)
      aiex.npu.rtp_write(@rtp_2_5, 3, 2)
      aiex.npu.rtp_write(@rtp_2_5, 4, 0)
      aiex.npu.rtp_write(@rtp_2_5, 5, 0)
      aiex.npu.rtp_write(@rtp_2_5, 6, 0)
      aiex.npu.rtp_write(@rtp_2_5, 7, 0)
      aiex.npu.rtp_write(@rtp_2_5, 8, 1)
      aiex.set_lock(%rtp_lock_2_5, 1)
      aiex.npu.rtp_write(@rtp_2_6, 0, 0)
      aiex.npu.rtp_write(@rtp_2_6, 1, 0)
      aiex.npu.rtp_write(@rtp_2_6, 2, 32)
      aiex.npu.rtp_write(@rtp_2_6, 3, 2)
      aiex.npu.rtp_write(@rtp_2_6, 4, 0)
      aiex.npu.rtp_write(@rtp_2_6, 5, 0)
      aiex.npu.rtp_write(@rtp_2_6, 6, 0)
      aiex.npu.rtp_write(@rtp_2_6, 7, 0)
      aiex.npu.rtp_write(@rtp_2_6, 8, 1)
      aiex.set_lock(%rtp_lock_2_6, 1)
      aiex.npu.rtp_write(@rtp_2_7, 0, 0)
      aiex.npu.rtp_write(@rtp_2_7, 1, 0)
      aiex.npu.rtp_write(@rtp_2_7, 2, 32)
      aiex.npu.rtp_write(@rtp_2_7, 3, 2)
      aiex.npu.rtp_write(@rtp_2_7, 4, 0)
      aiex.npu.rtp_write(@rtp_2_7, 5, 0)
      aiex.npu.rtp_write(@rtp_2_7, 6, 0)
      aiex.npu.rtp_write(@rtp_2_7, 7, 0)
      aiex.npu.rtp_write(@rtp_2_7, 8, 1)
      aiex.set_lock(%rtp_lock_2_7, 1)
      aiex.npu.rtp_write(@rtp_3_0, 0, 0)
      aiex.npu.rtp_write(@rtp_3_0, 1, 0)
      aiex.npu.rtp_write(@rtp_3_0, 2, 32)
      aiex.npu.rtp_write(@rtp_3_0, 3, 2)
      aiex.npu.rtp_write(@rtp_3_0, 4, 0)
      aiex.npu.rtp_write(@rtp_3_0, 5, 0)
      aiex.npu.rtp_write(@rtp_3_0, 6, 0)
      aiex.npu.rtp_write(@rtp_3_0, 7, 0)
      aiex.npu.rtp_write(@rtp_3_0, 8, 1)
      aiex.set_lock(%rtp_lock_3_0, 1)
      aiex.npu.rtp_write(@rtp_3_1, 0, 0)
      aiex.npu.rtp_write(@rtp_3_1, 1, 0)
      aiex.npu.rtp_write(@rtp_3_1, 2, 32)
      aiex.npu.rtp_write(@rtp_3_1, 3, 2)
      aiex.npu.rtp_write(@rtp_3_1, 4, 0)
      aiex.npu.rtp_write(@rtp_3_1, 5, 0)
      aiex.npu.rtp_write(@rtp_3_1, 6, 0)
      aiex.npu.rtp_write(@rtp_3_1, 7, 0)
      aiex.npu.rtp_write(@rtp_3_1, 8, 1)
      aiex.set_lock(%rtp_lock_3_1, 1)
      aiex.npu.rtp_write(@rtp_3_2, 0, 0)
      aiex.npu.rtp_write(@rtp_3_2, 1, 0)
      aiex.npu.rtp_write(@rtp_3_2, 2, 32)
      aiex.npu.rtp_write(@rtp_3_2, 3, 2)
      aiex.npu.rtp_write(@rtp_3_2, 4, 0)
      aiex.npu.rtp_write(@rtp_3_2, 5, 0)
      aiex.npu.rtp_write(@rtp_3_2, 6, 0)
      aiex.npu.rtp_write(@rtp_3_2, 7, 0)
      aiex.npu.rtp_write(@rtp_3_2, 8, 1)
      aiex.set_lock(%rtp_lock_3_2, 1)
      aiex.npu.rtp_write(@rtp_3_3, 0, 0)
      aiex.npu.rtp_write(@rtp_3_3, 1, 0)
      aiex.npu.rtp_write(@rtp_3_3, 2, 32)
      aiex.npu.rtp_write(@rtp_3_3, 3, 2)
      aiex.npu.rtp_write(@rtp_3_3, 4, 0)
      aiex.npu.rtp_write(@rtp_3_3, 5, 0)
      aiex.npu.rtp_write(@rtp_3_3, 6, 0)
      aiex.npu.rtp_write(@rtp_3_3, 7, 0)
      aiex.npu.rtp_write(@rtp_3_3, 8, 1)
      aiex.set_lock(%rtp_lock_3_3, 1)
      aiex.npu.rtp_write(@rtp_3_4, 0, 0)
      aiex.npu.rtp_write(@rtp_3_4, 1, 0)
      aiex.npu.rtp_write(@rtp_3_4, 2, 32)
      aiex.npu.rtp_write(@rtp_3_4, 3, 2)
      aiex.npu.rtp_write(@rtp_3_4, 4, 0)
      aiex.npu.rtp_write(@rtp_3_4, 5, 0)
      aiex.npu.rtp_write(@rtp_3_4, 6, 0)
      aiex.npu.rtp_write(@rtp_3_4, 7, 0)
      aiex.npu.rtp_write(@rtp_3_4, 8, 1)
      aiex.set_lock(%rtp_lock_3_4, 1)
      aiex.npu.rtp_write(@rtp_3_5, 0, 0)
      aiex.npu.rtp_write(@rtp_3_5, 1, 0)
      aiex.npu.rtp_write(@rtp_3_5, 2, 32)
      aiex.npu.rtp_write(@rtp_3_5, 3, 2)
      aiex.npu.rtp_write(@rtp_3_5, 4, 0)
      aiex.npu.rtp_write(@rtp_3_5, 5, 0)
      aiex.npu.rtp_write(@rtp_3_5, 6, 0)
      aiex.npu.rtp_write(@rtp_3_5, 7, 0)
      aiex.npu.rtp_write(@rtp_3_5, 8, 1)
      aiex.set_lock(%rtp_lock_3_5, 1)
      aiex.npu.rtp_write(@rtp_3_6, 0, 0)
      aiex.npu.rtp_write(@rtp_3_6, 1, 0)
      aiex.npu.rtp_write(@rtp_3_6, 2, 32)
      aiex.npu.rtp_write(@rtp_3_6, 3, 2)
      aiex.npu.rtp_write(@rtp_3_6, 4, 0)
      aiex.npu.rtp_write(@rtp_3_6, 5, 0)
      aiex.npu.rtp_write(@rtp_3_6, 6, 0)
      aiex.npu.rtp_write(@rtp_3_6, 7, 0)
      aiex.npu.rtp_write(@rtp_3_6, 8, 1)
      aiex.set_lock(%rtp_lock_3_6, 1)
      aiex.npu.rtp_write(@rtp_3_7, 0, 0)
      aiex.npu.rtp_write(@rtp_3_7, 1, 0)
      aiex.npu.rtp_write(@rtp_3_7, 2, 32)
      aiex.npu.rtp_write(@rtp_3_7, 3, 2)
      aiex.npu.rtp_write(@rtp_3_7, 4, 0)
      aiex.npu.rtp_write(@rtp_3_7, 5, 0)
      aiex.npu.rtp_write(@rtp_3_7, 6, 0)
      aiex.npu.rtp_write(@rtp_3_7, 7, 0)
      aiex.npu.rtp_write(@rtp_3_7, 8, 1)
      aiex.set_lock(%rtp_lock_3_7, 1)
      %0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<4096x!aiex.bfp<"v8bfp16ebs8">>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 0, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 196608, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 393216, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 589824, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 786432, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 983040, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 1179648, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 1376256, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 4096, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 8192, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 12288, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 16384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 20480, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 24576, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 28672, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<4096x!aiex.bfp<"v8bfp16ebs8">>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 1572864, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 1769472, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 1966080, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 2162688, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 2359296, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 2555904, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 2752512, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3145728xui8>, 2949120, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 32768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 36864, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 40960, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 45056, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 49152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 53248, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 57344, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<65536xui8>, 61440, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      aiex.dma_await_task(%9)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%16)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_await_task(%26)
      aiex.dma_await_task(%27)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%31)
      aiex.dma_await_task(%32)
      aiex.dma_await_task(%33)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%20)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
    }
  }
}
