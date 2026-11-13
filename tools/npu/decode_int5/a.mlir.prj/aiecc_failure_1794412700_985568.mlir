module {
  aie.device(npu2) {
    func.func private @zero_kernel(memref<4096xui8>, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @matmul_p1(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @matmul_dn(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_begin(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_q(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_tail(memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_feed(memref<1536xui8>, i32) attributes {link_with = "mm_16x64x128.o"}
    func.func private @attn_out(memref<4096xui8>, i32) attributes {link_with = "mm_16x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_4_1 = aie.tile(4, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_5_1 = aie.tile(5, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_6_1 = aie.tile(6, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_7_1 = aie.tile(7, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_4_2 = aie.tile(4, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_5_2 = aie.tile(5, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_6_2 = aie.tile(6, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_7_2 = aie.tile(7, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_4_3 = aie.tile(4, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_5_3 = aie.tile(5, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_6_3 = aie.tile(6, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_7_3 = aie.tile(7, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_4_4 = aie.tile(4, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_5_4 = aie.tile(5, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_6_4 = aie.tile(6, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_7_4 = aie.tile(7, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_4_5 = aie.tile(4, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_5_5 = aie.tile(5, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_6_5 = aie.tile(6, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_7_5 = aie.tile(7, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %C_L2L3_7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "C_L2L3_7_cons_prod_lock_0"}
    %C_L2L3_7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "C_L2L3_7_cons_cons_lock_0"}
    %C_L2L3_7_buff_0 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, sym_name = "C_L2L3_7_buff_0"} : memref<16384xui8> 
    %C_L2L3_7_buff_1 = aie.buffer(%mem_tile_7_1) {address = 16384 : i32, sym_name = "C_L2L3_7_buff_1"} : memref<16384xui8> 
    %C_L2L3_7_prod_lock_0 = aie.lock(%mem_tile_7_1, 8) {init = 2 : i32, sym_name = "C_L2L3_7_prod_lock_0"}
    %C_L2L3_7_cons_lock_0 = aie.lock(%mem_tile_7_1, 9) {init = 0 : i32, sym_name = "C_L2L3_7_cons_lock_0"}
    %C_L2L3_7_prod_lock_1 = aie.lock(%mem_tile_7_1, 10) {init = 2 : i32, sym_name = "C_L2L3_7_prod_lock_1"}
    %C_L2L3_7_cons_lock_1 = aie.lock(%mem_tile_7_1, 11) {init = 0 : i32, sym_name = "C_L2L3_7_cons_lock_1"}
    %C_L2L3_7_prod_lock_2 = aie.lock(%mem_tile_7_1, 12) {init = 2 : i32, sym_name = "C_L2L3_7_prod_lock_2"}
    %C_L2L3_7_cons_lock_2 = aie.lock(%mem_tile_7_1, 13) {init = 0 : i32, sym_name = "C_L2L3_7_cons_lock_2"}
    %C_L2L3_7_prod_lock_3 = aie.lock(%mem_tile_7_1, 14) {init = 2 : i32, sym_name = "C_L2L3_7_prod_lock_3"}
    %C_L2L3_7_cons_lock_3 = aie.lock(%mem_tile_7_1, 15) {init = 0 : i32, sym_name = "C_L2L3_7_cons_lock_3"}
    %C_L1L2_7_3_buff_0 = aie.buffer(%tile_7_5) {address = 3840 : i32, sym_name = "C_L1L2_7_3_buff_0"} : memref<4096xui8> 
    %C_L1L2_7_3_prod_lock_0 = aie.lock(%tile_7_5, 5) {init = 1 : i32, sym_name = "C_L1L2_7_3_prod_lock_0"}
    %C_L1L2_7_3_cons_lock_0 = aie.lock(%tile_7_5, 6) {init = 0 : i32, sym_name = "C_L1L2_7_3_cons_lock_0"}
    %C_L1L2_7_2_buff_0 = aie.buffer(%tile_7_4) {address = 3840 : i32, sym_name = "C_L1L2_7_2_buff_0"} : memref<4096xui8> 
    %C_L1L2_7_2_prod_lock_0 = aie.lock(%tile_7_4, 5) {init = 1 : i32, sym_name = "C_L1L2_7_2_prod_lock_0"}
    %C_L1L2_7_2_cons_lock_0 = aie.lock(%tile_7_4, 6) {init = 0 : i32, sym_name = "C_L1L2_7_2_cons_lock_0"}
    %C_L1L2_7_1_buff_0 = aie.buffer(%tile_7_3) {address = 3840 : i32, sym_name = "C_L1L2_7_1_buff_0"} : memref<4096xui8> 
    %C_L1L2_7_1_prod_lock_0 = aie.lock(%tile_7_3, 5) {init = 1 : i32, sym_name = "C_L1L2_7_1_prod_lock_0"}
    %C_L1L2_7_1_cons_lock_0 = aie.lock(%tile_7_3, 6) {init = 0 : i32, sym_name = "C_L1L2_7_1_cons_lock_0"}
    %C_L1L2_7_0_buff_0 = aie.buffer(%tile_7_2) {address = 3840 : i32, sym_name = "C_L1L2_7_0_buff_0"} : memref<4096xui8> 
    %C_L1L2_7_0_prod_lock_0 = aie.lock(%tile_7_2, 5) {init = 1 : i32, sym_name = "C_L1L2_7_0_prod_lock_0"}
    %C_L1L2_7_0_cons_lock_0 = aie.lock(%tile_7_2, 6) {init = 0 : i32, sym_name = "C_L1L2_7_0_cons_lock_0"}
    %B_L2L1_7_3_cons_buff_0 = aie.buffer(%tile_7_5) {address = 7936 : i32, sym_name = "B_L2L1_7_3_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_7_3_cons_buff_1 = aie.buffer(%tile_7_5) {address = 9472 : i32, sym_name = "B_L2L1_7_3_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_7_3_cons_prod_lock_0 = aie.lock(%tile_7_5, 3) {init = 2 : i32, sym_name = "B_L2L1_7_3_cons_prod_lock_0"}
    %B_L2L1_7_3_cons_cons_lock_0 = aie.lock(%tile_7_5, 4) {init = 0 : i32, sym_name = "B_L2L1_7_3_cons_cons_lock_0"}
    %B_L2L1_7_2_cons_buff_0 = aie.buffer(%tile_7_4) {address = 7936 : i32, sym_name = "B_L2L1_7_2_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_7_2_cons_buff_1 = aie.buffer(%tile_7_4) {address = 9472 : i32, sym_name = "B_L2L1_7_2_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_7_2_cons_prod_lock_0 = aie.lock(%tile_7_4, 3) {init = 2 : i32, sym_name = "B_L2L1_7_2_cons_prod_lock_0"}
    %B_L2L1_7_2_cons_cons_lock_0 = aie.lock(%tile_7_4, 4) {init = 0 : i32, sym_name = "B_L2L1_7_2_cons_cons_lock_0"}
    %B_L2L1_7_1_cons_buff_0 = aie.buffer(%tile_7_3) {address = 7936 : i32, sym_name = "B_L2L1_7_1_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_7_1_cons_buff_1 = aie.buffer(%tile_7_3) {address = 9472 : i32, sym_name = "B_L2L1_7_1_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_7_1_cons_prod_lock_0 = aie.lock(%tile_7_3, 3) {init = 2 : i32, sym_name = "B_L2L1_7_1_cons_prod_lock_0"}
    %B_L2L1_7_1_cons_cons_lock_0 = aie.lock(%tile_7_3, 4) {init = 0 : i32, sym_name = "B_L2L1_7_1_cons_cons_lock_0"}
    %B_L2L1_7_0_cons_buff_0 = aie.buffer(%tile_7_2) {address = 7936 : i32, sym_name = "B_L2L1_7_0_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_7_0_cons_buff_1 = aie.buffer(%tile_7_2) {address = 9472 : i32, sym_name = "B_L2L1_7_0_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_7_0_cons_prod_lock_0 = aie.lock(%tile_7_2, 3) {init = 2 : i32, sym_name = "B_L2L1_7_0_cons_prod_lock_0"}
    %B_L2L1_7_0_cons_cons_lock_0 = aie.lock(%tile_7_2, 4) {init = 0 : i32, sym_name = "B_L2L1_7_0_cons_cons_lock_0"}
    %B_L3L2_7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 32768 : i32, sym_name = "B_L3L2_7_cons_buff_0"} : memref<6144xui8> 
    %B_L3L2_7_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 38912 : i32, sym_name = "B_L3L2_7_cons_buff_1"} : memref<6144xui8> 
    %B_L3L2_7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 0) {init = 2 : i32, sym_name = "B_L3L2_7_cons_prod_lock_0"}
    %B_L3L2_7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "B_L3L2_7_cons_cons_lock_0"}
    %B_L3L2_7_cons_prod_lock_1 = aie.lock(%mem_tile_7_1, 2) {init = 2 : i32, sym_name = "B_L3L2_7_cons_prod_lock_1"}
    %B_L3L2_7_cons_cons_lock_1 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "B_L3L2_7_cons_cons_lock_1"}
    %B_L3L2_7_cons_prod_lock_2 = aie.lock(%mem_tile_7_1, 4) {init = 2 : i32, sym_name = "B_L3L2_7_cons_prod_lock_2"}
    %B_L3L2_7_cons_cons_lock_2 = aie.lock(%mem_tile_7_1, 5) {init = 0 : i32, sym_name = "B_L3L2_7_cons_cons_lock_2"}
    %B_L3L2_7_cons_prod_lock_3 = aie.lock(%mem_tile_7_1, 6) {init = 2 : i32, sym_name = "B_L3L2_7_cons_prod_lock_3"}
    %B_L3L2_7_cons_cons_lock_3 = aie.lock(%mem_tile_7_1, 7) {init = 0 : i32, sym_name = "B_L3L2_7_cons_cons_lock_3"}
    %B_L3L2_7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "B_L3L2_7_prod_lock_0"}
    %B_L3L2_7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "B_L3L2_7_cons_lock_0"}
    %C_L2L3_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "C_L2L3_6_cons_prod_lock_0"}
    %C_L2L3_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "C_L2L3_6_cons_cons_lock_0"}
    %C_L2L3_6_buff_0 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, sym_name = "C_L2L3_6_buff_0"} : memref<16384xui8> 
    %C_L2L3_6_buff_1 = aie.buffer(%mem_tile_6_1) {address = 16384 : i32, sym_name = "C_L2L3_6_buff_1"} : memref<16384xui8> 
    %C_L2L3_6_prod_lock_0 = aie.lock(%mem_tile_6_1, 8) {init = 2 : i32, sym_name = "C_L2L3_6_prod_lock_0"}
    %C_L2L3_6_cons_lock_0 = aie.lock(%mem_tile_6_1, 9) {init = 0 : i32, sym_name = "C_L2L3_6_cons_lock_0"}
    %C_L2L3_6_prod_lock_1 = aie.lock(%mem_tile_6_1, 10) {init = 2 : i32, sym_name = "C_L2L3_6_prod_lock_1"}
    %C_L2L3_6_cons_lock_1 = aie.lock(%mem_tile_6_1, 11) {init = 0 : i32, sym_name = "C_L2L3_6_cons_lock_1"}
    %C_L2L3_6_prod_lock_2 = aie.lock(%mem_tile_6_1, 12) {init = 2 : i32, sym_name = "C_L2L3_6_prod_lock_2"}
    %C_L2L3_6_cons_lock_2 = aie.lock(%mem_tile_6_1, 13) {init = 0 : i32, sym_name = "C_L2L3_6_cons_lock_2"}
    %C_L2L3_6_prod_lock_3 = aie.lock(%mem_tile_6_1, 14) {init = 2 : i32, sym_name = "C_L2L3_6_prod_lock_3"}
    %C_L2L3_6_cons_lock_3 = aie.lock(%mem_tile_6_1, 15) {init = 0 : i32, sym_name = "C_L2L3_6_cons_lock_3"}
    %C_L1L2_6_3_buff_0 = aie.buffer(%tile_6_5) {address = 3840 : i32, sym_name = "C_L1L2_6_3_buff_0"} : memref<4096xui8> 
    %C_L1L2_6_3_prod_lock_0 = aie.lock(%tile_6_5, 5) {init = 1 : i32, sym_name = "C_L1L2_6_3_prod_lock_0"}
    %C_L1L2_6_3_cons_lock_0 = aie.lock(%tile_6_5, 6) {init = 0 : i32, sym_name = "C_L1L2_6_3_cons_lock_0"}
    %C_L1L2_6_2_buff_0 = aie.buffer(%tile_6_4) {address = 3840 : i32, sym_name = "C_L1L2_6_2_buff_0"} : memref<4096xui8> 
    %C_L1L2_6_2_prod_lock_0 = aie.lock(%tile_6_4, 5) {init = 1 : i32, sym_name = "C_L1L2_6_2_prod_lock_0"}
    %C_L1L2_6_2_cons_lock_0 = aie.lock(%tile_6_4, 6) {init = 0 : i32, sym_name = "C_L1L2_6_2_cons_lock_0"}
    %C_L1L2_6_1_buff_0 = aie.buffer(%tile_6_3) {address = 3840 : i32, sym_name = "C_L1L2_6_1_buff_0"} : memref<4096xui8> 
    %C_L1L2_6_1_prod_lock_0 = aie.lock(%tile_6_3, 5) {init = 1 : i32, sym_name = "C_L1L2_6_1_prod_lock_0"}
    %C_L1L2_6_1_cons_lock_0 = aie.lock(%tile_6_3, 6) {init = 0 : i32, sym_name = "C_L1L2_6_1_cons_lock_0"}
    %C_L1L2_6_0_buff_0 = aie.buffer(%tile_6_2) {address = 3840 : i32, sym_name = "C_L1L2_6_0_buff_0"} : memref<4096xui8> 
    %C_L1L2_6_0_prod_lock_0 = aie.lock(%tile_6_2, 5) {init = 1 : i32, sym_name = "C_L1L2_6_0_prod_lock_0"}
    %C_L1L2_6_0_cons_lock_0 = aie.lock(%tile_6_2, 6) {init = 0 : i32, sym_name = "C_L1L2_6_0_cons_lock_0"}
    %B_L2L1_6_3_cons_buff_0 = aie.buffer(%tile_6_5) {address = 7936 : i32, sym_name = "B_L2L1_6_3_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_6_3_cons_buff_1 = aie.buffer(%tile_6_5) {address = 9472 : i32, sym_name = "B_L2L1_6_3_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_6_3_cons_prod_lock_0 = aie.lock(%tile_6_5, 3) {init = 2 : i32, sym_name = "B_L2L1_6_3_cons_prod_lock_0"}
    %B_L2L1_6_3_cons_cons_lock_0 = aie.lock(%tile_6_5, 4) {init = 0 : i32, sym_name = "B_L2L1_6_3_cons_cons_lock_0"}
    %B_L2L1_6_2_cons_buff_0 = aie.buffer(%tile_6_4) {address = 7936 : i32, sym_name = "B_L2L1_6_2_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_6_2_cons_buff_1 = aie.buffer(%tile_6_4) {address = 9472 : i32, sym_name = "B_L2L1_6_2_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_6_2_cons_prod_lock_0 = aie.lock(%tile_6_4, 3) {init = 2 : i32, sym_name = "B_L2L1_6_2_cons_prod_lock_0"}
    %B_L2L1_6_2_cons_cons_lock_0 = aie.lock(%tile_6_4, 4) {init = 0 : i32, sym_name = "B_L2L1_6_2_cons_cons_lock_0"}
    %B_L2L1_6_1_cons_buff_0 = aie.buffer(%tile_6_3) {address = 7936 : i32, sym_name = "B_L2L1_6_1_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_6_1_cons_buff_1 = aie.buffer(%tile_6_3) {address = 9472 : i32, sym_name = "B_L2L1_6_1_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_6_1_cons_prod_lock_0 = aie.lock(%tile_6_3, 3) {init = 2 : i32, sym_name = "B_L2L1_6_1_cons_prod_lock_0"}
    %B_L2L1_6_1_cons_cons_lock_0 = aie.lock(%tile_6_3, 4) {init = 0 : i32, sym_name = "B_L2L1_6_1_cons_cons_lock_0"}
    %B_L2L1_6_0_cons_buff_0 = aie.buffer(%tile_6_2) {address = 7936 : i32, sym_name = "B_L2L1_6_0_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_6_0_cons_buff_1 = aie.buffer(%tile_6_2) {address = 9472 : i32, sym_name = "B_L2L1_6_0_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_6_0_cons_prod_lock_0 = aie.lock(%tile_6_2, 3) {init = 2 : i32, sym_name = "B_L2L1_6_0_cons_prod_lock_0"}
    %B_L2L1_6_0_cons_cons_lock_0 = aie.lock(%tile_6_2, 4) {init = 0 : i32, sym_name = "B_L2L1_6_0_cons_cons_lock_0"}
    %B_L3L2_6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 32768 : i32, sym_name = "B_L3L2_6_cons_buff_0"} : memref<6144xui8> 
    %B_L3L2_6_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 38912 : i32, sym_name = "B_L3L2_6_cons_buff_1"} : memref<6144xui8> 
    %B_L3L2_6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 0) {init = 2 : i32, sym_name = "B_L3L2_6_cons_prod_lock_0"}
    %B_L3L2_6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "B_L3L2_6_cons_cons_lock_0"}
    %B_L3L2_6_cons_prod_lock_1 = aie.lock(%mem_tile_6_1, 2) {init = 2 : i32, sym_name = "B_L3L2_6_cons_prod_lock_1"}
    %B_L3L2_6_cons_cons_lock_1 = aie.lock(%mem_tile_6_1, 3) {init = 0 : i32, sym_name = "B_L3L2_6_cons_cons_lock_1"}
    %B_L3L2_6_cons_prod_lock_2 = aie.lock(%mem_tile_6_1, 4) {init = 2 : i32, sym_name = "B_L3L2_6_cons_prod_lock_2"}
    %B_L3L2_6_cons_cons_lock_2 = aie.lock(%mem_tile_6_1, 5) {init = 0 : i32, sym_name = "B_L3L2_6_cons_cons_lock_2"}
    %B_L3L2_6_cons_prod_lock_3 = aie.lock(%mem_tile_6_1, 6) {init = 2 : i32, sym_name = "B_L3L2_6_cons_prod_lock_3"}
    %B_L3L2_6_cons_cons_lock_3 = aie.lock(%mem_tile_6_1, 7) {init = 0 : i32, sym_name = "B_L3L2_6_cons_cons_lock_3"}
    %B_L3L2_6_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "B_L3L2_6_prod_lock_0"}
    %B_L3L2_6_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "B_L3L2_6_cons_lock_0"}
    %C_L2L3_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "C_L2L3_5_cons_prod_lock_0"}
    %C_L2L3_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "C_L2L3_5_cons_cons_lock_0"}
    %C_L2L3_5_buff_0 = aie.buffer(%mem_tile_5_1) {address = 0 : i32, sym_name = "C_L2L3_5_buff_0"} : memref<16384xui8> 
    %C_L2L3_5_buff_1 = aie.buffer(%mem_tile_5_1) {address = 16384 : i32, sym_name = "C_L2L3_5_buff_1"} : memref<16384xui8> 
    %C_L2L3_5_prod_lock_0 = aie.lock(%mem_tile_5_1, 8) {init = 2 : i32, sym_name = "C_L2L3_5_prod_lock_0"}
    %C_L2L3_5_cons_lock_0 = aie.lock(%mem_tile_5_1, 9) {init = 0 : i32, sym_name = "C_L2L3_5_cons_lock_0"}
    %C_L2L3_5_prod_lock_1 = aie.lock(%mem_tile_5_1, 10) {init = 2 : i32, sym_name = "C_L2L3_5_prod_lock_1"}
    %C_L2L3_5_cons_lock_1 = aie.lock(%mem_tile_5_1, 11) {init = 0 : i32, sym_name = "C_L2L3_5_cons_lock_1"}
    %C_L2L3_5_prod_lock_2 = aie.lock(%mem_tile_5_1, 12) {init = 2 : i32, sym_name = "C_L2L3_5_prod_lock_2"}
    %C_L2L3_5_cons_lock_2 = aie.lock(%mem_tile_5_1, 13) {init = 0 : i32, sym_name = "C_L2L3_5_cons_lock_2"}
    %C_L2L3_5_prod_lock_3 = aie.lock(%mem_tile_5_1, 14) {init = 2 : i32, sym_name = "C_L2L3_5_prod_lock_3"}
    %C_L2L3_5_cons_lock_3 = aie.lock(%mem_tile_5_1, 15) {init = 0 : i32, sym_name = "C_L2L3_5_cons_lock_3"}
    %C_L1L2_5_3_buff_0 = aie.buffer(%tile_5_5) {address = 3840 : i32, sym_name = "C_L1L2_5_3_buff_0"} : memref<4096xui8> 
    %C_L1L2_5_3_prod_lock_0 = aie.lock(%tile_5_5, 5) {init = 1 : i32, sym_name = "C_L1L2_5_3_prod_lock_0"}
    %C_L1L2_5_3_cons_lock_0 = aie.lock(%tile_5_5, 6) {init = 0 : i32, sym_name = "C_L1L2_5_3_cons_lock_0"}
    %C_L1L2_5_2_buff_0 = aie.buffer(%tile_5_4) {address = 3840 : i32, sym_name = "C_L1L2_5_2_buff_0"} : memref<4096xui8> 
    %C_L1L2_5_2_prod_lock_0 = aie.lock(%tile_5_4, 5) {init = 1 : i32, sym_name = "C_L1L2_5_2_prod_lock_0"}
    %C_L1L2_5_2_cons_lock_0 = aie.lock(%tile_5_4, 6) {init = 0 : i32, sym_name = "C_L1L2_5_2_cons_lock_0"}
    %C_L1L2_5_1_buff_0 = aie.buffer(%tile_5_3) {address = 3840 : i32, sym_name = "C_L1L2_5_1_buff_0"} : memref<4096xui8> 
    %C_L1L2_5_1_prod_lock_0 = aie.lock(%tile_5_3, 5) {init = 1 : i32, sym_name = "C_L1L2_5_1_prod_lock_0"}
    %C_L1L2_5_1_cons_lock_0 = aie.lock(%tile_5_3, 6) {init = 0 : i32, sym_name = "C_L1L2_5_1_cons_lock_0"}
    %C_L1L2_5_0_buff_0 = aie.buffer(%tile_5_2) {address = 3840 : i32, sym_name = "C_L1L2_5_0_buff_0"} : memref<4096xui8> 
    %C_L1L2_5_0_prod_lock_0 = aie.lock(%tile_5_2, 5) {init = 1 : i32, sym_name = "C_L1L2_5_0_prod_lock_0"}
    %C_L1L2_5_0_cons_lock_0 = aie.lock(%tile_5_2, 6) {init = 0 : i32, sym_name = "C_L1L2_5_0_cons_lock_0"}
    %B_L2L1_5_3_cons_buff_0 = aie.buffer(%tile_5_5) {address = 7936 : i32, sym_name = "B_L2L1_5_3_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_5_3_cons_buff_1 = aie.buffer(%tile_5_5) {address = 9472 : i32, sym_name = "B_L2L1_5_3_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_5_3_cons_prod_lock_0 = aie.lock(%tile_5_5, 3) {init = 2 : i32, sym_name = "B_L2L1_5_3_cons_prod_lock_0"}
    %B_L2L1_5_3_cons_cons_lock_0 = aie.lock(%tile_5_5, 4) {init = 0 : i32, sym_name = "B_L2L1_5_3_cons_cons_lock_0"}
    %B_L2L1_5_2_cons_buff_0 = aie.buffer(%tile_5_4) {address = 7936 : i32, sym_name = "B_L2L1_5_2_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_5_2_cons_buff_1 = aie.buffer(%tile_5_4) {address = 9472 : i32, sym_name = "B_L2L1_5_2_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_5_2_cons_prod_lock_0 = aie.lock(%tile_5_4, 3) {init = 2 : i32, sym_name = "B_L2L1_5_2_cons_prod_lock_0"}
    %B_L2L1_5_2_cons_cons_lock_0 = aie.lock(%tile_5_4, 4) {init = 0 : i32, sym_name = "B_L2L1_5_2_cons_cons_lock_0"}
    %B_L2L1_5_1_cons_buff_0 = aie.buffer(%tile_5_3) {address = 7936 : i32, sym_name = "B_L2L1_5_1_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_5_1_cons_buff_1 = aie.buffer(%tile_5_3) {address = 9472 : i32, sym_name = "B_L2L1_5_1_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_5_1_cons_prod_lock_0 = aie.lock(%tile_5_3, 3) {init = 2 : i32, sym_name = "B_L2L1_5_1_cons_prod_lock_0"}
    %B_L2L1_5_1_cons_cons_lock_0 = aie.lock(%tile_5_3, 4) {init = 0 : i32, sym_name = "B_L2L1_5_1_cons_cons_lock_0"}
    %B_L2L1_5_0_cons_buff_0 = aie.buffer(%tile_5_2) {address = 7936 : i32, sym_name = "B_L2L1_5_0_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_5_0_cons_buff_1 = aie.buffer(%tile_5_2) {address = 9472 : i32, sym_name = "B_L2L1_5_0_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_5_0_cons_prod_lock_0 = aie.lock(%tile_5_2, 3) {init = 2 : i32, sym_name = "B_L2L1_5_0_cons_prod_lock_0"}
    %B_L2L1_5_0_cons_cons_lock_0 = aie.lock(%tile_5_2, 4) {init = 0 : i32, sym_name = "B_L2L1_5_0_cons_cons_lock_0"}
    %B_L3L2_5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 32768 : i32, sym_name = "B_L3L2_5_cons_buff_0"} : memref<6144xui8> 
    %B_L3L2_5_cons_buff_1 = aie.buffer(%mem_tile_5_1) {address = 38912 : i32, sym_name = "B_L3L2_5_cons_buff_1"} : memref<6144xui8> 
    %B_L3L2_5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 0) {init = 2 : i32, sym_name = "B_L3L2_5_cons_prod_lock_0"}
    %B_L3L2_5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 1) {init = 0 : i32, sym_name = "B_L3L2_5_cons_cons_lock_0"}
    %B_L3L2_5_cons_prod_lock_1 = aie.lock(%mem_tile_5_1, 2) {init = 2 : i32, sym_name = "B_L3L2_5_cons_prod_lock_1"}
    %B_L3L2_5_cons_cons_lock_1 = aie.lock(%mem_tile_5_1, 3) {init = 0 : i32, sym_name = "B_L3L2_5_cons_cons_lock_1"}
    %B_L3L2_5_cons_prod_lock_2 = aie.lock(%mem_tile_5_1, 4) {init = 2 : i32, sym_name = "B_L3L2_5_cons_prod_lock_2"}
    %B_L3L2_5_cons_cons_lock_2 = aie.lock(%mem_tile_5_1, 5) {init = 0 : i32, sym_name = "B_L3L2_5_cons_cons_lock_2"}
    %B_L3L2_5_cons_prod_lock_3 = aie.lock(%mem_tile_5_1, 6) {init = 2 : i32, sym_name = "B_L3L2_5_cons_prod_lock_3"}
    %B_L3L2_5_cons_cons_lock_3 = aie.lock(%mem_tile_5_1, 7) {init = 0 : i32, sym_name = "B_L3L2_5_cons_cons_lock_3"}
    %B_L3L2_5_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "B_L3L2_5_prod_lock_0"}
    %B_L3L2_5_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "B_L3L2_5_cons_lock_0"}
    %C_L2L3_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "C_L2L3_4_cons_prod_lock_0"}
    %C_L2L3_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "C_L2L3_4_cons_cons_lock_0"}
    %C_L2L3_4_buff_0 = aie.buffer(%mem_tile_4_1) {address = 0 : i32, sym_name = "C_L2L3_4_buff_0"} : memref<16384xui8> 
    %C_L2L3_4_buff_1 = aie.buffer(%mem_tile_4_1) {address = 16384 : i32, sym_name = "C_L2L3_4_buff_1"} : memref<16384xui8> 
    %C_L2L3_4_prod_lock_0 = aie.lock(%mem_tile_4_1, 8) {init = 2 : i32, sym_name = "C_L2L3_4_prod_lock_0"}
    %C_L2L3_4_cons_lock_0 = aie.lock(%mem_tile_4_1, 9) {init = 0 : i32, sym_name = "C_L2L3_4_cons_lock_0"}
    %C_L2L3_4_prod_lock_1 = aie.lock(%mem_tile_4_1, 10) {init = 2 : i32, sym_name = "C_L2L3_4_prod_lock_1"}
    %C_L2L3_4_cons_lock_1 = aie.lock(%mem_tile_4_1, 11) {init = 0 : i32, sym_name = "C_L2L3_4_cons_lock_1"}
    %C_L2L3_4_prod_lock_2 = aie.lock(%mem_tile_4_1, 12) {init = 2 : i32, sym_name = "C_L2L3_4_prod_lock_2"}
    %C_L2L3_4_cons_lock_2 = aie.lock(%mem_tile_4_1, 13) {init = 0 : i32, sym_name = "C_L2L3_4_cons_lock_2"}
    %C_L2L3_4_prod_lock_3 = aie.lock(%mem_tile_4_1, 14) {init = 2 : i32, sym_name = "C_L2L3_4_prod_lock_3"}
    %C_L2L3_4_cons_lock_3 = aie.lock(%mem_tile_4_1, 15) {init = 0 : i32, sym_name = "C_L2L3_4_cons_lock_3"}
    %C_L1L2_4_3_buff_0 = aie.buffer(%tile_4_5) {address = 3840 : i32, sym_name = "C_L1L2_4_3_buff_0"} : memref<4096xui8> 
    %C_L1L2_4_3_prod_lock_0 = aie.lock(%tile_4_5, 5) {init = 1 : i32, sym_name = "C_L1L2_4_3_prod_lock_0"}
    %C_L1L2_4_3_cons_lock_0 = aie.lock(%tile_4_5, 6) {init = 0 : i32, sym_name = "C_L1L2_4_3_cons_lock_0"}
    %C_L1L2_4_2_buff_0 = aie.buffer(%tile_4_4) {address = 3840 : i32, sym_name = "C_L1L2_4_2_buff_0"} : memref<4096xui8> 
    %C_L1L2_4_2_prod_lock_0 = aie.lock(%tile_4_4, 5) {init = 1 : i32, sym_name = "C_L1L2_4_2_prod_lock_0"}
    %C_L1L2_4_2_cons_lock_0 = aie.lock(%tile_4_4, 6) {init = 0 : i32, sym_name = "C_L1L2_4_2_cons_lock_0"}
    %C_L1L2_4_1_buff_0 = aie.buffer(%tile_4_3) {address = 3840 : i32, sym_name = "C_L1L2_4_1_buff_0"} : memref<4096xui8> 
    %C_L1L2_4_1_prod_lock_0 = aie.lock(%tile_4_3, 5) {init = 1 : i32, sym_name = "C_L1L2_4_1_prod_lock_0"}
    %C_L1L2_4_1_cons_lock_0 = aie.lock(%tile_4_3, 6) {init = 0 : i32, sym_name = "C_L1L2_4_1_cons_lock_0"}
    %C_L1L2_4_0_buff_0 = aie.buffer(%tile_4_2) {address = 3840 : i32, sym_name = "C_L1L2_4_0_buff_0"} : memref<4096xui8> 
    %C_L1L2_4_0_prod_lock_0 = aie.lock(%tile_4_2, 5) {init = 1 : i32, sym_name = "C_L1L2_4_0_prod_lock_0"}
    %C_L1L2_4_0_cons_lock_0 = aie.lock(%tile_4_2, 6) {init = 0 : i32, sym_name = "C_L1L2_4_0_cons_lock_0"}
    %B_L2L1_4_3_cons_buff_0 = aie.buffer(%tile_4_5) {address = 7936 : i32, sym_name = "B_L2L1_4_3_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_4_3_cons_buff_1 = aie.buffer(%tile_4_5) {address = 9472 : i32, sym_name = "B_L2L1_4_3_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_4_3_cons_prod_lock_0 = aie.lock(%tile_4_5, 3) {init = 2 : i32, sym_name = "B_L2L1_4_3_cons_prod_lock_0"}
    %B_L2L1_4_3_cons_cons_lock_0 = aie.lock(%tile_4_5, 4) {init = 0 : i32, sym_name = "B_L2L1_4_3_cons_cons_lock_0"}
    %B_L2L1_4_2_cons_buff_0 = aie.buffer(%tile_4_4) {address = 7936 : i32, sym_name = "B_L2L1_4_2_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_4_2_cons_buff_1 = aie.buffer(%tile_4_4) {address = 9472 : i32, sym_name = "B_L2L1_4_2_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_4_2_cons_prod_lock_0 = aie.lock(%tile_4_4, 3) {init = 2 : i32, sym_name = "B_L2L1_4_2_cons_prod_lock_0"}
    %B_L2L1_4_2_cons_cons_lock_0 = aie.lock(%tile_4_4, 4) {init = 0 : i32, sym_name = "B_L2L1_4_2_cons_cons_lock_0"}
    %B_L2L1_4_1_cons_buff_0 = aie.buffer(%tile_4_3) {address = 7936 : i32, sym_name = "B_L2L1_4_1_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_4_1_cons_buff_1 = aie.buffer(%tile_4_3) {address = 9472 : i32, sym_name = "B_L2L1_4_1_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_4_1_cons_prod_lock_0 = aie.lock(%tile_4_3, 3) {init = 2 : i32, sym_name = "B_L2L1_4_1_cons_prod_lock_0"}
    %B_L2L1_4_1_cons_cons_lock_0 = aie.lock(%tile_4_3, 4) {init = 0 : i32, sym_name = "B_L2L1_4_1_cons_cons_lock_0"}
    %B_L2L1_4_0_cons_buff_0 = aie.buffer(%tile_4_2) {address = 7936 : i32, sym_name = "B_L2L1_4_0_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_4_0_cons_buff_1 = aie.buffer(%tile_4_2) {address = 9472 : i32, sym_name = "B_L2L1_4_0_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_4_0_cons_prod_lock_0 = aie.lock(%tile_4_2, 3) {init = 2 : i32, sym_name = "B_L2L1_4_0_cons_prod_lock_0"}
    %B_L2L1_4_0_cons_cons_lock_0 = aie.lock(%tile_4_2, 4) {init = 0 : i32, sym_name = "B_L2L1_4_0_cons_cons_lock_0"}
    %B_L3L2_4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 32768 : i32, sym_name = "B_L3L2_4_cons_buff_0"} : memref<6144xui8> 
    %B_L3L2_4_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 38912 : i32, sym_name = "B_L3L2_4_cons_buff_1"} : memref<6144xui8> 
    %B_L3L2_4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 0) {init = 2 : i32, sym_name = "B_L3L2_4_cons_prod_lock_0"}
    %B_L3L2_4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 1) {init = 0 : i32, sym_name = "B_L3L2_4_cons_cons_lock_0"}
    %B_L3L2_4_cons_prod_lock_1 = aie.lock(%mem_tile_4_1, 2) {init = 2 : i32, sym_name = "B_L3L2_4_cons_prod_lock_1"}
    %B_L3L2_4_cons_cons_lock_1 = aie.lock(%mem_tile_4_1, 3) {init = 0 : i32, sym_name = "B_L3L2_4_cons_cons_lock_1"}
    %B_L3L2_4_cons_prod_lock_2 = aie.lock(%mem_tile_4_1, 4) {init = 2 : i32, sym_name = "B_L3L2_4_cons_prod_lock_2"}
    %B_L3L2_4_cons_cons_lock_2 = aie.lock(%mem_tile_4_1, 5) {init = 0 : i32, sym_name = "B_L3L2_4_cons_cons_lock_2"}
    %B_L3L2_4_cons_prod_lock_3 = aie.lock(%mem_tile_4_1, 6) {init = 2 : i32, sym_name = "B_L3L2_4_cons_prod_lock_3"}
    %B_L3L2_4_cons_cons_lock_3 = aie.lock(%mem_tile_4_1, 7) {init = 0 : i32, sym_name = "B_L3L2_4_cons_cons_lock_3"}
    %B_L3L2_4_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "B_L3L2_4_prod_lock_0"}
    %B_L3L2_4_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "B_L3L2_4_cons_lock_0"}
    %C_L2L3_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "C_L2L3_3_cons_prod_lock_0"}
    %C_L2L3_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "C_L2L3_3_cons_cons_lock_0"}
    %C_L2L3_3_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "C_L2L3_3_buff_0"} : memref<16384xui8> 
    %C_L2L3_3_buff_1 = aie.buffer(%mem_tile_3_1) {address = 16384 : i32, sym_name = "C_L2L3_3_buff_1"} : memref<16384xui8> 
    %C_L2L3_3_prod_lock_0 = aie.lock(%mem_tile_3_1, 8) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_0"}
    %C_L2L3_3_cons_lock_0 = aie.lock(%mem_tile_3_1, 9) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_0"}
    %C_L2L3_3_prod_lock_1 = aie.lock(%mem_tile_3_1, 10) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_1"}
    %C_L2L3_3_cons_lock_1 = aie.lock(%mem_tile_3_1, 11) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_1"}
    %C_L2L3_3_prod_lock_2 = aie.lock(%mem_tile_3_1, 12) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_2"}
    %C_L2L3_3_cons_lock_2 = aie.lock(%mem_tile_3_1, 13) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_2"}
    %C_L2L3_3_prod_lock_3 = aie.lock(%mem_tile_3_1, 14) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_3"}
    %C_L2L3_3_cons_lock_3 = aie.lock(%mem_tile_3_1, 15) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_3"}
    %C_L1L2_3_3_buff_0 = aie.buffer(%tile_3_5) {address = 3840 : i32, sym_name = "C_L1L2_3_3_buff_0"} : memref<4096xui8> 
    %C_L1L2_3_3_prod_lock_0 = aie.lock(%tile_3_5, 5) {init = 1 : i32, sym_name = "C_L1L2_3_3_prod_lock_0"}
    %C_L1L2_3_3_cons_lock_0 = aie.lock(%tile_3_5, 6) {init = 0 : i32, sym_name = "C_L1L2_3_3_cons_lock_0"}
    %C_L1L2_3_2_buff_0 = aie.buffer(%tile_3_4) {address = 3840 : i32, sym_name = "C_L1L2_3_2_buff_0"} : memref<4096xui8> 
    %C_L1L2_3_2_prod_lock_0 = aie.lock(%tile_3_4, 5) {init = 1 : i32, sym_name = "C_L1L2_3_2_prod_lock_0"}
    %C_L1L2_3_2_cons_lock_0 = aie.lock(%tile_3_4, 6) {init = 0 : i32, sym_name = "C_L1L2_3_2_cons_lock_0"}
    %C_L1L2_3_1_buff_0 = aie.buffer(%tile_3_3) {address = 3840 : i32, sym_name = "C_L1L2_3_1_buff_0"} : memref<4096xui8> 
    %C_L1L2_3_1_prod_lock_0 = aie.lock(%tile_3_3, 5) {init = 1 : i32, sym_name = "C_L1L2_3_1_prod_lock_0"}
    %C_L1L2_3_1_cons_lock_0 = aie.lock(%tile_3_3, 6) {init = 0 : i32, sym_name = "C_L1L2_3_1_cons_lock_0"}
    %C_L1L2_3_0_buff_0 = aie.buffer(%tile_3_2) {address = 3840 : i32, sym_name = "C_L1L2_3_0_buff_0"} : memref<4096xui8> 
    %C_L1L2_3_0_prod_lock_0 = aie.lock(%tile_3_2, 5) {init = 1 : i32, sym_name = "C_L1L2_3_0_prod_lock_0"}
    %C_L1L2_3_0_cons_lock_0 = aie.lock(%tile_3_2, 6) {init = 0 : i32, sym_name = "C_L1L2_3_0_cons_lock_0"}
    %B_L2L1_3_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 7936 : i32, sym_name = "B_L2L1_3_3_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_3_3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 9472 : i32, sym_name = "B_L2L1_3_3_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_3_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 3) {init = 2 : i32, sym_name = "B_L2L1_3_3_cons_prod_lock_0"}
    %B_L2L1_3_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 4) {init = 0 : i32, sym_name = "B_L2L1_3_3_cons_cons_lock_0"}
    %B_L2L1_3_2_cons_buff_0 = aie.buffer(%tile_3_4) {address = 7936 : i32, sym_name = "B_L2L1_3_2_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_3_2_cons_buff_1 = aie.buffer(%tile_3_4) {address = 9472 : i32, sym_name = "B_L2L1_3_2_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_3_2_cons_prod_lock_0 = aie.lock(%tile_3_4, 3) {init = 2 : i32, sym_name = "B_L2L1_3_2_cons_prod_lock_0"}
    %B_L2L1_3_2_cons_cons_lock_0 = aie.lock(%tile_3_4, 4) {init = 0 : i32, sym_name = "B_L2L1_3_2_cons_cons_lock_0"}
    %B_L2L1_3_1_cons_buff_0 = aie.buffer(%tile_3_3) {address = 7936 : i32, sym_name = "B_L2L1_3_1_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_3_1_cons_buff_1 = aie.buffer(%tile_3_3) {address = 9472 : i32, sym_name = "B_L2L1_3_1_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_3_1_cons_prod_lock_0 = aie.lock(%tile_3_3, 3) {init = 2 : i32, sym_name = "B_L2L1_3_1_cons_prod_lock_0"}
    %B_L2L1_3_1_cons_cons_lock_0 = aie.lock(%tile_3_3, 4) {init = 0 : i32, sym_name = "B_L2L1_3_1_cons_cons_lock_0"}
    %B_L2L1_3_0_cons_buff_0 = aie.buffer(%tile_3_2) {address = 7936 : i32, sym_name = "B_L2L1_3_0_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_3_0_cons_buff_1 = aie.buffer(%tile_3_2) {address = 9472 : i32, sym_name = "B_L2L1_3_0_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_3_0_cons_prod_lock_0 = aie.lock(%tile_3_2, 3) {init = 2 : i32, sym_name = "B_L2L1_3_0_cons_prod_lock_0"}
    %B_L2L1_3_0_cons_cons_lock_0 = aie.lock(%tile_3_2, 4) {init = 0 : i32, sym_name = "B_L2L1_3_0_cons_cons_lock_0"}
    %B_L3L2_3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 32768 : i32, sym_name = "B_L3L2_3_cons_buff_0"} : memref<6144xui8> 
    %B_L3L2_3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 38912 : i32, sym_name = "B_L3L2_3_cons_buff_1"} : memref<6144xui8> 
    %B_L3L2_3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "B_L3L2_3_cons_prod_lock_0"}
    %B_L3L2_3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "B_L3L2_3_cons_cons_lock_0"}
    %B_L3L2_3_cons_prod_lock_1 = aie.lock(%mem_tile_3_1, 2) {init = 2 : i32, sym_name = "B_L3L2_3_cons_prod_lock_1"}
    %B_L3L2_3_cons_cons_lock_1 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "B_L3L2_3_cons_cons_lock_1"}
    %B_L3L2_3_cons_prod_lock_2 = aie.lock(%mem_tile_3_1, 4) {init = 2 : i32, sym_name = "B_L3L2_3_cons_prod_lock_2"}
    %B_L3L2_3_cons_cons_lock_2 = aie.lock(%mem_tile_3_1, 5) {init = 0 : i32, sym_name = "B_L3L2_3_cons_cons_lock_2"}
    %B_L3L2_3_cons_prod_lock_3 = aie.lock(%mem_tile_3_1, 6) {init = 2 : i32, sym_name = "B_L3L2_3_cons_prod_lock_3"}
    %B_L3L2_3_cons_cons_lock_3 = aie.lock(%mem_tile_3_1, 7) {init = 0 : i32, sym_name = "B_L3L2_3_cons_cons_lock_3"}
    %B_L3L2_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "B_L3L2_3_prod_lock_0"}
    %B_L3L2_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "B_L3L2_3_cons_lock_0"}
    %C_L2L3_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "C_L2L3_2_cons_prod_lock_0"}
    %C_L2L3_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "C_L2L3_2_cons_cons_lock_0"}
    %C_L2L3_2_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "C_L2L3_2_buff_0"} : memref<16384xui8> 
    %C_L2L3_2_buff_1 = aie.buffer(%mem_tile_2_1) {address = 16384 : i32, sym_name = "C_L2L3_2_buff_1"} : memref<16384xui8> 
    %C_L2L3_2_prod_lock_0 = aie.lock(%mem_tile_2_1, 8) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_0"}
    %C_L2L3_2_cons_lock_0 = aie.lock(%mem_tile_2_1, 9) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_0"}
    %C_L2L3_2_prod_lock_1 = aie.lock(%mem_tile_2_1, 10) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_1"}
    %C_L2L3_2_cons_lock_1 = aie.lock(%mem_tile_2_1, 11) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_1"}
    %C_L2L3_2_prod_lock_2 = aie.lock(%mem_tile_2_1, 12) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_2"}
    %C_L2L3_2_cons_lock_2 = aie.lock(%mem_tile_2_1, 13) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_2"}
    %C_L2L3_2_prod_lock_3 = aie.lock(%mem_tile_2_1, 14) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_3"}
    %C_L2L3_2_cons_lock_3 = aie.lock(%mem_tile_2_1, 15) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_3"}
    %C_L1L2_2_3_buff_0 = aie.buffer(%tile_2_5) {address = 3840 : i32, sym_name = "C_L1L2_2_3_buff_0"} : memref<4096xui8> 
    %C_L1L2_2_3_prod_lock_0 = aie.lock(%tile_2_5, 5) {init = 1 : i32, sym_name = "C_L1L2_2_3_prod_lock_0"}
    %C_L1L2_2_3_cons_lock_0 = aie.lock(%tile_2_5, 6) {init = 0 : i32, sym_name = "C_L1L2_2_3_cons_lock_0"}
    %C_L1L2_2_2_buff_0 = aie.buffer(%tile_2_4) {address = 3840 : i32, sym_name = "C_L1L2_2_2_buff_0"} : memref<4096xui8> 
    %C_L1L2_2_2_prod_lock_0 = aie.lock(%tile_2_4, 5) {init = 1 : i32, sym_name = "C_L1L2_2_2_prod_lock_0"}
    %C_L1L2_2_2_cons_lock_0 = aie.lock(%tile_2_4, 6) {init = 0 : i32, sym_name = "C_L1L2_2_2_cons_lock_0"}
    %C_L1L2_2_1_buff_0 = aie.buffer(%tile_2_3) {address = 3840 : i32, sym_name = "C_L1L2_2_1_buff_0"} : memref<4096xui8> 
    %C_L1L2_2_1_prod_lock_0 = aie.lock(%tile_2_3, 5) {init = 1 : i32, sym_name = "C_L1L2_2_1_prod_lock_0"}
    %C_L1L2_2_1_cons_lock_0 = aie.lock(%tile_2_3, 6) {init = 0 : i32, sym_name = "C_L1L2_2_1_cons_lock_0"}
    %C_L1L2_2_0_buff_0 = aie.buffer(%tile_2_2) {address = 3840 : i32, sym_name = "C_L1L2_2_0_buff_0"} : memref<4096xui8> 
    %C_L1L2_2_0_prod_lock_0 = aie.lock(%tile_2_2, 5) {init = 1 : i32, sym_name = "C_L1L2_2_0_prod_lock_0"}
    %C_L1L2_2_0_cons_lock_0 = aie.lock(%tile_2_2, 6) {init = 0 : i32, sym_name = "C_L1L2_2_0_cons_lock_0"}
    %B_L2L1_2_3_cons_buff_0 = aie.buffer(%tile_2_5) {address = 7936 : i32, sym_name = "B_L2L1_2_3_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_2_3_cons_buff_1 = aie.buffer(%tile_2_5) {address = 9472 : i32, sym_name = "B_L2L1_2_3_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_2_3_cons_prod_lock_0 = aie.lock(%tile_2_5, 3) {init = 2 : i32, sym_name = "B_L2L1_2_3_cons_prod_lock_0"}
    %B_L2L1_2_3_cons_cons_lock_0 = aie.lock(%tile_2_5, 4) {init = 0 : i32, sym_name = "B_L2L1_2_3_cons_cons_lock_0"}
    %B_L2L1_2_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 7936 : i32, sym_name = "B_L2L1_2_2_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_2_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 9472 : i32, sym_name = "B_L2L1_2_2_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_2_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 3) {init = 2 : i32, sym_name = "B_L2L1_2_2_cons_prod_lock_0"}
    %B_L2L1_2_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 4) {init = 0 : i32, sym_name = "B_L2L1_2_2_cons_cons_lock_0"}
    %B_L2L1_2_1_cons_buff_0 = aie.buffer(%tile_2_3) {address = 7936 : i32, sym_name = "B_L2L1_2_1_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_2_1_cons_buff_1 = aie.buffer(%tile_2_3) {address = 9472 : i32, sym_name = "B_L2L1_2_1_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_2_1_cons_prod_lock_0 = aie.lock(%tile_2_3, 3) {init = 2 : i32, sym_name = "B_L2L1_2_1_cons_prod_lock_0"}
    %B_L2L1_2_1_cons_cons_lock_0 = aie.lock(%tile_2_3, 4) {init = 0 : i32, sym_name = "B_L2L1_2_1_cons_cons_lock_0"}
    %B_L2L1_2_0_cons_buff_0 = aie.buffer(%tile_2_2) {address = 7936 : i32, sym_name = "B_L2L1_2_0_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_2_0_cons_buff_1 = aie.buffer(%tile_2_2) {address = 9472 : i32, sym_name = "B_L2L1_2_0_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_2_0_cons_prod_lock_0 = aie.lock(%tile_2_2, 3) {init = 2 : i32, sym_name = "B_L2L1_2_0_cons_prod_lock_0"}
    %B_L2L1_2_0_cons_cons_lock_0 = aie.lock(%tile_2_2, 4) {init = 0 : i32, sym_name = "B_L2L1_2_0_cons_cons_lock_0"}
    %B_L3L2_2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 32768 : i32, sym_name = "B_L3L2_2_cons_buff_0"} : memref<6144xui8> 
    %B_L3L2_2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 38912 : i32, sym_name = "B_L3L2_2_cons_buff_1"} : memref<6144xui8> 
    %B_L3L2_2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "B_L3L2_2_cons_prod_lock_0"}
    %B_L3L2_2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "B_L3L2_2_cons_cons_lock_0"}
    %B_L3L2_2_cons_prod_lock_1 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "B_L3L2_2_cons_prod_lock_1"}
    %B_L3L2_2_cons_cons_lock_1 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "B_L3L2_2_cons_cons_lock_1"}
    %B_L3L2_2_cons_prod_lock_2 = aie.lock(%mem_tile_2_1, 4) {init = 2 : i32, sym_name = "B_L3L2_2_cons_prod_lock_2"}
    %B_L3L2_2_cons_cons_lock_2 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "B_L3L2_2_cons_cons_lock_2"}
    %B_L3L2_2_cons_prod_lock_3 = aie.lock(%mem_tile_2_1, 6) {init = 2 : i32, sym_name = "B_L3L2_2_cons_prod_lock_3"}
    %B_L3L2_2_cons_cons_lock_3 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "B_L3L2_2_cons_cons_lock_3"}
    %B_L3L2_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "B_L3L2_2_prod_lock_0"}
    %B_L3L2_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "B_L3L2_2_cons_lock_0"}
    %C_L2L3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "C_L2L3_1_cons_prod_lock_0"}
    %C_L2L3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "C_L2L3_1_cons_cons_lock_0"}
    %C_L2L3_1_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "C_L2L3_1_buff_0"} : memref<16384xui8> 
    %C_L2L3_1_buff_1 = aie.buffer(%mem_tile_1_1) {address = 16384 : i32, sym_name = "C_L2L3_1_buff_1"} : memref<16384xui8> 
    %C_L2L3_1_prod_lock_0 = aie.lock(%mem_tile_1_1, 8) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_0"}
    %C_L2L3_1_cons_lock_0 = aie.lock(%mem_tile_1_1, 9) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_0"}
    %C_L2L3_1_prod_lock_1 = aie.lock(%mem_tile_1_1, 10) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_1"}
    %C_L2L3_1_cons_lock_1 = aie.lock(%mem_tile_1_1, 11) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_1"}
    %C_L2L3_1_prod_lock_2 = aie.lock(%mem_tile_1_1, 12) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_2"}
    %C_L2L3_1_cons_lock_2 = aie.lock(%mem_tile_1_1, 13) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_2"}
    %C_L2L3_1_prod_lock_3 = aie.lock(%mem_tile_1_1, 14) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_3"}
    %C_L2L3_1_cons_lock_3 = aie.lock(%mem_tile_1_1, 15) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_3"}
    %C_L1L2_1_3_buff_0 = aie.buffer(%tile_1_5) {address = 3840 : i32, sym_name = "C_L1L2_1_3_buff_0"} : memref<4096xui8> 
    %C_L1L2_1_3_prod_lock_0 = aie.lock(%tile_1_5, 5) {init = 1 : i32, sym_name = "C_L1L2_1_3_prod_lock_0"}
    %C_L1L2_1_3_cons_lock_0 = aie.lock(%tile_1_5, 6) {init = 0 : i32, sym_name = "C_L1L2_1_3_cons_lock_0"}
    %C_L1L2_1_2_buff_0 = aie.buffer(%tile_1_4) {address = 3840 : i32, sym_name = "C_L1L2_1_2_buff_0"} : memref<4096xui8> 
    %C_L1L2_1_2_prod_lock_0 = aie.lock(%tile_1_4, 5) {init = 1 : i32, sym_name = "C_L1L2_1_2_prod_lock_0"}
    %C_L1L2_1_2_cons_lock_0 = aie.lock(%tile_1_4, 6) {init = 0 : i32, sym_name = "C_L1L2_1_2_cons_lock_0"}
    %C_L1L2_1_1_buff_0 = aie.buffer(%tile_1_3) {address = 3840 : i32, sym_name = "C_L1L2_1_1_buff_0"} : memref<4096xui8> 
    %C_L1L2_1_1_prod_lock_0 = aie.lock(%tile_1_3, 5) {init = 1 : i32, sym_name = "C_L1L2_1_1_prod_lock_0"}
    %C_L1L2_1_1_cons_lock_0 = aie.lock(%tile_1_3, 6) {init = 0 : i32, sym_name = "C_L1L2_1_1_cons_lock_0"}
    %C_L1L2_1_0_buff_0 = aie.buffer(%tile_1_2) {address = 3840 : i32, sym_name = "C_L1L2_1_0_buff_0"} : memref<4096xui8> 
    %C_L1L2_1_0_prod_lock_0 = aie.lock(%tile_1_2, 5) {init = 1 : i32, sym_name = "C_L1L2_1_0_prod_lock_0"}
    %C_L1L2_1_0_cons_lock_0 = aie.lock(%tile_1_2, 6) {init = 0 : i32, sym_name = "C_L1L2_1_0_cons_lock_0"}
    %B_L2L1_1_3_cons_buff_0 = aie.buffer(%tile_1_5) {address = 7936 : i32, sym_name = "B_L2L1_1_3_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_1_3_cons_buff_1 = aie.buffer(%tile_1_5) {address = 9472 : i32, sym_name = "B_L2L1_1_3_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_1_3_cons_prod_lock_0 = aie.lock(%tile_1_5, 3) {init = 2 : i32, sym_name = "B_L2L1_1_3_cons_prod_lock_0"}
    %B_L2L1_1_3_cons_cons_lock_0 = aie.lock(%tile_1_5, 4) {init = 0 : i32, sym_name = "B_L2L1_1_3_cons_cons_lock_0"}
    %B_L2L1_1_2_cons_buff_0 = aie.buffer(%tile_1_4) {address = 7936 : i32, sym_name = "B_L2L1_1_2_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_1_2_cons_buff_1 = aie.buffer(%tile_1_4) {address = 9472 : i32, sym_name = "B_L2L1_1_2_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_1_2_cons_prod_lock_0 = aie.lock(%tile_1_4, 3) {init = 2 : i32, sym_name = "B_L2L1_1_2_cons_prod_lock_0"}
    %B_L2L1_1_2_cons_cons_lock_0 = aie.lock(%tile_1_4, 4) {init = 0 : i32, sym_name = "B_L2L1_1_2_cons_cons_lock_0"}
    %B_L2L1_1_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 7936 : i32, sym_name = "B_L2L1_1_1_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_1_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 9472 : i32, sym_name = "B_L2L1_1_1_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_1_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 3) {init = 2 : i32, sym_name = "B_L2L1_1_1_cons_prod_lock_0"}
    %B_L2L1_1_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 4) {init = 0 : i32, sym_name = "B_L2L1_1_1_cons_cons_lock_0"}
    %B_L2L1_1_0_cons_buff_0 = aie.buffer(%tile_1_2) {address = 7936 : i32, sym_name = "B_L2L1_1_0_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_1_0_cons_buff_1 = aie.buffer(%tile_1_2) {address = 9472 : i32, sym_name = "B_L2L1_1_0_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_1_0_cons_prod_lock_0 = aie.lock(%tile_1_2, 3) {init = 2 : i32, sym_name = "B_L2L1_1_0_cons_prod_lock_0"}
    %B_L2L1_1_0_cons_cons_lock_0 = aie.lock(%tile_1_2, 4) {init = 0 : i32, sym_name = "B_L2L1_1_0_cons_cons_lock_0"}
    %B_L3L2_1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 32768 : i32, sym_name = "B_L3L2_1_cons_buff_0"} : memref<6144xui8> 
    %B_L3L2_1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 38912 : i32, sym_name = "B_L3L2_1_cons_buff_1"} : memref<6144xui8> 
    %B_L3L2_1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "B_L3L2_1_cons_prod_lock_0"}
    %B_L3L2_1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "B_L3L2_1_cons_cons_lock_0"}
    %B_L3L2_1_cons_prod_lock_1 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "B_L3L2_1_cons_prod_lock_1"}
    %B_L3L2_1_cons_cons_lock_1 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "B_L3L2_1_cons_cons_lock_1"}
    %B_L3L2_1_cons_prod_lock_2 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "B_L3L2_1_cons_prod_lock_2"}
    %B_L3L2_1_cons_cons_lock_2 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "B_L3L2_1_cons_cons_lock_2"}
    %B_L3L2_1_cons_prod_lock_3 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "B_L3L2_1_cons_prod_lock_3"}
    %B_L3L2_1_cons_cons_lock_3 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "B_L3L2_1_cons_cons_lock_3"}
    %B_L3L2_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "B_L3L2_1_prod_lock_0"}
    %B_L3L2_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "B_L3L2_1_cons_lock_0"}
    %C_L2L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L2L3_0_cons_prod_lock_0"}
    %C_L2L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L2L3_0_cons_cons_lock_0"}
    %C_L2L3_0_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "C_L2L3_0_buff_0"} : memref<16384xui8> 
    %C_L2L3_0_buff_1 = aie.buffer(%mem_tile_0_1) {address = 16384 : i32, sym_name = "C_L2L3_0_buff_1"} : memref<16384xui8> 
    %C_L2L3_0_prod_lock_0 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_0"}
    %C_L2L3_0_cons_lock_0 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_0"}
    %C_L2L3_0_prod_lock_1 = aie.lock(%mem_tile_0_1, 12) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_1"}
    %C_L2L3_0_cons_lock_1 = aie.lock(%mem_tile_0_1, 13) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_1"}
    %C_L2L3_0_prod_lock_2 = aie.lock(%mem_tile_0_1, 14) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_2"}
    %C_L2L3_0_cons_lock_2 = aie.lock(%mem_tile_0_1, 15) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_2"}
    %C_L2L3_0_prod_lock_3 = aie.lock(%mem_tile_0_1, 16) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_3"}
    %C_L2L3_0_cons_lock_3 = aie.lock(%mem_tile_0_1, 17) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_3"}
    %C_L1L2_0_3_buff_0 = aie.buffer(%tile_0_5) {address = 3840 : i32, sym_name = "C_L1L2_0_3_buff_0"} : memref<4096xui8> 
    %C_L1L2_0_3_prod_lock_0 = aie.lock(%tile_0_5, 5) {init = 1 : i32, sym_name = "C_L1L2_0_3_prod_lock_0"}
    %C_L1L2_0_3_cons_lock_0 = aie.lock(%tile_0_5, 6) {init = 0 : i32, sym_name = "C_L1L2_0_3_cons_lock_0"}
    %C_L1L2_0_2_buff_0 = aie.buffer(%tile_0_4) {address = 3840 : i32, sym_name = "C_L1L2_0_2_buff_0"} : memref<4096xui8> 
    %C_L1L2_0_2_prod_lock_0 = aie.lock(%tile_0_4, 5) {init = 1 : i32, sym_name = "C_L1L2_0_2_prod_lock_0"}
    %C_L1L2_0_2_cons_lock_0 = aie.lock(%tile_0_4, 6) {init = 0 : i32, sym_name = "C_L1L2_0_2_cons_lock_0"}
    %C_L1L2_0_1_buff_0 = aie.buffer(%tile_0_3) {address = 3840 : i32, sym_name = "C_L1L2_0_1_buff_0"} : memref<4096xui8> 
    %C_L1L2_0_1_prod_lock_0 = aie.lock(%tile_0_3, 5) {init = 1 : i32, sym_name = "C_L1L2_0_1_prod_lock_0"}
    %C_L1L2_0_1_cons_lock_0 = aie.lock(%tile_0_3, 6) {init = 0 : i32, sym_name = "C_L1L2_0_1_cons_lock_0"}
    %C_L1L2_0_0_buff_0 = aie.buffer(%tile_0_2) {address = 3840 : i32, sym_name = "C_L1L2_0_0_buff_0"} : memref<4096xui8> 
    %C_L1L2_0_0_prod_lock_0 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "C_L1L2_0_0_prod_lock_0"}
    %C_L1L2_0_0_cons_lock_0 = aie.lock(%tile_0_2, 6) {init = 0 : i32, sym_name = "C_L1L2_0_0_cons_lock_0"}
    %B_L2L1_0_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 7936 : i32, sym_name = "B_L2L1_0_3_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_0_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 9472 : i32, sym_name = "B_L2L1_0_3_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_0_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 3) {init = 2 : i32, sym_name = "B_L2L1_0_3_cons_prod_lock_0"}
    %B_L2L1_0_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 4) {init = 0 : i32, sym_name = "B_L2L1_0_3_cons_cons_lock_0"}
    %B_L2L1_0_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 7936 : i32, sym_name = "B_L2L1_0_2_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_0_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 9472 : i32, sym_name = "B_L2L1_0_2_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_0_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 3) {init = 2 : i32, sym_name = "B_L2L1_0_2_cons_prod_lock_0"}
    %B_L2L1_0_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 4) {init = 0 : i32, sym_name = "B_L2L1_0_2_cons_cons_lock_0"}
    %B_L2L1_0_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 7936 : i32, sym_name = "B_L2L1_0_1_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_0_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 9472 : i32, sym_name = "B_L2L1_0_1_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_0_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 3) {init = 2 : i32, sym_name = "B_L2L1_0_1_cons_prod_lock_0"}
    %B_L2L1_0_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 4) {init = 0 : i32, sym_name = "B_L2L1_0_1_cons_cons_lock_0"}
    %B_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 7936 : i32, sym_name = "B_L2L1_0_0_cons_buff_0"} : memref<1536xui8> 
    %B_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 9472 : i32, sym_name = "B_L2L1_0_0_cons_buff_1"} : memref<1536xui8> 
    %B_L2L1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 3) {init = 2 : i32, sym_name = "B_L2L1_0_0_cons_prod_lock_0"}
    %B_L2L1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "B_L2L1_0_0_cons_cons_lock_0"}
    %B_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 32768 : i32, sym_name = "B_L3L2_0_cons_buff_0"} : memref<6144xui8> 
    %B_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 38912 : i32, sym_name = "B_L3L2_0_cons_buff_1"} : memref<6144xui8> 
    %B_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "B_L3L2_0_cons_prod_lock_0"}
    %B_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_cons_lock_0"}
    %B_L3L2_0_cons_prod_lock_1 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "B_L3L2_0_cons_prod_lock_1"}
    %B_L3L2_0_cons_cons_lock_1 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "B_L3L2_0_cons_cons_lock_1"}
    %B_L3L2_0_cons_prod_lock_2 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "B_L3L2_0_cons_prod_lock_2"}
    %B_L3L2_0_cons_cons_lock_2 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "B_L3L2_0_cons_cons_lock_2"}
    %B_L3L2_0_cons_prod_lock_3 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "B_L3L2_0_cons_prod_lock_3"}
    %B_L3L2_0_cons_cons_lock_3 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "B_L3L2_0_cons_cons_lock_3"}
    %B_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L2_0_prod_lock_0"}
    %B_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_lock_0"}
    %A_L2L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 11008 : i32, sym_name = "A_L2L1_0_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 12160 : i32, sym_name = "A_L2L1_0_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 1) {init = 2 : i32, sym_name = "A_L2L1_0_cons_prod_lock_0"}
    %A_L2L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "A_L2L1_0_cons_cons_lock_0"}
    %A_L2L1_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 11008 : i32, sym_name = "A_L2L1_1_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 12160 : i32, sym_name = "A_L2L1_1_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 1) {init = 2 : i32, sym_name = "A_L2L1_1_cons_prod_lock_0"}
    %A_L2L1_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 2) {init = 0 : i32, sym_name = "A_L2L1_1_cons_cons_lock_0"}
    %A_L2L1_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 11008 : i32, sym_name = "A_L2L1_2_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 12160 : i32, sym_name = "A_L2L1_2_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 1) {init = 2 : i32, sym_name = "A_L2L1_2_cons_prod_lock_0"}
    %A_L2L1_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 2) {init = 0 : i32, sym_name = "A_L2L1_2_cons_cons_lock_0"}
    %A_L2L1_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 11008 : i32, sym_name = "A_L2L1_3_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 12160 : i32, sym_name = "A_L2L1_3_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 1) {init = 2 : i32, sym_name = "A_L2L1_3_cons_prod_lock_0"}
    %A_L2L1_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 2) {init = 0 : i32, sym_name = "A_L2L1_3_cons_cons_lock_0"}
    %A_L2L1_4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 11008 : i32, sym_name = "A_L2L1_4_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 12160 : i32, sym_name = "A_L2L1_4_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_4_cons_prod_lock_0 = aie.lock(%tile_4_2, 1) {init = 2 : i32, sym_name = "A_L2L1_4_cons_prod_lock_0"}
    %A_L2L1_4_cons_cons_lock_0 = aie.lock(%tile_4_2, 2) {init = 0 : i32, sym_name = "A_L2L1_4_cons_cons_lock_0"}
    %A_L2L1_5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 11008 : i32, sym_name = "A_L2L1_5_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 12160 : i32, sym_name = "A_L2L1_5_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_5_cons_prod_lock_0 = aie.lock(%tile_5_2, 1) {init = 2 : i32, sym_name = "A_L2L1_5_cons_prod_lock_0"}
    %A_L2L1_5_cons_cons_lock_0 = aie.lock(%tile_5_2, 2) {init = 0 : i32, sym_name = "A_L2L1_5_cons_cons_lock_0"}
    %A_L2L1_6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 11008 : i32, sym_name = "A_L2L1_6_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 12160 : i32, sym_name = "A_L2L1_6_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_6_cons_prod_lock_0 = aie.lock(%tile_6_2, 1) {init = 2 : i32, sym_name = "A_L2L1_6_cons_prod_lock_0"}
    %A_L2L1_6_cons_cons_lock_0 = aie.lock(%tile_6_2, 2) {init = 0 : i32, sym_name = "A_L2L1_6_cons_cons_lock_0"}
    %A_L2L1_7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 11008 : i32, sym_name = "A_L2L1_7_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 12160 : i32, sym_name = "A_L2L1_7_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_7_cons_prod_lock_0 = aie.lock(%tile_7_2, 1) {init = 2 : i32, sym_name = "A_L2L1_7_cons_prod_lock_0"}
    %A_L2L1_7_cons_cons_lock_0 = aie.lock(%tile_7_2, 2) {init = 0 : i32, sym_name = "A_L2L1_7_cons_cons_lock_0"}
    %A_L2L1_8_cons_buff_0 = aie.buffer(%tile_0_3) {address = 11008 : i32, sym_name = "A_L2L1_8_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_8_cons_buff_1 = aie.buffer(%tile_0_3) {address = 12160 : i32, sym_name = "A_L2L1_8_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_8_cons_prod_lock_0 = aie.lock(%tile_0_3, 1) {init = 2 : i32, sym_name = "A_L2L1_8_cons_prod_lock_0"}
    %A_L2L1_8_cons_cons_lock_0 = aie.lock(%tile_0_3, 2) {init = 0 : i32, sym_name = "A_L2L1_8_cons_cons_lock_0"}
    %A_L2L1_9_cons_buff_0 = aie.buffer(%tile_1_3) {address = 11008 : i32, sym_name = "A_L2L1_9_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_9_cons_buff_1 = aie.buffer(%tile_1_3) {address = 12160 : i32, sym_name = "A_L2L1_9_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_9_cons_prod_lock_0 = aie.lock(%tile_1_3, 1) {init = 2 : i32, sym_name = "A_L2L1_9_cons_prod_lock_0"}
    %A_L2L1_9_cons_cons_lock_0 = aie.lock(%tile_1_3, 2) {init = 0 : i32, sym_name = "A_L2L1_9_cons_cons_lock_0"}
    %A_L2L1_10_cons_buff_0 = aie.buffer(%tile_2_3) {address = 11008 : i32, sym_name = "A_L2L1_10_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_10_cons_buff_1 = aie.buffer(%tile_2_3) {address = 12160 : i32, sym_name = "A_L2L1_10_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_10_cons_prod_lock_0 = aie.lock(%tile_2_3, 1) {init = 2 : i32, sym_name = "A_L2L1_10_cons_prod_lock_0"}
    %A_L2L1_10_cons_cons_lock_0 = aie.lock(%tile_2_3, 2) {init = 0 : i32, sym_name = "A_L2L1_10_cons_cons_lock_0"}
    %A_L2L1_11_cons_buff_0 = aie.buffer(%tile_3_3) {address = 11008 : i32, sym_name = "A_L2L1_11_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_11_cons_buff_1 = aie.buffer(%tile_3_3) {address = 12160 : i32, sym_name = "A_L2L1_11_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_11_cons_prod_lock_0 = aie.lock(%tile_3_3, 1) {init = 2 : i32, sym_name = "A_L2L1_11_cons_prod_lock_0"}
    %A_L2L1_11_cons_cons_lock_0 = aie.lock(%tile_3_3, 2) {init = 0 : i32, sym_name = "A_L2L1_11_cons_cons_lock_0"}
    %A_L2L1_12_cons_buff_0 = aie.buffer(%tile_4_3) {address = 11008 : i32, sym_name = "A_L2L1_12_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_12_cons_buff_1 = aie.buffer(%tile_4_3) {address = 12160 : i32, sym_name = "A_L2L1_12_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_12_cons_prod_lock_0 = aie.lock(%tile_4_3, 1) {init = 2 : i32, sym_name = "A_L2L1_12_cons_prod_lock_0"}
    %A_L2L1_12_cons_cons_lock_0 = aie.lock(%tile_4_3, 2) {init = 0 : i32, sym_name = "A_L2L1_12_cons_cons_lock_0"}
    %A_L2L1_13_cons_buff_0 = aie.buffer(%tile_5_3) {address = 11008 : i32, sym_name = "A_L2L1_13_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_13_cons_buff_1 = aie.buffer(%tile_5_3) {address = 12160 : i32, sym_name = "A_L2L1_13_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_13_cons_prod_lock_0 = aie.lock(%tile_5_3, 1) {init = 2 : i32, sym_name = "A_L2L1_13_cons_prod_lock_0"}
    %A_L2L1_13_cons_cons_lock_0 = aie.lock(%tile_5_3, 2) {init = 0 : i32, sym_name = "A_L2L1_13_cons_cons_lock_0"}
    %A_L2L1_14_cons_buff_0 = aie.buffer(%tile_6_3) {address = 11008 : i32, sym_name = "A_L2L1_14_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_14_cons_buff_1 = aie.buffer(%tile_6_3) {address = 12160 : i32, sym_name = "A_L2L1_14_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_14_cons_prod_lock_0 = aie.lock(%tile_6_3, 1) {init = 2 : i32, sym_name = "A_L2L1_14_cons_prod_lock_0"}
    %A_L2L1_14_cons_cons_lock_0 = aie.lock(%tile_6_3, 2) {init = 0 : i32, sym_name = "A_L2L1_14_cons_cons_lock_0"}
    %A_L2L1_15_cons_buff_0 = aie.buffer(%tile_7_3) {address = 11008 : i32, sym_name = "A_L2L1_15_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_15_cons_buff_1 = aie.buffer(%tile_7_3) {address = 12160 : i32, sym_name = "A_L2L1_15_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_15_cons_prod_lock_0 = aie.lock(%tile_7_3, 1) {init = 2 : i32, sym_name = "A_L2L1_15_cons_prod_lock_0"}
    %A_L2L1_15_cons_cons_lock_0 = aie.lock(%tile_7_3, 2) {init = 0 : i32, sym_name = "A_L2L1_15_cons_cons_lock_0"}
    %A_L2L1_16_cons_buff_0 = aie.buffer(%tile_0_4) {address = 11008 : i32, sym_name = "A_L2L1_16_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_16_cons_buff_1 = aie.buffer(%tile_0_4) {address = 12160 : i32, sym_name = "A_L2L1_16_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_16_cons_prod_lock_0 = aie.lock(%tile_0_4, 1) {init = 2 : i32, sym_name = "A_L2L1_16_cons_prod_lock_0"}
    %A_L2L1_16_cons_cons_lock_0 = aie.lock(%tile_0_4, 2) {init = 0 : i32, sym_name = "A_L2L1_16_cons_cons_lock_0"}
    %A_L2L1_17_cons_buff_0 = aie.buffer(%tile_1_4) {address = 11008 : i32, sym_name = "A_L2L1_17_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_17_cons_buff_1 = aie.buffer(%tile_1_4) {address = 12160 : i32, sym_name = "A_L2L1_17_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_17_cons_prod_lock_0 = aie.lock(%tile_1_4, 1) {init = 2 : i32, sym_name = "A_L2L1_17_cons_prod_lock_0"}
    %A_L2L1_17_cons_cons_lock_0 = aie.lock(%tile_1_4, 2) {init = 0 : i32, sym_name = "A_L2L1_17_cons_cons_lock_0"}
    %A_L2L1_18_cons_buff_0 = aie.buffer(%tile_2_4) {address = 11008 : i32, sym_name = "A_L2L1_18_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_18_cons_buff_1 = aie.buffer(%tile_2_4) {address = 12160 : i32, sym_name = "A_L2L1_18_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_18_cons_prod_lock_0 = aie.lock(%tile_2_4, 1) {init = 2 : i32, sym_name = "A_L2L1_18_cons_prod_lock_0"}
    %A_L2L1_18_cons_cons_lock_0 = aie.lock(%tile_2_4, 2) {init = 0 : i32, sym_name = "A_L2L1_18_cons_cons_lock_0"}
    %A_L2L1_19_cons_buff_0 = aie.buffer(%tile_3_4) {address = 11008 : i32, sym_name = "A_L2L1_19_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_19_cons_buff_1 = aie.buffer(%tile_3_4) {address = 12160 : i32, sym_name = "A_L2L1_19_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_19_cons_prod_lock_0 = aie.lock(%tile_3_4, 1) {init = 2 : i32, sym_name = "A_L2L1_19_cons_prod_lock_0"}
    %A_L2L1_19_cons_cons_lock_0 = aie.lock(%tile_3_4, 2) {init = 0 : i32, sym_name = "A_L2L1_19_cons_cons_lock_0"}
    %A_L2L1_20_cons_buff_0 = aie.buffer(%tile_4_4) {address = 11008 : i32, sym_name = "A_L2L1_20_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_20_cons_buff_1 = aie.buffer(%tile_4_4) {address = 12160 : i32, sym_name = "A_L2L1_20_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_20_cons_prod_lock_0 = aie.lock(%tile_4_4, 1) {init = 2 : i32, sym_name = "A_L2L1_20_cons_prod_lock_0"}
    %A_L2L1_20_cons_cons_lock_0 = aie.lock(%tile_4_4, 2) {init = 0 : i32, sym_name = "A_L2L1_20_cons_cons_lock_0"}
    %A_L2L1_21_cons_buff_0 = aie.buffer(%tile_5_4) {address = 11008 : i32, sym_name = "A_L2L1_21_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_21_cons_buff_1 = aie.buffer(%tile_5_4) {address = 12160 : i32, sym_name = "A_L2L1_21_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_21_cons_prod_lock_0 = aie.lock(%tile_5_4, 1) {init = 2 : i32, sym_name = "A_L2L1_21_cons_prod_lock_0"}
    %A_L2L1_21_cons_cons_lock_0 = aie.lock(%tile_5_4, 2) {init = 0 : i32, sym_name = "A_L2L1_21_cons_cons_lock_0"}
    %A_L2L1_22_cons_buff_0 = aie.buffer(%tile_6_4) {address = 11008 : i32, sym_name = "A_L2L1_22_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_22_cons_buff_1 = aie.buffer(%tile_6_4) {address = 12160 : i32, sym_name = "A_L2L1_22_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_22_cons_prod_lock_0 = aie.lock(%tile_6_4, 1) {init = 2 : i32, sym_name = "A_L2L1_22_cons_prod_lock_0"}
    %A_L2L1_22_cons_cons_lock_0 = aie.lock(%tile_6_4, 2) {init = 0 : i32, sym_name = "A_L2L1_22_cons_cons_lock_0"}
    %A_L2L1_23_cons_buff_0 = aie.buffer(%tile_7_4) {address = 11008 : i32, sym_name = "A_L2L1_23_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_23_cons_buff_1 = aie.buffer(%tile_7_4) {address = 12160 : i32, sym_name = "A_L2L1_23_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_23_cons_prod_lock_0 = aie.lock(%tile_7_4, 1) {init = 2 : i32, sym_name = "A_L2L1_23_cons_prod_lock_0"}
    %A_L2L1_23_cons_cons_lock_0 = aie.lock(%tile_7_4, 2) {init = 0 : i32, sym_name = "A_L2L1_23_cons_cons_lock_0"}
    %A_L2L1_24_cons_buff_0 = aie.buffer(%tile_0_5) {address = 11008 : i32, sym_name = "A_L2L1_24_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_24_cons_buff_1 = aie.buffer(%tile_0_5) {address = 12160 : i32, sym_name = "A_L2L1_24_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_24_cons_prod_lock_0 = aie.lock(%tile_0_5, 1) {init = 2 : i32, sym_name = "A_L2L1_24_cons_prod_lock_0"}
    %A_L2L1_24_cons_cons_lock_0 = aie.lock(%tile_0_5, 2) {init = 0 : i32, sym_name = "A_L2L1_24_cons_cons_lock_0"}
    %A_L2L1_25_cons_buff_0 = aie.buffer(%tile_1_5) {address = 11008 : i32, sym_name = "A_L2L1_25_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_25_cons_buff_1 = aie.buffer(%tile_1_5) {address = 12160 : i32, sym_name = "A_L2L1_25_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_25_cons_prod_lock_0 = aie.lock(%tile_1_5, 1) {init = 2 : i32, sym_name = "A_L2L1_25_cons_prod_lock_0"}
    %A_L2L1_25_cons_cons_lock_0 = aie.lock(%tile_1_5, 2) {init = 0 : i32, sym_name = "A_L2L1_25_cons_cons_lock_0"}
    %A_L2L1_26_cons_buff_0 = aie.buffer(%tile_2_5) {address = 11008 : i32, sym_name = "A_L2L1_26_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_26_cons_buff_1 = aie.buffer(%tile_2_5) {address = 12160 : i32, sym_name = "A_L2L1_26_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_26_cons_prod_lock_0 = aie.lock(%tile_2_5, 1) {init = 2 : i32, sym_name = "A_L2L1_26_cons_prod_lock_0"}
    %A_L2L1_26_cons_cons_lock_0 = aie.lock(%tile_2_5, 2) {init = 0 : i32, sym_name = "A_L2L1_26_cons_cons_lock_0"}
    %A_L2L1_27_cons_buff_0 = aie.buffer(%tile_3_5) {address = 11008 : i32, sym_name = "A_L2L1_27_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_27_cons_buff_1 = aie.buffer(%tile_3_5) {address = 12160 : i32, sym_name = "A_L2L1_27_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_27_cons_prod_lock_0 = aie.lock(%tile_3_5, 1) {init = 2 : i32, sym_name = "A_L2L1_27_cons_prod_lock_0"}
    %A_L2L1_27_cons_cons_lock_0 = aie.lock(%tile_3_5, 2) {init = 0 : i32, sym_name = "A_L2L1_27_cons_cons_lock_0"}
    %A_L2L1_28_cons_buff_0 = aie.buffer(%tile_4_5) {address = 11008 : i32, sym_name = "A_L2L1_28_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_28_cons_buff_1 = aie.buffer(%tile_4_5) {address = 12160 : i32, sym_name = "A_L2L1_28_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_28_cons_prod_lock_0 = aie.lock(%tile_4_5, 1) {init = 2 : i32, sym_name = "A_L2L1_28_cons_prod_lock_0"}
    %A_L2L1_28_cons_cons_lock_0 = aie.lock(%tile_4_5, 2) {init = 0 : i32, sym_name = "A_L2L1_28_cons_cons_lock_0"}
    %A_L2L1_29_cons_buff_0 = aie.buffer(%tile_5_5) {address = 11008 : i32, sym_name = "A_L2L1_29_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_29_cons_buff_1 = aie.buffer(%tile_5_5) {address = 12160 : i32, sym_name = "A_L2L1_29_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_29_cons_prod_lock_0 = aie.lock(%tile_5_5, 1) {init = 2 : i32, sym_name = "A_L2L1_29_cons_prod_lock_0"}
    %A_L2L1_29_cons_cons_lock_0 = aie.lock(%tile_5_5, 2) {init = 0 : i32, sym_name = "A_L2L1_29_cons_cons_lock_0"}
    %A_L2L1_30_cons_buff_0 = aie.buffer(%tile_6_5) {address = 11008 : i32, sym_name = "A_L2L1_30_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_30_cons_buff_1 = aie.buffer(%tile_6_5) {address = 12160 : i32, sym_name = "A_L2L1_30_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_30_cons_prod_lock_0 = aie.lock(%tile_6_5, 1) {init = 2 : i32, sym_name = "A_L2L1_30_cons_prod_lock_0"}
    %A_L2L1_30_cons_cons_lock_0 = aie.lock(%tile_6_5, 2) {init = 0 : i32, sym_name = "A_L2L1_30_cons_cons_lock_0"}
    %A_L2L1_31_cons_buff_0 = aie.buffer(%tile_7_5) {address = 11008 : i32, sym_name = "A_L2L1_31_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_31_cons_buff_1 = aie.buffer(%tile_7_5) {address = 12160 : i32, sym_name = "A_L2L1_31_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L2L1_31_cons_prod_lock_0 = aie.lock(%tile_7_5, 1) {init = 2 : i32, sym_name = "A_L2L1_31_cons_prod_lock_0"}
    %A_L2L1_31_cons_cons_lock_0 = aie.lock(%tile_7_5, 2) {init = 0 : i32, sym_name = "A_L2L1_31_cons_cons_lock_0"}
    %A_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 45056 : i32, sym_name = "A_L3L2_cons_buff_0"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L3L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 46208 : i32, sym_name = "A_L3L2_cons_buff_1"} : memref<16x8x!aiex.bfp<"v8bfp16ebs8">> 
    %A_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "A_L3L2_cons_prod_lock_0"}
    %A_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_cons_lock_0"}
    %A_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_prod_lock_0"}
    %A_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_lock_0"}
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_6_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_5_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_4_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_3_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_2_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_7_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_6_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_5_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_4_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_7_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_6_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_5_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_4_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    %rtp_0_0 = aie.buffer(%tile_0_2) {address = 13312 : i32, sym_name = "rtp_0_0"} : memref<9xi32> 
    %rtp_lock_0_0 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "rtp_lock_0_0"}
    %rtp_0_1 = aie.buffer(%tile_1_2) {address = 13312 : i32, sym_name = "rtp_0_1"} : memref<9xi32> 
    %rtp_lock_0_1 = aie.lock(%tile_1_2, 0) {init = 0 : i32, sym_name = "rtp_lock_0_1"}
    %rtp_0_2 = aie.buffer(%tile_2_2) {address = 13312 : i32, sym_name = "rtp_0_2"} : memref<9xi32> 
    %rtp_lock_0_2 = aie.lock(%tile_2_2, 0) {init = 0 : i32, sym_name = "rtp_lock_0_2"}
    %rtp_0_3 = aie.buffer(%tile_3_2) {address = 13312 : i32, sym_name = "rtp_0_3"} : memref<9xi32> 
    %rtp_lock_0_3 = aie.lock(%tile_3_2, 0) {init = 0 : i32, sym_name = "rtp_lock_0_3"}
    %rtp_0_4 = aie.buffer(%tile_4_2) {address = 13312 : i32, sym_name = "rtp_0_4"} : memref<9xi32> 
    %rtp_lock_0_4 = aie.lock(%tile_4_2, 0) {init = 0 : i32, sym_name = "rtp_lock_0_4"}
    %rtp_0_5 = aie.buffer(%tile_5_2) {address = 13312 : i32, sym_name = "rtp_0_5"} : memref<9xi32> 
    %rtp_lock_0_5 = aie.lock(%tile_5_2, 0) {init = 0 : i32, sym_name = "rtp_lock_0_5"}
    %rtp_0_6 = aie.buffer(%tile_6_2) {address = 13312 : i32, sym_name = "rtp_0_6"} : memref<9xi32> 
    %rtp_lock_0_6 = aie.lock(%tile_6_2, 0) {init = 0 : i32, sym_name = "rtp_lock_0_6"}
    %rtp_0_7 = aie.buffer(%tile_7_2) {address = 13312 : i32, sym_name = "rtp_0_7"} : memref<9xi32> 
    %rtp_lock_0_7 = aie.lock(%tile_7_2, 0) {init = 0 : i32, sym_name = "rtp_lock_0_7"}
    %rtp_1_0 = aie.buffer(%tile_0_3) {address = 13312 : i32, sym_name = "rtp_1_0"} : memref<9xi32> 
    %rtp_lock_1_0 = aie.lock(%tile_0_3, 0) {init = 0 : i32, sym_name = "rtp_lock_1_0"}
    %rtp_1_1 = aie.buffer(%tile_1_3) {address = 13312 : i32, sym_name = "rtp_1_1"} : memref<9xi32> 
    %rtp_lock_1_1 = aie.lock(%tile_1_3, 0) {init = 0 : i32, sym_name = "rtp_lock_1_1"}
    %rtp_1_2 = aie.buffer(%tile_2_3) {address = 13312 : i32, sym_name = "rtp_1_2"} : memref<9xi32> 
    %rtp_lock_1_2 = aie.lock(%tile_2_3, 0) {init = 0 : i32, sym_name = "rtp_lock_1_2"}
    %rtp_1_3 = aie.buffer(%tile_3_3) {address = 13312 : i32, sym_name = "rtp_1_3"} : memref<9xi32> 
    %rtp_lock_1_3 = aie.lock(%tile_3_3, 0) {init = 0 : i32, sym_name = "rtp_lock_1_3"}
    %rtp_1_4 = aie.buffer(%tile_4_3) {address = 13312 : i32, sym_name = "rtp_1_4"} : memref<9xi32> 
    %rtp_lock_1_4 = aie.lock(%tile_4_3, 0) {init = 0 : i32, sym_name = "rtp_lock_1_4"}
    %rtp_1_5 = aie.buffer(%tile_5_3) {address = 13312 : i32, sym_name = "rtp_1_5"} : memref<9xi32> 
    %rtp_lock_1_5 = aie.lock(%tile_5_3, 0) {init = 0 : i32, sym_name = "rtp_lock_1_5"}
    %rtp_1_6 = aie.buffer(%tile_6_3) {address = 13312 : i32, sym_name = "rtp_1_6"} : memref<9xi32> 
    %rtp_lock_1_6 = aie.lock(%tile_6_3, 0) {init = 0 : i32, sym_name = "rtp_lock_1_6"}
    %rtp_1_7 = aie.buffer(%tile_7_3) {address = 13312 : i32, sym_name = "rtp_1_7"} : memref<9xi32> 
    %rtp_lock_1_7 = aie.lock(%tile_7_3, 0) {init = 0 : i32, sym_name = "rtp_lock_1_7"}
    %rtp_2_0 = aie.buffer(%tile_0_4) {address = 13312 : i32, sym_name = "rtp_2_0"} : memref<9xi32> 
    %rtp_lock_2_0 = aie.lock(%tile_0_4, 0) {init = 0 : i32, sym_name = "rtp_lock_2_0"}
    %rtp_2_1 = aie.buffer(%tile_1_4) {address = 13312 : i32, sym_name = "rtp_2_1"} : memref<9xi32> 
    %rtp_lock_2_1 = aie.lock(%tile_1_4, 0) {init = 0 : i32, sym_name = "rtp_lock_2_1"}
    %rtp_2_2 = aie.buffer(%tile_2_4) {address = 13312 : i32, sym_name = "rtp_2_2"} : memref<9xi32> 
    %rtp_lock_2_2 = aie.lock(%tile_2_4, 0) {init = 0 : i32, sym_name = "rtp_lock_2_2"}
    %rtp_2_3 = aie.buffer(%tile_3_4) {address = 13312 : i32, sym_name = "rtp_2_3"} : memref<9xi32> 
    %rtp_lock_2_3 = aie.lock(%tile_3_4, 0) {init = 0 : i32, sym_name = "rtp_lock_2_3"}
    %rtp_2_4 = aie.buffer(%tile_4_4) {address = 13312 : i32, sym_name = "rtp_2_4"} : memref<9xi32> 
    %rtp_lock_2_4 = aie.lock(%tile_4_4, 0) {init = 0 : i32, sym_name = "rtp_lock_2_4"}
    %rtp_2_5 = aie.buffer(%tile_5_4) {address = 13312 : i32, sym_name = "rtp_2_5"} : memref<9xi32> 
    %rtp_lock_2_5 = aie.lock(%tile_5_4, 0) {init = 0 : i32, sym_name = "rtp_lock_2_5"}
    %rtp_2_6 = aie.buffer(%tile_6_4) {address = 13312 : i32, sym_name = "rtp_2_6"} : memref<9xi32> 
    %rtp_lock_2_6 = aie.lock(%tile_6_4, 0) {init = 0 : i32, sym_name = "rtp_lock_2_6"}
    %rtp_2_7 = aie.buffer(%tile_7_4) {address = 13312 : i32, sym_name = "rtp_2_7"} : memref<9xi32> 
    %rtp_lock_2_7 = aie.lock(%tile_7_4, 0) {init = 0 : i32, sym_name = "rtp_lock_2_7"}
    %rtp_3_0 = aie.buffer(%tile_0_5) {address = 13312 : i32, sym_name = "rtp_3_0"} : memref<9xi32> 
    %rtp_lock_3_0 = aie.lock(%tile_0_5, 0) {init = 0 : i32, sym_name = "rtp_lock_3_0"}
    %rtp_3_1 = aie.buffer(%tile_1_5) {address = 13312 : i32, sym_name = "rtp_3_1"} : memref<9xi32> 
    %rtp_lock_3_1 = aie.lock(%tile_1_5, 0) {init = 0 : i32, sym_name = "rtp_lock_3_1"}
    %rtp_3_2 = aie.buffer(%tile_2_5) {address = 13312 : i32, sym_name = "rtp_3_2"} : memref<9xi32> 
    %rtp_lock_3_2 = aie.lock(%tile_2_5, 0) {init = 0 : i32, sym_name = "rtp_lock_3_2"}
    %rtp_3_3 = aie.buffer(%tile_3_5) {address = 13312 : i32, sym_name = "rtp_3_3"} : memref<9xi32> 
    %rtp_lock_3_3 = aie.lock(%tile_3_5, 0) {init = 0 : i32, sym_name = "rtp_lock_3_3"}
    %rtp_3_4 = aie.buffer(%tile_4_5) {address = 13312 : i32, sym_name = "rtp_3_4"} : memref<9xi32> 
    %rtp_lock_3_4 = aie.lock(%tile_4_5, 0) {init = 0 : i32, sym_name = "rtp_lock_3_4"}
    %rtp_3_5 = aie.buffer(%tile_5_5) {address = 13312 : i32, sym_name = "rtp_3_5"} : memref<9xi32> 
    %rtp_lock_3_5 = aie.lock(%tile_5_5, 0) {init = 0 : i32, sym_name = "rtp_lock_3_5"}
    %rtp_3_6 = aie.buffer(%tile_6_5) {address = 13312 : i32, sym_name = "rtp_3_6"} : memref<9xi32> 
    %rtp_lock_3_6 = aie.lock(%tile_6_5, 0) {init = 0 : i32, sym_name = "rtp_lock_3_6"}
    %rtp_3_7 = aie.buffer(%tile_7_5) {address = 13312 : i32, sym_name = "rtp_3_7"} : memref<9xi32> 
    %rtp_lock_3_7 = aie.lock(%tile_7_5, 0) {init = 0 : i32, sym_name = "rtp_lock_3_7"}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 13348 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_0_0, Acquire, 1)
      %2 = memref.load %rtp_0_0[%c0] : memref<9xi32>
      %3 = memref.load %rtp_0_0[%c1] : memref<9xi32>
      %4 = memref.load %rtp_0_0[%c2] : memref<9xi32>
      %5 = memref.load %rtp_0_0[%c3] : memref<9xi32>
      %6 = memref.load %rtp_0_0[%c4] : memref<9xi32>
      %7 = memref.load %rtp_0_0[%c5] : memref<9xi32>
      %8 = memref.load %rtp_0_0[%c6] : memref<9xi32>
      %9 = memref.load %rtp_0_0[%c7] : memref<9xi32>
      %10 = memref.load %rtp_0_0[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_0_0, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_0_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_0_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_0_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_0_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_0_0_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous0[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous0[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous0[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_0_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_0_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_0_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_0_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_0_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_0_0_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous0[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous0[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous0[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_0_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous0[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_0_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c0_i32, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous0[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_0_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_0_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_0_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c0_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous0[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_0_0_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous0[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous1 = aie.buffer(%tile_1_2) {address = 13348 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous1[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_0_1, Acquire, 1)
      %2 = memref.load %rtp_0_1[%c0] : memref<9xi32>
      %3 = memref.load %rtp_0_1[%c1] : memref<9xi32>
      %4 = memref.load %rtp_0_1[%c2] : memref<9xi32>
      %5 = memref.load %rtp_0_1[%c3] : memref<9xi32>
      %6 = memref.load %rtp_0_1[%c4] : memref<9xi32>
      %7 = memref.load %rtp_0_1[%c5] : memref<9xi32>
      %8 = memref.load %rtp_0_1[%c6] : memref<9xi32>
      %9 = memref.load %rtp_0_1[%c7] : memref<9xi32>
      %10 = memref.load %rtp_0_1[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_0_1, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_1_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_0_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_0_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_0_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_1_0_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous1[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous1[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous1[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_1_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_0_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_0_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_0_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_1_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_1_0_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous1[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous1[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous1[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_1_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous1[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_1_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous1[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_1_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c0_i32, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous1[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_0_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_0_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_0_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c1_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous1[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_1_0_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous1[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous2 = aie.buffer(%tile_2_2) {address = 13348 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous2[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_0_2, Acquire, 1)
      %2 = memref.load %rtp_0_2[%c0] : memref<9xi32>
      %3 = memref.load %rtp_0_2[%c1] : memref<9xi32>
      %4 = memref.load %rtp_0_2[%c2] : memref<9xi32>
      %5 = memref.load %rtp_0_2[%c3] : memref<9xi32>
      %6 = memref.load %rtp_0_2[%c4] : memref<9xi32>
      %7 = memref.load %rtp_0_2[%c5] : memref<9xi32>
      %8 = memref.load %rtp_0_2[%c6] : memref<9xi32>
      %9 = memref.load %rtp_0_2[%c7] : memref<9xi32>
      %10 = memref.load %rtp_0_2[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_0_2, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_2_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_0_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_0_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_0_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_2_0_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous2[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous2[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous2[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_2_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_0_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_0_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_0_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_2_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_2_0_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous2[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous2[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous2[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_2_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous2[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_2_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous2[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_2_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c0_i32, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous2[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_0_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_0_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_0_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c2_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous2[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_2_0_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous2[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous3 = aie.buffer(%tile_3_2) {address = 13348 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous3[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_0_3, Acquire, 1)
      %2 = memref.load %rtp_0_3[%c0] : memref<9xi32>
      %3 = memref.load %rtp_0_3[%c1] : memref<9xi32>
      %4 = memref.load %rtp_0_3[%c2] : memref<9xi32>
      %5 = memref.load %rtp_0_3[%c3] : memref<9xi32>
      %6 = memref.load %rtp_0_3[%c4] : memref<9xi32>
      %7 = memref.load %rtp_0_3[%c5] : memref<9xi32>
      %8 = memref.load %rtp_0_3[%c6] : memref<9xi32>
      %9 = memref.load %rtp_0_3[%c7] : memref<9xi32>
      %10 = memref.load %rtp_0_3[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_0_3, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_3_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_0_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_0_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_0_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_3_0_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous3[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous3[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous3[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_3_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_0_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_0_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_0_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_3_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_3_0_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous3[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous3[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous3[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_3_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous3[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_3_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous3[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_3_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c0_i32, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous3[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_0_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_0_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_0_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c3_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous3[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_3_0_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous3[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous4 = aie.buffer(%tile_4_2) {address = 13348 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
    %core_4_2 = aie.core(%tile_4_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c4_i32 = arith.constant 4 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous4[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_0_4, Acquire, 1)
      %2 = memref.load %rtp_0_4[%c0] : memref<9xi32>
      %3 = memref.load %rtp_0_4[%c1] : memref<9xi32>
      %4 = memref.load %rtp_0_4[%c2] : memref<9xi32>
      %5 = memref.load %rtp_0_4[%c3] : memref<9xi32>
      %6 = memref.load %rtp_0_4[%c4] : memref<9xi32>
      %7 = memref.load %rtp_0_4[%c5] : memref<9xi32>
      %8 = memref.load %rtp_0_4[%c6] : memref<9xi32>
      %9 = memref.load %rtp_0_4[%c7] : memref<9xi32>
      %10 = memref.load %rtp_0_4[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_0_4, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_4_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_0_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_0_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_0_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_4_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_4_0_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous4[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_4_0_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous4[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous4[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_4_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_0_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_0_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_0_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_4_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_4_0_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous4[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_4_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous4[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous4[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_4_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous4[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_4_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous4[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_4_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c0_i32, %c4_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous4[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_0_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_0_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_0_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c4_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_4_0_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous4[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_4_0_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous4[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous5 = aie.buffer(%tile_5_2) {address = 13348 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_5_2 = aie.core(%tile_5_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c5_i32 = arith.constant 5 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous5[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_0_5, Acquire, 1)
      %2 = memref.load %rtp_0_5[%c0] : memref<9xi32>
      %3 = memref.load %rtp_0_5[%c1] : memref<9xi32>
      %4 = memref.load %rtp_0_5[%c2] : memref<9xi32>
      %5 = memref.load %rtp_0_5[%c3] : memref<9xi32>
      %6 = memref.load %rtp_0_5[%c4] : memref<9xi32>
      %7 = memref.load %rtp_0_5[%c5] : memref<9xi32>
      %8 = memref.load %rtp_0_5[%c6] : memref<9xi32>
      %9 = memref.load %rtp_0_5[%c7] : memref<9xi32>
      %10 = memref.load %rtp_0_5[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_0_5, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_5_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_0_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_0_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_0_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_5_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_5_0_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous5[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_5_0_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous5[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous5[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_5_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_0_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_0_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_0_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_5_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_5_0_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous5[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_5_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous5[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous5[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_5_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous5[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_5_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous5[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_5_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c0_i32, %c5_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous5[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_0_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_0_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_0_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c5_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_5_0_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous5[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_5_0_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous5[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous6 = aie.buffer(%tile_6_2) {address = 13348 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
    %core_6_2 = aie.core(%tile_6_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c6_i32 = arith.constant 6 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous6[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_0_6, Acquire, 1)
      %2 = memref.load %rtp_0_6[%c0] : memref<9xi32>
      %3 = memref.load %rtp_0_6[%c1] : memref<9xi32>
      %4 = memref.load %rtp_0_6[%c2] : memref<9xi32>
      %5 = memref.load %rtp_0_6[%c3] : memref<9xi32>
      %6 = memref.load %rtp_0_6[%c4] : memref<9xi32>
      %7 = memref.load %rtp_0_6[%c5] : memref<9xi32>
      %8 = memref.load %rtp_0_6[%c6] : memref<9xi32>
      %9 = memref.load %rtp_0_6[%c7] : memref<9xi32>
      %10 = memref.load %rtp_0_6[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_0_6, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_6_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_0_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_0_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_0_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_6_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_6_0_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous6[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_6_0_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous6[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous6[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_6_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_0_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_0_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_0_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_6_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_6_0_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous6[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_6_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous6[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous6[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_6_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous6[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_6_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous6[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_6_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c0_i32, %c6_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous6[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_0_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_0_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_0_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c6_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_6_0_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous6[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_6_0_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous6[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous7 = aie.buffer(%tile_7_2) {address = 13348 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
    %core_7_2 = aie.core(%tile_7_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c7_i32 = arith.constant 7 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous7[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_0_7, Acquire, 1)
      %2 = memref.load %rtp_0_7[%c0] : memref<9xi32>
      %3 = memref.load %rtp_0_7[%c1] : memref<9xi32>
      %4 = memref.load %rtp_0_7[%c2] : memref<9xi32>
      %5 = memref.load %rtp_0_7[%c3] : memref<9xi32>
      %6 = memref.load %rtp_0_7[%c4] : memref<9xi32>
      %7 = memref.load %rtp_0_7[%c5] : memref<9xi32>
      %8 = memref.load %rtp_0_7[%c6] : memref<9xi32>
      %9 = memref.load %rtp_0_7[%c7] : memref<9xi32>
      %10 = memref.load %rtp_0_7[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_0_7, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_7_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_0_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_0_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_0_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_7_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_7_0_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous7[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_7_0_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous7[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous7[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_7_0_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_0_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_0_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_0_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_7_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_7_0_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous7[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_7_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous7[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous7[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_7_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous7[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_7_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous7[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_7_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c0_i32, %c7_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous7[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_0_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_0_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_0_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c7_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_7_0_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous7[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_7_0_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous7[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous8 = aie.buffer(%tile_0_3) {address = 13348 : i32, sym_name = "_anonymous8"} : memref<3xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous8[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_1_0, Acquire, 1)
      %2 = memref.load %rtp_1_0[%c0] : memref<9xi32>
      %3 = memref.load %rtp_1_0[%c1] : memref<9xi32>
      %4 = memref.load %rtp_1_0[%c2] : memref<9xi32>
      %5 = memref.load %rtp_1_0[%c3] : memref<9xi32>
      %6 = memref.load %rtp_1_0[%c4] : memref<9xi32>
      %7 = memref.load %rtp_1_0[%c5] : memref<9xi32>
      %8 = memref.load %rtp_1_0[%c6] : memref<9xi32>
      %9 = memref.load %rtp_1_0[%c7] : memref<9xi32>
      %10 = memref.load %rtp_1_0[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_1_0, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_0_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_1_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_1_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_1_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_8_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_0_1_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_8_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous8[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous8[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous8[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_0_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_1_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_1_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_1_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_8_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_0_1_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_8_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous8[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous8[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous8[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_8_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_8_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous8[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_8_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_8_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous8[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_8_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c1_i32, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_8_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous8[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_1_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_1_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_1_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c0_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous8[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_0_1_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous8[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous9 = aie.buffer(%tile_1_3) {address = 13348 : i32, sym_name = "_anonymous9"} : memref<3xi32> 
    %core_1_3 = aie.core(%tile_1_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous9[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_1_1, Acquire, 1)
      %2 = memref.load %rtp_1_1[%c0] : memref<9xi32>
      %3 = memref.load %rtp_1_1[%c1] : memref<9xi32>
      %4 = memref.load %rtp_1_1[%c2] : memref<9xi32>
      %5 = memref.load %rtp_1_1[%c3] : memref<9xi32>
      %6 = memref.load %rtp_1_1[%c4] : memref<9xi32>
      %7 = memref.load %rtp_1_1[%c5] : memref<9xi32>
      %8 = memref.load %rtp_1_1[%c6] : memref<9xi32>
      %9 = memref.load %rtp_1_1[%c7] : memref<9xi32>
      %10 = memref.load %rtp_1_1[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_1_1, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_1_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_1_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_1_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_1_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_9_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_9_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_1_1_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_9_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous9[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous9[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous9[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_1_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_1_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_1_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_1_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_9_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_9_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_1_1_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_9_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous9[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous9[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous9[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_9_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_9_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_9_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous9[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_9_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_9_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_9_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous9[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_9_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_9_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c1_i32, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_9_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous9[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_1_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_1_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_1_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c1_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous9[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_1_1_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous9[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous10 = aie.buffer(%tile_2_3) {address = 13348 : i32, sym_name = "_anonymous10"} : memref<3xi32> 
    %core_2_3 = aie.core(%tile_2_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous10[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_1_2, Acquire, 1)
      %2 = memref.load %rtp_1_2[%c0] : memref<9xi32>
      %3 = memref.load %rtp_1_2[%c1] : memref<9xi32>
      %4 = memref.load %rtp_1_2[%c2] : memref<9xi32>
      %5 = memref.load %rtp_1_2[%c3] : memref<9xi32>
      %6 = memref.load %rtp_1_2[%c4] : memref<9xi32>
      %7 = memref.load %rtp_1_2[%c5] : memref<9xi32>
      %8 = memref.load %rtp_1_2[%c6] : memref<9xi32>
      %9 = memref.load %rtp_1_2[%c7] : memref<9xi32>
      %10 = memref.load %rtp_1_2[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_1_2, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_2_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_1_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_1_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_1_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_10_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_10_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_2_1_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_10_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous10[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous10[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous10[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_2_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_1_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_1_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_1_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_10_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_10_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_2_1_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_10_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous10[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous10[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous10[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_10_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_10_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_10_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous10[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_10_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_10_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_10_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous10[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_10_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_10_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c1_i32, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_10_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous10[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_1_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_1_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_1_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c2_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous10[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_2_1_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous10[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous11 = aie.buffer(%tile_3_3) {address = 13348 : i32, sym_name = "_anonymous11"} : memref<3xi32> 
    %core_3_3 = aie.core(%tile_3_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous11[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_1_3, Acquire, 1)
      %2 = memref.load %rtp_1_3[%c0] : memref<9xi32>
      %3 = memref.load %rtp_1_3[%c1] : memref<9xi32>
      %4 = memref.load %rtp_1_3[%c2] : memref<9xi32>
      %5 = memref.load %rtp_1_3[%c3] : memref<9xi32>
      %6 = memref.load %rtp_1_3[%c4] : memref<9xi32>
      %7 = memref.load %rtp_1_3[%c5] : memref<9xi32>
      %8 = memref.load %rtp_1_3[%c6] : memref<9xi32>
      %9 = memref.load %rtp_1_3[%c7] : memref<9xi32>
      %10 = memref.load %rtp_1_3[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_1_3, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_3_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_1_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_1_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_1_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_11_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_11_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_3_1_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_11_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous11[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous11[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous11[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_3_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_1_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_1_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_1_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_11_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_11_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_3_1_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_11_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous11[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous11[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous11[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_11_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_11_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_11_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous11[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_11_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_11_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_11_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous11[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_11_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_11_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c1_i32, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_11_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous11[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_1_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_1_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_1_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c3_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous11[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_3_1_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous11[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous12 = aie.buffer(%tile_4_3) {address = 13348 : i32, sym_name = "_anonymous12"} : memref<3xi32> 
    %core_4_3 = aie.core(%tile_4_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c4_i32 = arith.constant 4 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous12[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_1_4, Acquire, 1)
      %2 = memref.load %rtp_1_4[%c0] : memref<9xi32>
      %3 = memref.load %rtp_1_4[%c1] : memref<9xi32>
      %4 = memref.load %rtp_1_4[%c2] : memref<9xi32>
      %5 = memref.load %rtp_1_4[%c3] : memref<9xi32>
      %6 = memref.load %rtp_1_4[%c4] : memref<9xi32>
      %7 = memref.load %rtp_1_4[%c5] : memref<9xi32>
      %8 = memref.load %rtp_1_4[%c6] : memref<9xi32>
      %9 = memref.load %rtp_1_4[%c7] : memref<9xi32>
      %10 = memref.load %rtp_1_4[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_1_4, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_4_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_1_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_1_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_1_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_12_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_12_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_4_1_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_12_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous12[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_4_1_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous12[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_4_1_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous12[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_4_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_4_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_1_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_1_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_1_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_12_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_12_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_4_1_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_12_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous12[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_4_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous12[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_4_1_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous12[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_12_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_12_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_12_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous12[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_12_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_12_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_12_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous12[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_12_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_12_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c1_i32, %c4_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_12_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous12[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_1_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_1_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_1_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c4_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_4_1_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous12[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_4_1_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_4_1_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_4_1_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous12[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous13 = aie.buffer(%tile_5_3) {address = 13348 : i32, sym_name = "_anonymous13"} : memref<3xi32> 
    %core_5_3 = aie.core(%tile_5_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c5_i32 = arith.constant 5 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous13[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_1_5, Acquire, 1)
      %2 = memref.load %rtp_1_5[%c0] : memref<9xi32>
      %3 = memref.load %rtp_1_5[%c1] : memref<9xi32>
      %4 = memref.load %rtp_1_5[%c2] : memref<9xi32>
      %5 = memref.load %rtp_1_5[%c3] : memref<9xi32>
      %6 = memref.load %rtp_1_5[%c4] : memref<9xi32>
      %7 = memref.load %rtp_1_5[%c5] : memref<9xi32>
      %8 = memref.load %rtp_1_5[%c6] : memref<9xi32>
      %9 = memref.load %rtp_1_5[%c7] : memref<9xi32>
      %10 = memref.load %rtp_1_5[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_1_5, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_5_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_1_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_1_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_1_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_13_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_13_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_5_1_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_13_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous13[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_5_1_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous13[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_5_1_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous13[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_5_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_5_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_1_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_1_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_1_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_13_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_13_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_5_1_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_13_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous13[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_5_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous13[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_5_1_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous13[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_13_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_13_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_13_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous13[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_13_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_13_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_13_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous13[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_13_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_13_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c1_i32, %c5_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_13_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous13[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_1_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_1_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_1_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c5_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_5_1_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous13[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_5_1_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_5_1_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_5_1_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous13[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous14 = aie.buffer(%tile_6_3) {address = 13348 : i32, sym_name = "_anonymous14"} : memref<3xi32> 
    %core_6_3 = aie.core(%tile_6_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c6_i32 = arith.constant 6 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous14[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_1_6, Acquire, 1)
      %2 = memref.load %rtp_1_6[%c0] : memref<9xi32>
      %3 = memref.load %rtp_1_6[%c1] : memref<9xi32>
      %4 = memref.load %rtp_1_6[%c2] : memref<9xi32>
      %5 = memref.load %rtp_1_6[%c3] : memref<9xi32>
      %6 = memref.load %rtp_1_6[%c4] : memref<9xi32>
      %7 = memref.load %rtp_1_6[%c5] : memref<9xi32>
      %8 = memref.load %rtp_1_6[%c6] : memref<9xi32>
      %9 = memref.load %rtp_1_6[%c7] : memref<9xi32>
      %10 = memref.load %rtp_1_6[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_1_6, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_6_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_1_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_1_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_1_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_14_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_14_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_6_1_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_14_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous14[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_6_1_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous14[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_6_1_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous14[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_6_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_6_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_1_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_1_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_1_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_14_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_14_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_6_1_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_14_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous14[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_6_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous14[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_6_1_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous14[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_14_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_14_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_14_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous14[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_14_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_14_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_14_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous14[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_14_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_14_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c1_i32, %c6_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_14_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous14[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_1_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_1_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_1_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c6_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_6_1_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous14[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_6_1_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_6_1_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_6_1_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous14[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous15 = aie.buffer(%tile_7_3) {address = 13348 : i32, sym_name = "_anonymous15"} : memref<3xi32> 
    %core_7_3 = aie.core(%tile_7_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c7_i32 = arith.constant 7 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous15[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_1_7, Acquire, 1)
      %2 = memref.load %rtp_1_7[%c0] : memref<9xi32>
      %3 = memref.load %rtp_1_7[%c1] : memref<9xi32>
      %4 = memref.load %rtp_1_7[%c2] : memref<9xi32>
      %5 = memref.load %rtp_1_7[%c3] : memref<9xi32>
      %6 = memref.load %rtp_1_7[%c4] : memref<9xi32>
      %7 = memref.load %rtp_1_7[%c5] : memref<9xi32>
      %8 = memref.load %rtp_1_7[%c6] : memref<9xi32>
      %9 = memref.load %rtp_1_7[%c7] : memref<9xi32>
      %10 = memref.load %rtp_1_7[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_1_7, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_7_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_1_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_1_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_1_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_15_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_15_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_7_1_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_15_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous15[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_7_1_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous15[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_7_1_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous15[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_7_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_7_1_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_1_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_1_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_1_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_15_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_15_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_7_1_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_15_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous15[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_7_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous15[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_7_1_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous15[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_15_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_15_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_15_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous15[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_15_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_15_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_15_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous15[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_15_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_15_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c1_i32, %c7_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_15_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous15[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_1_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_1_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_1_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c7_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_7_1_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous15[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_7_1_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_7_1_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_7_1_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous15[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous16 = aie.buffer(%tile_0_4) {address = 13348 : i32, sym_name = "_anonymous16"} : memref<3xi32> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous16[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous16[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous16[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_2_0, Acquire, 1)
      %2 = memref.load %rtp_2_0[%c0] : memref<9xi32>
      %3 = memref.load %rtp_2_0[%c1] : memref<9xi32>
      %4 = memref.load %rtp_2_0[%c2] : memref<9xi32>
      %5 = memref.load %rtp_2_0[%c3] : memref<9xi32>
      %6 = memref.load %rtp_2_0[%c4] : memref<9xi32>
      %7 = memref.load %rtp_2_0[%c5] : memref<9xi32>
      %8 = memref.load %rtp_2_0[%c6] : memref<9xi32>
      %9 = memref.load %rtp_2_0[%c7] : memref<9xi32>
      %10 = memref.load %rtp_2_0[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_2_0, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_0_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_2_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_2_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_2_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_16_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_16_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_0_2_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_16_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous16[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous16[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous16[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous16[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_0_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_2_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_2_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_2_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_16_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_16_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_0_2_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_16_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous16[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous16[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous16[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous16[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_16_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_16_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_16_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous16[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_16_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_16_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_16_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous16[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_16_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_16_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c2_i32, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_16_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous16[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_2_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_2_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_2_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c0_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous16[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_0_2_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous16[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous16[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous17 = aie.buffer(%tile_1_4) {address = 13348 : i32, sym_name = "_anonymous17"} : memref<3xi32> 
    %core_1_4 = aie.core(%tile_1_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous17[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous17[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous17[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_2_1, Acquire, 1)
      %2 = memref.load %rtp_2_1[%c0] : memref<9xi32>
      %3 = memref.load %rtp_2_1[%c1] : memref<9xi32>
      %4 = memref.load %rtp_2_1[%c2] : memref<9xi32>
      %5 = memref.load %rtp_2_1[%c3] : memref<9xi32>
      %6 = memref.load %rtp_2_1[%c4] : memref<9xi32>
      %7 = memref.load %rtp_2_1[%c5] : memref<9xi32>
      %8 = memref.load %rtp_2_1[%c6] : memref<9xi32>
      %9 = memref.load %rtp_2_1[%c7] : memref<9xi32>
      %10 = memref.load %rtp_2_1[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_2_1, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_1_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_2_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_2_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_2_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_17_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_17_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_1_2_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_17_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous17[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous17[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous17[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous17[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_1_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_2_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_2_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_2_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_17_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_17_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_1_2_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_17_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous17[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous17[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous17[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous17[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_17_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_17_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_17_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous17[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_17_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_17_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_17_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous17[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_17_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_17_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c2_i32, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_17_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous17[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_2_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_2_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_2_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c1_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous17[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_1_2_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous17[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous17[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous18 = aie.buffer(%tile_2_4) {address = 13348 : i32, sym_name = "_anonymous18"} : memref<3xi32> 
    %core_2_4 = aie.core(%tile_2_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous18[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous18[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous18[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_2_2, Acquire, 1)
      %2 = memref.load %rtp_2_2[%c0] : memref<9xi32>
      %3 = memref.load %rtp_2_2[%c1] : memref<9xi32>
      %4 = memref.load %rtp_2_2[%c2] : memref<9xi32>
      %5 = memref.load %rtp_2_2[%c3] : memref<9xi32>
      %6 = memref.load %rtp_2_2[%c4] : memref<9xi32>
      %7 = memref.load %rtp_2_2[%c5] : memref<9xi32>
      %8 = memref.load %rtp_2_2[%c6] : memref<9xi32>
      %9 = memref.load %rtp_2_2[%c7] : memref<9xi32>
      %10 = memref.load %rtp_2_2[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_2_2, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_2_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_2_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_2_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_2_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_18_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_18_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_2_2_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_18_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous18[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous18[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous18[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous18[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_2_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_2_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_2_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_2_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_18_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_18_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_2_2_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_18_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous18[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous18[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous18[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous18[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_18_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_18_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_18_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous18[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_18_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_18_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_18_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous18[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_18_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_18_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c2_i32, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_18_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous18[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_2_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_2_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_2_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c2_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous18[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_2_2_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous18[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous18[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous19 = aie.buffer(%tile_3_4) {address = 13348 : i32, sym_name = "_anonymous19"} : memref<3xi32> 
    %core_3_4 = aie.core(%tile_3_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous19[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous19[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous19[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_2_3, Acquire, 1)
      %2 = memref.load %rtp_2_3[%c0] : memref<9xi32>
      %3 = memref.load %rtp_2_3[%c1] : memref<9xi32>
      %4 = memref.load %rtp_2_3[%c2] : memref<9xi32>
      %5 = memref.load %rtp_2_3[%c3] : memref<9xi32>
      %6 = memref.load %rtp_2_3[%c4] : memref<9xi32>
      %7 = memref.load %rtp_2_3[%c5] : memref<9xi32>
      %8 = memref.load %rtp_2_3[%c6] : memref<9xi32>
      %9 = memref.load %rtp_2_3[%c7] : memref<9xi32>
      %10 = memref.load %rtp_2_3[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_2_3, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_3_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_2_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_2_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_2_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_19_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_19_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_3_2_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_19_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous19[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous19[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous19[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous19[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_3_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_2_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_2_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_2_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_19_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_19_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_3_2_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_19_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous19[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous19[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous19[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous19[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_19_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_19_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_19_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous19[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_19_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_19_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_19_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous19[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_19_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_19_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c2_i32, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_19_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous19[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_2_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_2_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_2_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c3_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous19[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_3_2_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous19[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous19[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous20 = aie.buffer(%tile_4_4) {address = 13348 : i32, sym_name = "_anonymous20"} : memref<3xi32> 
    %core_4_4 = aie.core(%tile_4_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c4_i32 = arith.constant 4 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous20[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous20[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous20[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_2_4, Acquire, 1)
      %2 = memref.load %rtp_2_4[%c0] : memref<9xi32>
      %3 = memref.load %rtp_2_4[%c1] : memref<9xi32>
      %4 = memref.load %rtp_2_4[%c2] : memref<9xi32>
      %5 = memref.load %rtp_2_4[%c3] : memref<9xi32>
      %6 = memref.load %rtp_2_4[%c4] : memref<9xi32>
      %7 = memref.load %rtp_2_4[%c5] : memref<9xi32>
      %8 = memref.load %rtp_2_4[%c6] : memref<9xi32>
      %9 = memref.load %rtp_2_4[%c7] : memref<9xi32>
      %10 = memref.load %rtp_2_4[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_2_4, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_4_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_4_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_2_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_2_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_2_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_20_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_20_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_4_2_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_20_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous20[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_4_2_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous20[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_4_2_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous20[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous20[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_4_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_4_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_4_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_2_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_2_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_2_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_20_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_20_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_4_2_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_20_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous20[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_4_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous20[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_4_2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous20[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous20[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_20_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_20_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_20_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous20[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_20_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_20_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_20_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous20[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_20_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_20_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c2_i32, %c4_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_20_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous20[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_4_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_2_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_2_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_2_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c4_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_4_2_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous20[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_4_2_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_4_2_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_4_2_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous20[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous20[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous21 = aie.buffer(%tile_5_4) {address = 13348 : i32, sym_name = "_anonymous21"} : memref<3xi32> 
    %core_5_4 = aie.core(%tile_5_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c5_i32 = arith.constant 5 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous21[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous21[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous21[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_2_5, Acquire, 1)
      %2 = memref.load %rtp_2_5[%c0] : memref<9xi32>
      %3 = memref.load %rtp_2_5[%c1] : memref<9xi32>
      %4 = memref.load %rtp_2_5[%c2] : memref<9xi32>
      %5 = memref.load %rtp_2_5[%c3] : memref<9xi32>
      %6 = memref.load %rtp_2_5[%c4] : memref<9xi32>
      %7 = memref.load %rtp_2_5[%c5] : memref<9xi32>
      %8 = memref.load %rtp_2_5[%c6] : memref<9xi32>
      %9 = memref.load %rtp_2_5[%c7] : memref<9xi32>
      %10 = memref.load %rtp_2_5[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_2_5, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_5_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_5_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_2_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_2_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_2_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_21_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_21_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_5_2_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_21_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous21[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_5_2_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous21[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_5_2_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous21[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous21[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_5_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_5_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_5_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_2_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_2_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_2_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_21_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_21_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_5_2_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_21_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous21[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_5_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous21[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_5_2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous21[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous21[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_21_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_21_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_21_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous21[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_21_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_21_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_21_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous21[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_21_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_21_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c2_i32, %c5_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_21_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous21[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_5_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_2_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_2_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_2_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c5_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_5_2_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous21[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_5_2_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_5_2_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_5_2_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous21[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous21[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous22 = aie.buffer(%tile_6_4) {address = 13348 : i32, sym_name = "_anonymous22"} : memref<3xi32> 
    %core_6_4 = aie.core(%tile_6_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c6_i32 = arith.constant 6 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous22[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous22[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous22[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_2_6, Acquire, 1)
      %2 = memref.load %rtp_2_6[%c0] : memref<9xi32>
      %3 = memref.load %rtp_2_6[%c1] : memref<9xi32>
      %4 = memref.load %rtp_2_6[%c2] : memref<9xi32>
      %5 = memref.load %rtp_2_6[%c3] : memref<9xi32>
      %6 = memref.load %rtp_2_6[%c4] : memref<9xi32>
      %7 = memref.load %rtp_2_6[%c5] : memref<9xi32>
      %8 = memref.load %rtp_2_6[%c6] : memref<9xi32>
      %9 = memref.load %rtp_2_6[%c7] : memref<9xi32>
      %10 = memref.load %rtp_2_6[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_2_6, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_6_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_6_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_2_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_2_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_2_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_22_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_22_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_6_2_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_22_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous22[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_6_2_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous22[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_6_2_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous22[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous22[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_6_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_6_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_6_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_2_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_2_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_2_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_22_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_22_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_6_2_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_22_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous22[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_6_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous22[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_6_2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous22[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous22[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_22_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_22_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_22_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous22[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_22_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_22_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_22_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous22[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_22_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_22_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c2_i32, %c6_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_22_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous22[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_6_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_2_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_2_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_2_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c6_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_6_2_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous22[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_6_2_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_6_2_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_6_2_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous22[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous22[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous23 = aie.buffer(%tile_7_4) {address = 13348 : i32, sym_name = "_anonymous23"} : memref<3xi32> 
    %core_7_4 = aie.core(%tile_7_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c7_i32 = arith.constant 7 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous23[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous23[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous23[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_2_7, Acquire, 1)
      %2 = memref.load %rtp_2_7[%c0] : memref<9xi32>
      %3 = memref.load %rtp_2_7[%c1] : memref<9xi32>
      %4 = memref.load %rtp_2_7[%c2] : memref<9xi32>
      %5 = memref.load %rtp_2_7[%c3] : memref<9xi32>
      %6 = memref.load %rtp_2_7[%c4] : memref<9xi32>
      %7 = memref.load %rtp_2_7[%c5] : memref<9xi32>
      %8 = memref.load %rtp_2_7[%c6] : memref<9xi32>
      %9 = memref.load %rtp_2_7[%c7] : memref<9xi32>
      %10 = memref.load %rtp_2_7[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_2_7, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_7_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_7_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_2_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_2_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_2_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_23_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_23_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_7_2_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_23_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous23[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_7_2_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous23[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_7_2_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous23[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous23[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_7_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_7_2_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_7_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_2_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_2_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_2_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_23_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_23_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_7_2_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_23_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous23[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_7_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous23[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_7_2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous23[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous23[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_23_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_23_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_23_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous23[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_23_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_23_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_23_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous23[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_23_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_23_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c2_i32, %c7_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_23_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous23[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_7_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_2_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_2_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_2_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c7_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_7_2_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous23[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_7_2_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_7_2_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_7_2_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous23[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous23[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous24 = aie.buffer(%tile_0_5) {address = 13348 : i32, sym_name = "_anonymous24"} : memref<3xi32> 
    %core_0_5 = aie.core(%tile_0_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous24[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous24[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous24[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_3_0, Acquire, 1)
      %2 = memref.load %rtp_3_0[%c0] : memref<9xi32>
      %3 = memref.load %rtp_3_0[%c1] : memref<9xi32>
      %4 = memref.load %rtp_3_0[%c2] : memref<9xi32>
      %5 = memref.load %rtp_3_0[%c3] : memref<9xi32>
      %6 = memref.load %rtp_3_0[%c4] : memref<9xi32>
      %7 = memref.load %rtp_3_0[%c5] : memref<9xi32>
      %8 = memref.load %rtp_3_0[%c6] : memref<9xi32>
      %9 = memref.load %rtp_3_0[%c7] : memref<9xi32>
      %10 = memref.load %rtp_3_0[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_3_0, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_0_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_3_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_3_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_3_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_24_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_24_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_0_3_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_24_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous24[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous24[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous24[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous24[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_0_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_3_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_3_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_0_3_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_24_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_24_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_0_3_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_24_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous24[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous24[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous24[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous24[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_24_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_24_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_24_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous24[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_24_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_24_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_24_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous24[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_24_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_24_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c3_i32, %c0_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_24_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous24[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_3_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_3_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_0_3_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c0_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous24[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_0_3_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous24[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous24[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous25 = aie.buffer(%tile_1_5) {address = 13348 : i32, sym_name = "_anonymous25"} : memref<3xi32> 
    %core_1_5 = aie.core(%tile_1_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous25[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous25[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous25[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_3_1, Acquire, 1)
      %2 = memref.load %rtp_3_1[%c0] : memref<9xi32>
      %3 = memref.load %rtp_3_1[%c1] : memref<9xi32>
      %4 = memref.load %rtp_3_1[%c2] : memref<9xi32>
      %5 = memref.load %rtp_3_1[%c3] : memref<9xi32>
      %6 = memref.load %rtp_3_1[%c4] : memref<9xi32>
      %7 = memref.load %rtp_3_1[%c5] : memref<9xi32>
      %8 = memref.load %rtp_3_1[%c6] : memref<9xi32>
      %9 = memref.load %rtp_3_1[%c7] : memref<9xi32>
      %10 = memref.load %rtp_3_1[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_3_1, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_1_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_3_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_3_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_3_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_25_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_25_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_1_3_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_25_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous25[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous25[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous25[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous25[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_1_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_3_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_3_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_1_3_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_25_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_25_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_1_3_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_25_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous25[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous25[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous25[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous25[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_25_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_25_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_25_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous25[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_25_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_25_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_25_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous25[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_25_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_25_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c3_i32, %c1_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_25_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous25[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_3_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_3_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_1_3_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c1_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous25[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_1_3_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous25[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous25[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous26 = aie.buffer(%tile_2_5) {address = 13348 : i32, sym_name = "_anonymous26"} : memref<3xi32> 
    %core_2_5 = aie.core(%tile_2_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous26[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous26[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous26[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_3_2, Acquire, 1)
      %2 = memref.load %rtp_3_2[%c0] : memref<9xi32>
      %3 = memref.load %rtp_3_2[%c1] : memref<9xi32>
      %4 = memref.load %rtp_3_2[%c2] : memref<9xi32>
      %5 = memref.load %rtp_3_2[%c3] : memref<9xi32>
      %6 = memref.load %rtp_3_2[%c4] : memref<9xi32>
      %7 = memref.load %rtp_3_2[%c5] : memref<9xi32>
      %8 = memref.load %rtp_3_2[%c6] : memref<9xi32>
      %9 = memref.load %rtp_3_2[%c7] : memref<9xi32>
      %10 = memref.load %rtp_3_2[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_3_2, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_2_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_3_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_3_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_3_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_26_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_26_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_2_3_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_26_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous26[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous26[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous26[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous26[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_2_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_3_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_3_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_2_3_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_26_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_26_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_2_3_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_26_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous26[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous26[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous26[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous26[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_26_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_26_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_26_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous26[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_26_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_26_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_26_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous26[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_26_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_26_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c3_i32, %c2_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_26_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous26[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_3_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_3_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_2_3_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c2_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous26[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_2_3_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous26[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous26[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous27 = aie.buffer(%tile_3_5) {address = 13348 : i32, sym_name = "_anonymous27"} : memref<3xi32> 
    %core_3_5 = aie.core(%tile_3_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous27[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous27[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous27[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_3_3, Acquire, 1)
      %2 = memref.load %rtp_3_3[%c0] : memref<9xi32>
      %3 = memref.load %rtp_3_3[%c1] : memref<9xi32>
      %4 = memref.load %rtp_3_3[%c2] : memref<9xi32>
      %5 = memref.load %rtp_3_3[%c3] : memref<9xi32>
      %6 = memref.load %rtp_3_3[%c4] : memref<9xi32>
      %7 = memref.load %rtp_3_3[%c5] : memref<9xi32>
      %8 = memref.load %rtp_3_3[%c6] : memref<9xi32>
      %9 = memref.load %rtp_3_3[%c7] : memref<9xi32>
      %10 = memref.load %rtp_3_3[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_3_3, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_3_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_3_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_3_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_3_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_27_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_27_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_3_3_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_27_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous27[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous27[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous27[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous27[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_3_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_3_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_3_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_3_3_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_27_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_27_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_3_3_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_27_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous27[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous27[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous27[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous27[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_27_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_27_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_27_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous27[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_27_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_27_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_27_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous27[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_27_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_27_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c3_i32, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_27_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous27[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_3_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_3_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_3_3_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c3_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous27[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_3_3_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous27[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous27[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous28 = aie.buffer(%tile_4_5) {address = 13348 : i32, sym_name = "_anonymous28"} : memref<3xi32> 
    %core_4_5 = aie.core(%tile_4_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous28[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous28[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous28[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_3_4, Acquire, 1)
      %2 = memref.load %rtp_3_4[%c0] : memref<9xi32>
      %3 = memref.load %rtp_3_4[%c1] : memref<9xi32>
      %4 = memref.load %rtp_3_4[%c2] : memref<9xi32>
      %5 = memref.load %rtp_3_4[%c3] : memref<9xi32>
      %6 = memref.load %rtp_3_4[%c4] : memref<9xi32>
      %7 = memref.load %rtp_3_4[%c5] : memref<9xi32>
      %8 = memref.load %rtp_3_4[%c6] : memref<9xi32>
      %9 = memref.load %rtp_3_4[%c7] : memref<9xi32>
      %10 = memref.load %rtp_3_4[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_3_4, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_4_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_4_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_3_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_3_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_3_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_28_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_28_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_4_3_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_28_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous28[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_4_3_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous28[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_4_3_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous28[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous28[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_4_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_4_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_4_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_3_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_3_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_4_3_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_28_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_28_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_4_3_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_28_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous28[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_4_3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous28[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_4_3_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous28[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous28[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_28_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_28_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_28_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous28[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_28_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_28_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_28_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous28[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_28_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_28_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c3_i32, %c4_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_28_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous28[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_4_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_3_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_3_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_4_3_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c4_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_4_3_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous28[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_4_3_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_4_3_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_4_3_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous28[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous28[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous29 = aie.buffer(%tile_5_5) {address = 13348 : i32, sym_name = "_anonymous29"} : memref<3xi32> 
    %core_5_5 = aie.core(%tile_5_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c5_i32 = arith.constant 5 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous29[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous29[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous29[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_3_5, Acquire, 1)
      %2 = memref.load %rtp_3_5[%c0] : memref<9xi32>
      %3 = memref.load %rtp_3_5[%c1] : memref<9xi32>
      %4 = memref.load %rtp_3_5[%c2] : memref<9xi32>
      %5 = memref.load %rtp_3_5[%c3] : memref<9xi32>
      %6 = memref.load %rtp_3_5[%c4] : memref<9xi32>
      %7 = memref.load %rtp_3_5[%c5] : memref<9xi32>
      %8 = memref.load %rtp_3_5[%c6] : memref<9xi32>
      %9 = memref.load %rtp_3_5[%c7] : memref<9xi32>
      %10 = memref.load %rtp_3_5[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_3_5, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_5_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_5_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_3_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_3_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_3_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_29_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_29_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_5_3_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_29_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous29[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_5_3_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous29[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_5_3_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous29[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous29[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_5_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_5_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_5_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_3_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_3_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_5_3_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_29_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_29_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_5_3_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_29_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous29[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_5_3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous29[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_5_3_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous29[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous29[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_29_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_29_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_29_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous29[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_29_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_29_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_29_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous29[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_29_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_29_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c3_i32, %c5_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_29_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous29[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_5_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_3_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_3_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_5_3_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c5_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_5_3_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous29[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_5_3_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_5_3_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_5_3_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous29[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous29[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous30 = aie.buffer(%tile_6_5) {address = 13348 : i32, sym_name = "_anonymous30"} : memref<3xi32> 
    %core_6_5 = aie.core(%tile_6_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c6_i32 = arith.constant 6 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous30[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous30[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous30[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_3_6, Acquire, 1)
      %2 = memref.load %rtp_3_6[%c0] : memref<9xi32>
      %3 = memref.load %rtp_3_6[%c1] : memref<9xi32>
      %4 = memref.load %rtp_3_6[%c2] : memref<9xi32>
      %5 = memref.load %rtp_3_6[%c3] : memref<9xi32>
      %6 = memref.load %rtp_3_6[%c4] : memref<9xi32>
      %7 = memref.load %rtp_3_6[%c5] : memref<9xi32>
      %8 = memref.load %rtp_3_6[%c6] : memref<9xi32>
      %9 = memref.load %rtp_3_6[%c7] : memref<9xi32>
      %10 = memref.load %rtp_3_6[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_3_6, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_6_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_6_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_3_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_3_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_3_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_30_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_30_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_6_3_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_30_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous30[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_6_3_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous30[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_6_3_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous30[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous30[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_6_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_6_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_6_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_3_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_3_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_6_3_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_30_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_30_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_6_3_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_30_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous30[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_6_3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous30[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_6_3_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous30[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous30[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_30_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_30_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_30_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous30[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_30_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_30_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_30_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous30[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_30_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_30_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c3_i32, %c6_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_30_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous30[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_6_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_3_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_3_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_6_3_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c6_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_6_3_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous30[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_6_3_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_6_3_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_6_3_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous30[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous30[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    %_anonymous31 = aie.buffer(%tile_7_5) {address = 13348 : i32, sym_name = "_anonymous31"} : memref<3xi32> 
    %core_7_5 = aie.core(%tile_7_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c3_i32 = arith.constant 3 : i32
      %c7_i32 = arith.constant 7 : i32
      %c32 = arith.constant 32 : index
      %c36 = arith.constant 36 : index
      %c34 = arith.constant 34 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous31[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous31[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous31[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb67
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb68
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rtp_lock_3_7, Acquire, 1)
      %2 = memref.load %rtp_3_7[%c0] : memref<9xi32>
      %3 = memref.load %rtp_3_7[%c1] : memref<9xi32>
      %4 = memref.load %rtp_3_7[%c2] : memref<9xi32>
      %5 = memref.load %rtp_3_7[%c3] : memref<9xi32>
      %6 = memref.load %rtp_3_7[%c4] : memref<9xi32>
      %7 = memref.load %rtp_3_7[%c5] : memref<9xi32>
      %8 = memref.load %rtp_3_7[%c6] : memref<9xi32>
      %9 = memref.load %rtp_3_7[%c7] : memref<9xi32>
      %10 = memref.load %rtp_3_7[%c8] : memref<9xi32>
      aie.use_lock(%rtp_lock_3_7, Release, 1)
      %11 = arith.index_cast %2 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %4 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.index_cast %10 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%16: index):  // 2 preds: ^bb2, ^bb18
      %17 = arith.cmpi slt, %16, %12 : index
      cf.cond_br %17, ^bb4, ^bb19
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_7_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%18: index):  // 2 preds: ^bb4, ^bb17
      %19 = arith.cmpi slt, %18, %11 : index
      cf.cond_br %19, ^bb6, ^bb18
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_7_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %20 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %21 : index to i32
      cf.switch %22 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_3_cons_buff_0 : memref<1536xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_3_cons_buff_1 : memref<1536xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_3_cons_buff_0 : memref<1536xui8>)
    ^bb10(%23: memref<1536xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%24: index):  // 2 preds: ^bb10, ^bb16
      %25 = arith.cmpi slt, %24, %15 : index
      cf.cond_br %25, ^bb12, ^bb17
    ^bb12:  // pred: ^bb11
      aie.use_lock(%A_L2L1_31_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_31_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb16(%29: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_p1(%29, %23, %C_L1L2_7_3_buff_0, %2, %6, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_31_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous31[%c2] : memref<3xi32>
      %35 = arith.addi %24, %c1 : index
      cf.br ^bb11(%35 : index)
    ^bb17:  // pred: ^bb11
      aie.use_lock(%B_L2L1_7_3_cons_prod_lock_0, Release, 1)
      %36 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous31[%c1] : memref<3xi32>
      %41 = arith.addi %18, %c1 : index
      cf.br ^bb5(%41 : index)
    ^bb18:  // pred: ^bb5
      aie.use_lock(%C_L1L2_7_3_cons_lock_0, Release, 1)
      %42 = memref.load %_anonymous31[%c0] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c1_i32 : i32
      %45 = arith.select %44, %42, %43 : i32
      memref.store %45, %_anonymous31[%c0] : memref<3xi32>
      %46 = arith.addi %16, %c1 : index
      cf.br ^bb3(%46 : index)
    ^bb19:  // pred: ^bb3
      cf.br ^bb20(%c0 : index)
    ^bb20(%47: index):  // 2 preds: ^bb19, ^bb35
      %48 = arith.cmpi slt, %47, %14 : index
      cf.cond_br %48, ^bb21, ^bb36
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L1L2_7_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel(%C_L1L2_7_3_buff_0, %10) : (memref<4096xui8>, i32) -> ()
      cf.br ^bb22(%c0 : index)
    ^bb22(%49: index):  // 2 preds: ^bb21, ^bb34
      %50 = arith.cmpi slt, %49, %13 : index
      cf.cond_br %50, ^bb23, ^bb35
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L2L1_7_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %51 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.index_cast %52 : index to i32
      cf.switch %53 : i32, [
        default: ^bb26,
        0: ^bb24,
        1: ^bb25
      ]
    ^bb24:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_3_cons_buff_0 : memref<1536xui8>)
    ^bb25:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_3_cons_buff_1 : memref<1536xui8>)
    ^bb26:  // pred: ^bb23
      cf.br ^bb27(%B_L2L1_7_3_cons_buff_0 : memref<1536xui8>)
    ^bb27(%54: memref<1536xui8>):  // 3 preds: ^bb24, ^bb25, ^bb26
      cf.br ^bb28(%c0 : index)
    ^bb28(%55: index):  // 2 preds: ^bb27, ^bb33
      %56 = arith.cmpi slt, %55, %15 : index
      cf.cond_br %56, ^bb29, ^bb34
    ^bb29:  // pred: ^bb28
      aie.use_lock(%A_L2L1_31_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %57 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.index_cast %58 : index to i32
      cf.switch %59 : i32, [
        default: ^bb32,
        0: ^bb30,
        1: ^bb31
      ]
    ^bb30:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_31_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb32:  // pred: ^bb29
      cf.br ^bb33(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb33(%60: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb30, ^bb31, ^bb32
      func.call @matmul_dn(%60, %54, %C_L1L2_7_3_buff_0, %4, %9, %10) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, memref<1536xui8>, memref<4096xui8>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_31_cons_prod_lock_0, Release, 1)
      %61 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous31[%c2] : memref<3xi32>
      %66 = arith.addi %55, %c1 : index
      cf.br ^bb28(%66 : index)
    ^bb34:  // pred: ^bb28
      aie.use_lock(%B_L2L1_7_3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous31[%c1] : memref<3xi32>
      %72 = arith.addi %49, %c1 : index
      cf.br ^bb22(%72 : index)
    ^bb35:  // pred: ^bb22
      aie.use_lock(%C_L1L2_7_3_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous31[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous31[%c0] : memref<3xi32>
      %77 = arith.addi %47, %c1 : index
      cf.br ^bb20(%77 : index)
    ^bb36:  // pred: ^bb20
      %78 = arith.index_cast %7 : i32 to index
      %79 = arith.index_cast %8 : i32 to index
      cf.br ^bb37(%c0 : index)
    ^bb37(%80: index):  // 2 preds: ^bb36, ^bb66
      %81 = arith.cmpi slt, %80, %78 : index
      cf.cond_br %81, ^bb38, ^bb67
    ^bb38:  // pred: ^bb37
      aie.use_lock(%A_L2L1_31_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %82 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %83 = arith.index_cast %82 : i32 to index
      %84 = arith.index_cast %83 : index to i32
      cf.switch %84 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_31_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb42(%85: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @attn_begin(%85, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32) -> ()
      aie.use_lock(%A_L2L1_31_cons_prod_lock_0, Release, 1)
      %86 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %87 = arith.addi %86, %c1_i32 : i32
      %88 = arith.cmpi sge, %87, %c2_i32 : i32
      %89 = arith.subi %87, %c2_i32 : i32
      %90 = arith.select %88, %89, %87 : i32
      memref.store %90, %_anonymous31[%c2] : memref<3xi32>
      cf.br ^bb43(%c0 : index)
    ^bb43(%91: index):  // 2 preds: ^bb42, ^bb48
      %92 = arith.cmpi slt, %91, %c32 : index
      cf.cond_br %92, ^bb44, ^bb49
    ^bb44:  // pred: ^bb43
      aie.use_lock(%A_L2L1_31_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %93 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %94 = arith.index_cast %93 : i32 to index
      %95 = arith.index_cast %94 : index to i32
      cf.switch %95 : i32, [
        default: ^bb47,
        0: ^bb45,
        1: ^bb46
      ]
    ^bb45:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb46:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_31_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb47:  // pred: ^bb44
      cf.br ^bb48(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb48(%96: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb45, ^bb46, ^bb47
      %97 = arith.index_cast %91 : index to i32
      func.call @attn_q(%96, %97, %c3_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32) -> ()
      aie.use_lock(%A_L2L1_31_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous31[%c2] : memref<3xi32>
      %103 = arith.addi %91, %c1 : index
      cf.br ^bb43(%103 : index)
    ^bb49:  // pred: ^bb43
      cf.br ^bb50(%c0 : index)
    ^bb50(%104: index):  // 2 preds: ^bb49, ^bb55
      %105 = arith.cmpi slt, %104, %c36 : index
      cf.cond_br %105, ^bb51, ^bb56
    ^bb51:  // pred: ^bb50
      aie.use_lock(%A_L2L1_31_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %106 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %107 = arith.index_cast %106 : i32 to index
      %108 = arith.index_cast %107 : index to i32
      cf.switch %108 : i32, [
        default: ^bb54,
        0: ^bb52,
        1: ^bb53
      ]
    ^bb52:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb53:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_31_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb54:  // pred: ^bb51
      cf.br ^bb55(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb55(%109: memref<16x8x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb52, ^bb53, ^bb54
      %110 = arith.index_cast %104 : index to i32
      func.call @attn_tail(%109, %110, %c3_i32, %c7_i32) : (memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, i32, i32, i32) -> ()
      aie.use_lock(%A_L2L1_31_cons_prod_lock_0, Release, 1)
      %111 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %112 = arith.addi %111, %c1_i32 : i32
      %113 = arith.cmpi sge, %112, %c2_i32 : i32
      %114 = arith.subi %112, %c2_i32 : i32
      %115 = arith.select %113, %114, %112 : i32
      memref.store %115, %_anonymous31[%c2] : memref<3xi32>
      %116 = arith.addi %104, %c1 : index
      cf.br ^bb50(%116 : index)
    ^bb56:  // pred: ^bb50
      cf.br ^bb57(%c0 : index)
    ^bb57(%117: index):  // 2 preds: ^bb56, ^bb62
      %118 = arith.cmpi slt, %117, %79 : index
      cf.cond_br %118, ^bb58, ^bb63
    ^bb58:  // pred: ^bb57
      aie.use_lock(%B_L2L1_7_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %119 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %120 = arith.index_cast %119 : i32 to index
      %121 = arith.index_cast %120 : index to i32
      cf.switch %121 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_3_cons_buff_0 : memref<1536xui8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_3_cons_buff_1 : memref<1536xui8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%B_L2L1_7_3_cons_buff_0 : memref<1536xui8>)
    ^bb62(%122: memref<1536xui8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @attn_feed(%122, %c7_i32) : (memref<1536xui8>, i32) -> ()
      aie.use_lock(%B_L2L1_7_3_cons_prod_lock_0, Release, 1)
      %123 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c2_i32 : i32
      %126 = arith.subi %124, %c2_i32 : i32
      %127 = arith.select %125, %126, %124 : i32
      memref.store %127, %_anonymous31[%c1] : memref<3xi32>
      %128 = arith.addi %117, %c1 : index
      cf.br ^bb57(%128 : index)
    ^bb63:  // pred: ^bb57
      cf.br ^bb64(%c0 : index)
    ^bb64(%129: index):  // 2 preds: ^bb63, ^bb65
      %130 = arith.cmpi slt, %129, %c34 : index
      cf.cond_br %130, ^bb65, ^bb66
    ^bb65:  // pred: ^bb64
      aie.use_lock(%C_L1L2_7_3_prod_lock_0, AcquireGreaterEqual, 1)
      %131 = arith.index_cast %129 : index to i32
      func.call @attn_out(%C_L1L2_7_3_buff_0, %131) : (memref<4096xui8>, i32) -> ()
      aie.use_lock(%C_L1L2_7_3_cons_lock_0, Release, 1)
      %132 = memref.load %_anonymous31[%c0] : memref<3xi32>
      %133 = arith.addi %132, %c1_i32 : i32
      %134 = arith.cmpi sge, %133, %c1_i32 : i32
      %135 = arith.select %134, %132, %133 : i32
      memref.store %135, %_anonymous31[%c0] : memref<3xi32>
      %136 = arith.addi %129, %c1 : index
      cf.br ^bb64(%136 : index)
    ^bb66:  // pred: ^bb64
      %137 = arith.addi %80, %c1 : index
      cf.br ^bb37(%137 : index)
    ^bb67:  // pred: ^bb37
      %138 = arith.addi %0, %c1 : index
      cf.br ^bb1(%138 : index)
    ^bb68:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_16x64x128.o"], stack_size = 3840 : i32}
    aie.runtime_sequence(%arg0: memref<16384x!aiex.bfp<"v8bfp16ebs8">>, %arg1: memref<35389440xui8>, %arg2: memref<2228224xui8>) {
      aiex.npu.rtp_write(@rtp_0_0, 0, 32)
      aiex.npu.rtp_write(@rtp_0_0, 1, 15)
      aiex.npu.rtp_write(@rtp_0_0, 2, 120)
      aiex.npu.rtp_write(@rtp_0_0, 3, 2)
      aiex.npu.rtp_write(@rtp_0_0, 4, 0)
      aiex.npu.rtp_write(@rtp_0_0, 5, 0)
      aiex.npu.rtp_write(@rtp_0_0, 6, 0)
      aiex.npu.rtp_write(@rtp_0_0, 7, 0)
      aiex.npu.rtp_write(@rtp_0_0, 8, 4)
      aiex.set_lock(%rtp_lock_0_0, 1)
      aiex.npu.rtp_write(@rtp_0_1, 0, 32)
      aiex.npu.rtp_write(@rtp_0_1, 1, 15)
      aiex.npu.rtp_write(@rtp_0_1, 2, 120)
      aiex.npu.rtp_write(@rtp_0_1, 3, 2)
      aiex.npu.rtp_write(@rtp_0_1, 4, 0)
      aiex.npu.rtp_write(@rtp_0_1, 5, 0)
      aiex.npu.rtp_write(@rtp_0_1, 6, 0)
      aiex.npu.rtp_write(@rtp_0_1, 7, 0)
      aiex.npu.rtp_write(@rtp_0_1, 8, 4)
      aiex.set_lock(%rtp_lock_0_1, 1)
      aiex.npu.rtp_write(@rtp_0_2, 0, 32)
      aiex.npu.rtp_write(@rtp_0_2, 1, 15)
      aiex.npu.rtp_write(@rtp_0_2, 2, 120)
      aiex.npu.rtp_write(@rtp_0_2, 3, 2)
      aiex.npu.rtp_write(@rtp_0_2, 4, 0)
      aiex.npu.rtp_write(@rtp_0_2, 5, 0)
      aiex.npu.rtp_write(@rtp_0_2, 6, 0)
      aiex.npu.rtp_write(@rtp_0_2, 7, 0)
      aiex.npu.rtp_write(@rtp_0_2, 8, 4)
      aiex.set_lock(%rtp_lock_0_2, 1)
      aiex.npu.rtp_write(@rtp_0_3, 0, 32)
      aiex.npu.rtp_write(@rtp_0_3, 1, 15)
      aiex.npu.rtp_write(@rtp_0_3, 2, 120)
      aiex.npu.rtp_write(@rtp_0_3, 3, 2)
      aiex.npu.rtp_write(@rtp_0_3, 4, 0)
      aiex.npu.rtp_write(@rtp_0_3, 5, 0)
      aiex.npu.rtp_write(@rtp_0_3, 6, 0)
      aiex.npu.rtp_write(@rtp_0_3, 7, 0)
      aiex.npu.rtp_write(@rtp_0_3, 8, 4)
      aiex.set_lock(%rtp_lock_0_3, 1)
      aiex.npu.rtp_write(@rtp_0_4, 0, 32)
      aiex.npu.rtp_write(@rtp_0_4, 1, 15)
      aiex.npu.rtp_write(@rtp_0_4, 2, 120)
      aiex.npu.rtp_write(@rtp_0_4, 3, 2)
      aiex.npu.rtp_write(@rtp_0_4, 4, 0)
      aiex.npu.rtp_write(@rtp_0_4, 5, 0)
      aiex.npu.rtp_write(@rtp_0_4, 6, 0)
      aiex.npu.rtp_write(@rtp_0_4, 7, 0)
      aiex.npu.rtp_write(@rtp_0_4, 8, 4)
      aiex.set_lock(%rtp_lock_0_4, 1)
      aiex.npu.rtp_write(@rtp_0_5, 0, 32)
      aiex.npu.rtp_write(@rtp_0_5, 1, 15)
      aiex.npu.rtp_write(@rtp_0_5, 2, 120)
      aiex.npu.rtp_write(@rtp_0_5, 3, 2)
      aiex.npu.rtp_write(@rtp_0_5, 4, 0)
      aiex.npu.rtp_write(@rtp_0_5, 5, 0)
      aiex.npu.rtp_write(@rtp_0_5, 6, 0)
      aiex.npu.rtp_write(@rtp_0_5, 7, 0)
      aiex.npu.rtp_write(@rtp_0_5, 8, 4)
      aiex.set_lock(%rtp_lock_0_5, 1)
      aiex.npu.rtp_write(@rtp_0_6, 0, 32)
      aiex.npu.rtp_write(@rtp_0_6, 1, 15)
      aiex.npu.rtp_write(@rtp_0_6, 2, 120)
      aiex.npu.rtp_write(@rtp_0_6, 3, 2)
      aiex.npu.rtp_write(@rtp_0_6, 4, 0)
      aiex.npu.rtp_write(@rtp_0_6, 5, 0)
      aiex.npu.rtp_write(@rtp_0_6, 6, 0)
      aiex.npu.rtp_write(@rtp_0_6, 7, 0)
      aiex.npu.rtp_write(@rtp_0_6, 8, 4)
      aiex.set_lock(%rtp_lock_0_6, 1)
      aiex.npu.rtp_write(@rtp_0_7, 0, 32)
      aiex.npu.rtp_write(@rtp_0_7, 1, 15)
      aiex.npu.rtp_write(@rtp_0_7, 2, 120)
      aiex.npu.rtp_write(@rtp_0_7, 3, 2)
      aiex.npu.rtp_write(@rtp_0_7, 4, 0)
      aiex.npu.rtp_write(@rtp_0_7, 5, 0)
      aiex.npu.rtp_write(@rtp_0_7, 6, 0)
      aiex.npu.rtp_write(@rtp_0_7, 7, 0)
      aiex.npu.rtp_write(@rtp_0_7, 8, 4)
      aiex.set_lock(%rtp_lock_0_7, 1)
      aiex.npu.rtp_write(@rtp_1_0, 0, 32)
      aiex.npu.rtp_write(@rtp_1_0, 1, 15)
      aiex.npu.rtp_write(@rtp_1_0, 2, 120)
      aiex.npu.rtp_write(@rtp_1_0, 3, 2)
      aiex.npu.rtp_write(@rtp_1_0, 4, 0)
      aiex.npu.rtp_write(@rtp_1_0, 5, 0)
      aiex.npu.rtp_write(@rtp_1_0, 6, 0)
      aiex.npu.rtp_write(@rtp_1_0, 7, 0)
      aiex.npu.rtp_write(@rtp_1_0, 8, 4)
      aiex.set_lock(%rtp_lock_1_0, 1)
      aiex.npu.rtp_write(@rtp_1_1, 0, 32)
      aiex.npu.rtp_write(@rtp_1_1, 1, 15)
      aiex.npu.rtp_write(@rtp_1_1, 2, 120)
      aiex.npu.rtp_write(@rtp_1_1, 3, 2)
      aiex.npu.rtp_write(@rtp_1_1, 4, 0)
      aiex.npu.rtp_write(@rtp_1_1, 5, 0)
      aiex.npu.rtp_write(@rtp_1_1, 6, 0)
      aiex.npu.rtp_write(@rtp_1_1, 7, 0)
      aiex.npu.rtp_write(@rtp_1_1, 8, 4)
      aiex.set_lock(%rtp_lock_1_1, 1)
      aiex.npu.rtp_write(@rtp_1_2, 0, 32)
      aiex.npu.rtp_write(@rtp_1_2, 1, 15)
      aiex.npu.rtp_write(@rtp_1_2, 2, 120)
      aiex.npu.rtp_write(@rtp_1_2, 3, 2)
      aiex.npu.rtp_write(@rtp_1_2, 4, 0)
      aiex.npu.rtp_write(@rtp_1_2, 5, 0)
      aiex.npu.rtp_write(@rtp_1_2, 6, 0)
      aiex.npu.rtp_write(@rtp_1_2, 7, 0)
      aiex.npu.rtp_write(@rtp_1_2, 8, 4)
      aiex.set_lock(%rtp_lock_1_2, 1)
      aiex.npu.rtp_write(@rtp_1_3, 0, 32)
      aiex.npu.rtp_write(@rtp_1_3, 1, 15)
      aiex.npu.rtp_write(@rtp_1_3, 2, 120)
      aiex.npu.rtp_write(@rtp_1_3, 3, 2)
      aiex.npu.rtp_write(@rtp_1_3, 4, 0)
      aiex.npu.rtp_write(@rtp_1_3, 5, 0)
      aiex.npu.rtp_write(@rtp_1_3, 6, 0)
      aiex.npu.rtp_write(@rtp_1_3, 7, 0)
      aiex.npu.rtp_write(@rtp_1_3, 8, 4)
      aiex.set_lock(%rtp_lock_1_3, 1)
      aiex.npu.rtp_write(@rtp_1_4, 0, 32)
      aiex.npu.rtp_write(@rtp_1_4, 1, 15)
      aiex.npu.rtp_write(@rtp_1_4, 2, 120)
      aiex.npu.rtp_write(@rtp_1_4, 3, 2)
      aiex.npu.rtp_write(@rtp_1_4, 4, 0)
      aiex.npu.rtp_write(@rtp_1_4, 5, 0)
      aiex.npu.rtp_write(@rtp_1_4, 6, 0)
      aiex.npu.rtp_write(@rtp_1_4, 7, 0)
      aiex.npu.rtp_write(@rtp_1_4, 8, 4)
      aiex.set_lock(%rtp_lock_1_4, 1)
      aiex.npu.rtp_write(@rtp_1_5, 0, 32)
      aiex.npu.rtp_write(@rtp_1_5, 1, 15)
      aiex.npu.rtp_write(@rtp_1_5, 2, 120)
      aiex.npu.rtp_write(@rtp_1_5, 3, 2)
      aiex.npu.rtp_write(@rtp_1_5, 4, 0)
      aiex.npu.rtp_write(@rtp_1_5, 5, 0)
      aiex.npu.rtp_write(@rtp_1_5, 6, 0)
      aiex.npu.rtp_write(@rtp_1_5, 7, 0)
      aiex.npu.rtp_write(@rtp_1_5, 8, 4)
      aiex.set_lock(%rtp_lock_1_5, 1)
      aiex.npu.rtp_write(@rtp_1_6, 0, 32)
      aiex.npu.rtp_write(@rtp_1_6, 1, 15)
      aiex.npu.rtp_write(@rtp_1_6, 2, 120)
      aiex.npu.rtp_write(@rtp_1_6, 3, 2)
      aiex.npu.rtp_write(@rtp_1_6, 4, 0)
      aiex.npu.rtp_write(@rtp_1_6, 5, 0)
      aiex.npu.rtp_write(@rtp_1_6, 6, 0)
      aiex.npu.rtp_write(@rtp_1_6, 7, 0)
      aiex.npu.rtp_write(@rtp_1_6, 8, 4)
      aiex.set_lock(%rtp_lock_1_6, 1)
      aiex.npu.rtp_write(@rtp_1_7, 0, 32)
      aiex.npu.rtp_write(@rtp_1_7, 1, 15)
      aiex.npu.rtp_write(@rtp_1_7, 2, 120)
      aiex.npu.rtp_write(@rtp_1_7, 3, 2)
      aiex.npu.rtp_write(@rtp_1_7, 4, 0)
      aiex.npu.rtp_write(@rtp_1_7, 5, 0)
      aiex.npu.rtp_write(@rtp_1_7, 6, 0)
      aiex.npu.rtp_write(@rtp_1_7, 7, 0)
      aiex.npu.rtp_write(@rtp_1_7, 8, 4)
      aiex.set_lock(%rtp_lock_1_7, 1)
      aiex.npu.rtp_write(@rtp_2_0, 0, 32)
      aiex.npu.rtp_write(@rtp_2_0, 1, 15)
      aiex.npu.rtp_write(@rtp_2_0, 2, 120)
      aiex.npu.rtp_write(@rtp_2_0, 3, 2)
      aiex.npu.rtp_write(@rtp_2_0, 4, 0)
      aiex.npu.rtp_write(@rtp_2_0, 5, 0)
      aiex.npu.rtp_write(@rtp_2_0, 6, 0)
      aiex.npu.rtp_write(@rtp_2_0, 7, 0)
      aiex.npu.rtp_write(@rtp_2_0, 8, 4)
      aiex.set_lock(%rtp_lock_2_0, 1)
      aiex.npu.rtp_write(@rtp_2_1, 0, 32)
      aiex.npu.rtp_write(@rtp_2_1, 1, 15)
      aiex.npu.rtp_write(@rtp_2_1, 2, 120)
      aiex.npu.rtp_write(@rtp_2_1, 3, 2)
      aiex.npu.rtp_write(@rtp_2_1, 4, 0)
      aiex.npu.rtp_write(@rtp_2_1, 5, 0)
      aiex.npu.rtp_write(@rtp_2_1, 6, 0)
      aiex.npu.rtp_write(@rtp_2_1, 7, 0)
      aiex.npu.rtp_write(@rtp_2_1, 8, 4)
      aiex.set_lock(%rtp_lock_2_1, 1)
      aiex.npu.rtp_write(@rtp_2_2, 0, 32)
      aiex.npu.rtp_write(@rtp_2_2, 1, 15)
      aiex.npu.rtp_write(@rtp_2_2, 2, 120)
      aiex.npu.rtp_write(@rtp_2_2, 3, 2)
      aiex.npu.rtp_write(@rtp_2_2, 4, 0)
      aiex.npu.rtp_write(@rtp_2_2, 5, 0)
      aiex.npu.rtp_write(@rtp_2_2, 6, 0)
      aiex.npu.rtp_write(@rtp_2_2, 7, 0)
      aiex.npu.rtp_write(@rtp_2_2, 8, 4)
      aiex.set_lock(%rtp_lock_2_2, 1)
      aiex.npu.rtp_write(@rtp_2_3, 0, 32)
      aiex.npu.rtp_write(@rtp_2_3, 1, 15)
      aiex.npu.rtp_write(@rtp_2_3, 2, 120)
      aiex.npu.rtp_write(@rtp_2_3, 3, 2)
      aiex.npu.rtp_write(@rtp_2_3, 4, 0)
      aiex.npu.rtp_write(@rtp_2_3, 5, 0)
      aiex.npu.rtp_write(@rtp_2_3, 6, 0)
      aiex.npu.rtp_write(@rtp_2_3, 7, 0)
      aiex.npu.rtp_write(@rtp_2_3, 8, 4)
      aiex.set_lock(%rtp_lock_2_3, 1)
      aiex.npu.rtp_write(@rtp_2_4, 0, 32)
      aiex.npu.rtp_write(@rtp_2_4, 1, 15)
      aiex.npu.rtp_write(@rtp_2_4, 2, 120)
      aiex.npu.rtp_write(@rtp_2_4, 3, 2)
      aiex.npu.rtp_write(@rtp_2_4, 4, 0)
      aiex.npu.rtp_write(@rtp_2_4, 5, 0)
      aiex.npu.rtp_write(@rtp_2_4, 6, 0)
      aiex.npu.rtp_write(@rtp_2_4, 7, 0)
      aiex.npu.rtp_write(@rtp_2_4, 8, 4)
      aiex.set_lock(%rtp_lock_2_4, 1)
      aiex.npu.rtp_write(@rtp_2_5, 0, 32)
      aiex.npu.rtp_write(@rtp_2_5, 1, 15)
      aiex.npu.rtp_write(@rtp_2_5, 2, 120)
      aiex.npu.rtp_write(@rtp_2_5, 3, 2)
      aiex.npu.rtp_write(@rtp_2_5, 4, 0)
      aiex.npu.rtp_write(@rtp_2_5, 5, 0)
      aiex.npu.rtp_write(@rtp_2_5, 6, 0)
      aiex.npu.rtp_write(@rtp_2_5, 7, 0)
      aiex.npu.rtp_write(@rtp_2_5, 8, 4)
      aiex.set_lock(%rtp_lock_2_5, 1)
      aiex.npu.rtp_write(@rtp_2_6, 0, 32)
      aiex.npu.rtp_write(@rtp_2_6, 1, 15)
      aiex.npu.rtp_write(@rtp_2_6, 2, 120)
      aiex.npu.rtp_write(@rtp_2_6, 3, 2)
      aiex.npu.rtp_write(@rtp_2_6, 4, 0)
      aiex.npu.rtp_write(@rtp_2_6, 5, 0)
      aiex.npu.rtp_write(@rtp_2_6, 6, 0)
      aiex.npu.rtp_write(@rtp_2_6, 7, 0)
      aiex.npu.rtp_write(@rtp_2_6, 8, 4)
      aiex.set_lock(%rtp_lock_2_6, 1)
      aiex.npu.rtp_write(@rtp_2_7, 0, 32)
      aiex.npu.rtp_write(@rtp_2_7, 1, 15)
      aiex.npu.rtp_write(@rtp_2_7, 2, 120)
      aiex.npu.rtp_write(@rtp_2_7, 3, 2)
      aiex.npu.rtp_write(@rtp_2_7, 4, 0)
      aiex.npu.rtp_write(@rtp_2_7, 5, 0)
      aiex.npu.rtp_write(@rtp_2_7, 6, 0)
      aiex.npu.rtp_write(@rtp_2_7, 7, 0)
      aiex.npu.rtp_write(@rtp_2_7, 8, 4)
      aiex.set_lock(%rtp_lock_2_7, 1)
      aiex.npu.rtp_write(@rtp_3_0, 0, 32)
      aiex.npu.rtp_write(@rtp_3_0, 1, 15)
      aiex.npu.rtp_write(@rtp_3_0, 2, 120)
      aiex.npu.rtp_write(@rtp_3_0, 3, 2)
      aiex.npu.rtp_write(@rtp_3_0, 4, 0)
      aiex.npu.rtp_write(@rtp_3_0, 5, 0)
      aiex.npu.rtp_write(@rtp_3_0, 6, 0)
      aiex.npu.rtp_write(@rtp_3_0, 7, 0)
      aiex.npu.rtp_write(@rtp_3_0, 8, 4)
      aiex.set_lock(%rtp_lock_3_0, 1)
      aiex.npu.rtp_write(@rtp_3_1, 0, 32)
      aiex.npu.rtp_write(@rtp_3_1, 1, 15)
      aiex.npu.rtp_write(@rtp_3_1, 2, 120)
      aiex.npu.rtp_write(@rtp_3_1, 3, 2)
      aiex.npu.rtp_write(@rtp_3_1, 4, 0)
      aiex.npu.rtp_write(@rtp_3_1, 5, 0)
      aiex.npu.rtp_write(@rtp_3_1, 6, 0)
      aiex.npu.rtp_write(@rtp_3_1, 7, 0)
      aiex.npu.rtp_write(@rtp_3_1, 8, 4)
      aiex.set_lock(%rtp_lock_3_1, 1)
      aiex.npu.rtp_write(@rtp_3_2, 0, 32)
      aiex.npu.rtp_write(@rtp_3_2, 1, 15)
      aiex.npu.rtp_write(@rtp_3_2, 2, 120)
      aiex.npu.rtp_write(@rtp_3_2, 3, 2)
      aiex.npu.rtp_write(@rtp_3_2, 4, 0)
      aiex.npu.rtp_write(@rtp_3_2, 5, 0)
      aiex.npu.rtp_write(@rtp_3_2, 6, 0)
      aiex.npu.rtp_write(@rtp_3_2, 7, 0)
      aiex.npu.rtp_write(@rtp_3_2, 8, 4)
      aiex.set_lock(%rtp_lock_3_2, 1)
      aiex.npu.rtp_write(@rtp_3_3, 0, 32)
      aiex.npu.rtp_write(@rtp_3_3, 1, 15)
      aiex.npu.rtp_write(@rtp_3_3, 2, 120)
      aiex.npu.rtp_write(@rtp_3_3, 3, 2)
      aiex.npu.rtp_write(@rtp_3_3, 4, 0)
      aiex.npu.rtp_write(@rtp_3_3, 5, 0)
      aiex.npu.rtp_write(@rtp_3_3, 6, 0)
      aiex.npu.rtp_write(@rtp_3_3, 7, 0)
      aiex.npu.rtp_write(@rtp_3_3, 8, 4)
      aiex.set_lock(%rtp_lock_3_3, 1)
      aiex.npu.rtp_write(@rtp_3_4, 0, 32)
      aiex.npu.rtp_write(@rtp_3_4, 1, 15)
      aiex.npu.rtp_write(@rtp_3_4, 2, 120)
      aiex.npu.rtp_write(@rtp_3_4, 3, 2)
      aiex.npu.rtp_write(@rtp_3_4, 4, 0)
      aiex.npu.rtp_write(@rtp_3_4, 5, 0)
      aiex.npu.rtp_write(@rtp_3_4, 6, 0)
      aiex.npu.rtp_write(@rtp_3_4, 7, 0)
      aiex.npu.rtp_write(@rtp_3_4, 8, 4)
      aiex.set_lock(%rtp_lock_3_4, 1)
      aiex.npu.rtp_write(@rtp_3_5, 0, 32)
      aiex.npu.rtp_write(@rtp_3_5, 1, 15)
      aiex.npu.rtp_write(@rtp_3_5, 2, 120)
      aiex.npu.rtp_write(@rtp_3_5, 3, 2)
      aiex.npu.rtp_write(@rtp_3_5, 4, 0)
      aiex.npu.rtp_write(@rtp_3_5, 5, 0)
      aiex.npu.rtp_write(@rtp_3_5, 6, 0)
      aiex.npu.rtp_write(@rtp_3_5, 7, 0)
      aiex.npu.rtp_write(@rtp_3_5, 8, 4)
      aiex.set_lock(%rtp_lock_3_5, 1)
      aiex.npu.rtp_write(@rtp_3_6, 0, 32)
      aiex.npu.rtp_write(@rtp_3_6, 1, 15)
      aiex.npu.rtp_write(@rtp_3_6, 2, 120)
      aiex.npu.rtp_write(@rtp_3_6, 3, 2)
      aiex.npu.rtp_write(@rtp_3_6, 4, 0)
      aiex.npu.rtp_write(@rtp_3_6, 5, 0)
      aiex.npu.rtp_write(@rtp_3_6, 6, 0)
      aiex.npu.rtp_write(@rtp_3_6, 7, 0)
      aiex.npu.rtp_write(@rtp_3_6, 8, 4)
      aiex.set_lock(%rtp_lock_3_6, 1)
      aiex.npu.rtp_write(@rtp_3_7, 0, 32)
      aiex.npu.rtp_write(@rtp_3_7, 1, 15)
      aiex.npu.rtp_write(@rtp_3_7, 2, 120)
      aiex.npu.rtp_write(@rtp_3_7, 3, 2)
      aiex.npu.rtp_write(@rtp_3_7, 4, 0)
      aiex.npu.rtp_write(@rtp_3_7, 5, 0)
      aiex.npu.rtp_write(@rtp_3_7, 6, 0)
      aiex.npu.rtp_write(@rtp_3_7, 7, 0)
      aiex.npu.rtp_write(@rtp_3_7, 8, 4)
      aiex.set_lock(%rtp_lock_3_7, 1)
      %0 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 0, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 196608, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 393216, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 589824, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 786432, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 983040, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 1179648, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 1376256, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 16384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 32768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 49152, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 65536, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 81920, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 98304, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 114688, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 1572864, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 1769472, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 1966080, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 2162688, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 2359296, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 2555904, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 2752512, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 2949120, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 131072, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 147456, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 163840, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 180224, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 196608, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 212992, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 229376, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 245760, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 3145728, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 3342336, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 3538944, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 3735552, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 3932160, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 4128768, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 4325376, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 4521984, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 262144, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 278528, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 294912, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 311296, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 327680, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 344064, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 360448, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 376832, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 4718592, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 4915200, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 5111808, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 5308416, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 5505024, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 5701632, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 5898240, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 6094848, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 393216, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 409600, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 425984, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 442368, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 458752, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 475136, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 491520, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 507904, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
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
      %68 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 6291456, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 6488064, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 6684672, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 6881280, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 7077888, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 7274496, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 7471104, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 7667712, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 524288, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 540672, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 557056, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 573440, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 589824, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 606208, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 622592, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 638976, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 7864320, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 8060928, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 8257536, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 8454144, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 8650752, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 8847360, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 9043968, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 9240576, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 655360, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 671744, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 688128, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 704512, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 720896, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 737280, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 753664, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 770048, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%101)
      aiex.dma_await_task(%43)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%47)
      aiex.dma_await_task(%48)
      aiex.dma_await_task(%49)
      aiex.dma_await_task(%50)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%42)
      aiex.dma_await_task(%60)
      aiex.dma_await_task(%61)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%63)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%65)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%67)
      aiex.dma_free_task(%51)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%59)
      %102 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 9437184, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 9633792, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 9830400, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 10027008, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 10223616, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 10420224, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 10616832, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 10813440, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 786432, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 802816, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 819200, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 835584, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 851968, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 868352, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 884736, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 901120, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 11010048, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 11206656, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 11403264, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 11599872, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 11796480, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 11993088, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 12189696, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 12386304, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 917504, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 933888, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 950272, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 966656, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 983040, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 999424, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1015808, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1032192, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%135)
      aiex.dma_await_task(%77)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%83)
      aiex.dma_await_task(%84)
      aiex.dma_free_task(%68)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%71)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%74)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%76)
      aiex.dma_await_task(%94)
      aiex.dma_await_task(%95)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%97)
      aiex.dma_await_task(%98)
      aiex.dma_await_task(%99)
      aiex.dma_await_task(%100)
      aiex.dma_await_task(%101)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%92)
      aiex.dma_free_task(%93)
      %136 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 12582912, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 12779520, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 12976128, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 13172736, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 13369344, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 13565952, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 13762560, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%143)
      %144 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 13959168, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1048576, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1064960, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1081344, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1097728, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1114112, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1130496, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1146880, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1163264, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 14155776, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 14352384, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 14548992, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 14745600, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 14942208, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 15138816, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%159)
      %160 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 15335424, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 15532032, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1179648, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%162)
      %163 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1196032, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1212416, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1228800, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%165)
      %166 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1245184, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1261568, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1277952, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%168)
      %169 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1294336, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%169)
      aiex.dma_await_task(%111)
      aiex.dma_await_task(%112)
      aiex.dma_await_task(%113)
      aiex.dma_await_task(%114)
      aiex.dma_await_task(%115)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%118)
      aiex.dma_free_task(%102)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%106)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%110)
      aiex.dma_await_task(%128)
      aiex.dma_await_task(%129)
      aiex.dma_await_task(%130)
      aiex.dma_await_task(%131)
      aiex.dma_await_task(%132)
      aiex.dma_await_task(%133)
      aiex.dma_await_task(%134)
      aiex.dma_await_task(%135)
      aiex.dma_free_task(%119)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%122)
      aiex.dma_free_task(%123)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%125)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%127)
      %170 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 15728640, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 15925248, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%172)
      %173 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 16121856, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 16318464, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 16515072, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%175)
      %176 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 16711680, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 16908288, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 17104896, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%178)
      %179 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1310720, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1327104, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1343488, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%181)
      %182 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1359872, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%182)
      %183 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1376256, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%183)
      %184 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1392640, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1409024, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%185)
      %186 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1425408, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%186)
      %187 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 17301504, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%188)
      %189 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 17498112, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 17694720, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 17891328, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%191)
      %192 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 18087936, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%192)
      %193 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 18284544, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%193)
      %194 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 18481152, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%194)
      %195 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 18677760, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%195)
      %196 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1441792, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%196)
      %197 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1458176, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%197)
      %198 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1474560, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%198)
      %199 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1490944, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%199)
      %200 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1507328, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%200)
      %201 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1523712, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%201)
      %202 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1540096, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%202)
      %203 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1556480, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%203)
      aiex.dma_await_task(%145)
      aiex.dma_await_task(%146)
      aiex.dma_await_task(%147)
      aiex.dma_await_task(%148)
      aiex.dma_await_task(%149)
      aiex.dma_await_task(%150)
      aiex.dma_await_task(%151)
      aiex.dma_await_task(%152)
      aiex.dma_free_task(%136)
      aiex.dma_free_task(%137)
      aiex.dma_free_task(%138)
      aiex.dma_free_task(%139)
      aiex.dma_free_task(%140)
      aiex.dma_free_task(%141)
      aiex.dma_free_task(%142)
      aiex.dma_free_task(%143)
      aiex.dma_free_task(%144)
      aiex.dma_await_task(%162)
      aiex.dma_await_task(%163)
      aiex.dma_await_task(%164)
      aiex.dma_await_task(%165)
      aiex.dma_await_task(%166)
      aiex.dma_await_task(%167)
      aiex.dma_await_task(%168)
      aiex.dma_await_task(%169)
      aiex.dma_free_task(%153)
      aiex.dma_free_task(%154)
      aiex.dma_free_task(%155)
      aiex.dma_free_task(%156)
      aiex.dma_free_task(%157)
      aiex.dma_free_task(%158)
      aiex.dma_free_task(%159)
      aiex.dma_free_task(%160)
      aiex.dma_free_task(%161)
      %204 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%204)
      %205 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 18874368, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%205)
      %206 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 19070976, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%206)
      %207 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 19267584, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%207)
      %208 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 19464192, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%208)
      %209 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 19660800, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%209)
      %210 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 19857408, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%210)
      %211 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 20054016, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%211)
      %212 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 20250624, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%212)
      %213 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1572864, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%213)
      %214 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1589248, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%214)
      %215 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1605632, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%215)
      %216 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1622016, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%216)
      %217 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1638400, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%217)
      %218 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1654784, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%218)
      %219 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1671168, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%219)
      %220 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1687552, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%220)
      %221 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%221)
      %222 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 20447232, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%222)
      %223 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 20643840, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%223)
      %224 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 20840448, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%224)
      %225 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 21037056, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%225)
      %226 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 21233664, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%226)
      %227 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 21430272, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%227)
      %228 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 21626880, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%228)
      %229 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 21823488, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%229)
      %230 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1703936, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%230)
      %231 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1720320, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%231)
      %232 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1736704, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%232)
      %233 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1753088, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%233)
      %234 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1769472, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%234)
      %235 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1785856, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%235)
      %236 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1802240, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%236)
      %237 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1818624, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%237)
      aiex.dma_await_task(%179)
      aiex.dma_await_task(%180)
      aiex.dma_await_task(%181)
      aiex.dma_await_task(%182)
      aiex.dma_await_task(%183)
      aiex.dma_await_task(%184)
      aiex.dma_await_task(%185)
      aiex.dma_await_task(%186)
      aiex.dma_free_task(%170)
      aiex.dma_free_task(%171)
      aiex.dma_free_task(%172)
      aiex.dma_free_task(%173)
      aiex.dma_free_task(%174)
      aiex.dma_free_task(%175)
      aiex.dma_free_task(%176)
      aiex.dma_free_task(%177)
      aiex.dma_free_task(%178)
      aiex.dma_await_task(%196)
      aiex.dma_await_task(%197)
      aiex.dma_await_task(%198)
      aiex.dma_await_task(%199)
      aiex.dma_await_task(%200)
      aiex.dma_await_task(%201)
      aiex.dma_await_task(%202)
      aiex.dma_await_task(%203)
      aiex.dma_free_task(%187)
      aiex.dma_free_task(%188)
      aiex.dma_free_task(%189)
      aiex.dma_free_task(%190)
      aiex.dma_free_task(%191)
      aiex.dma_free_task(%192)
      aiex.dma_free_task(%193)
      aiex.dma_free_task(%194)
      aiex.dma_free_task(%195)
      %238 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%238)
      %239 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 22020096, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%239)
      %240 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 22216704, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%240)
      %241 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 22413312, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%241)
      %242 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 22609920, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%242)
      %243 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 22806528, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%243)
      %244 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 23003136, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%244)
      %245 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 23199744, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%245)
      %246 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 23396352, 196608, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 196608, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%246)
      %247 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1835008, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%247)
      %248 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1851392, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%248)
      %249 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1867776, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%249)
      %250 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1884160, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%250)
      %251 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1900544, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%251)
      %252 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1916928, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%252)
      %253 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1933312, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%253)
      %254 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1949696, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%254)
      aiex.dma_await_task(%213)
      aiex.dma_await_task(%214)
      aiex.dma_await_task(%215)
      aiex.dma_await_task(%216)
      aiex.dma_await_task(%217)
      aiex.dma_await_task(%218)
      aiex.dma_await_task(%219)
      aiex.dma_await_task(%220)
      aiex.dma_free_task(%204)
      aiex.dma_free_task(%205)
      aiex.dma_free_task(%206)
      aiex.dma_free_task(%207)
      aiex.dma_free_task(%208)
      aiex.dma_free_task(%209)
      aiex.dma_free_task(%210)
      aiex.dma_free_task(%211)
      aiex.dma_free_task(%212)
      aiex.dma_await_task(%230)
      aiex.dma_await_task(%231)
      aiex.dma_await_task(%232)
      aiex.dma_await_task(%233)
      aiex.dma_await_task(%234)
      aiex.dma_await_task(%235)
      aiex.dma_await_task(%236)
      aiex.dma_await_task(%237)
      aiex.dma_free_task(%221)
      aiex.dma_free_task(%222)
      aiex.dma_free_task(%223)
      aiex.dma_free_task(%224)
      aiex.dma_free_task(%225)
      aiex.dma_free_task(%226)
      aiex.dma_free_task(%227)
      aiex.dma_free_task(%228)
      aiex.dma_free_task(%229)
      aiex.dma_await_task(%247)
      aiex.dma_await_task(%248)
      aiex.dma_await_task(%249)
      aiex.dma_await_task(%250)
      aiex.dma_await_task(%251)
      aiex.dma_await_task(%252)
      aiex.dma_await_task(%253)
      aiex.dma_await_task(%254)
      aiex.dma_free_task(%238)
      aiex.dma_free_task(%239)
      aiex.dma_free_task(%240)
      aiex.dma_free_task(%241)
      aiex.dma_free_task(%242)
      aiex.dma_free_task(%243)
      aiex.dma_free_task(%244)
      aiex.dma_free_task(%245)
      aiex.dma_free_task(%246)
      %255 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 0, 4608, [<size = 1, stride = 0>, <size = 120, stride = 16384>, <size = 4, stride = 288>, <size = 4, stride = 4096>, <size = 288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%255)
      %256 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 23592960, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%256)
      %257 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 24330240, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%257)
      %258 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 25067520, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%258)
      %259 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 25804800, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%259)
      %260 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 26542080, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%260)
      %261 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 27279360, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%261)
      %262 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 28016640, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%262)
      %263 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 28753920, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%263)
      %264 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1966080, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%264)
      %265 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1982464, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%265)
      %266 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 1998848, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%266)
      %267 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2015232, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%267)
      %268 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2031616, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%268)
      %269 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2048000, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%269)
      %270 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2064384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%270)
      %271 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2080768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%271)
      %272 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 0, 4608, [<size = 1, stride = 0>, <size = 120, stride = 16384>, <size = 4, stride = 288>, <size = 4, stride = 4096>, <size = 288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%272)
      %273 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 29491200, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%273)
      %274 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 30228480, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%274)
      %275 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 30965760, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%275)
      %276 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 31703040, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%276)
      %277 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 32440320, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%277)
      %278 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 33177600, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%278)
      %279 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 33914880, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%279)
      %280 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<35389440xui8>, 34652160, 737280, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 737280, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%280)
      %281 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2097152, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%281)
      %282 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2113536, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%282)
      %283 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2129920, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%283)
      %284 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2146304, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%284)
      %285 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2162688, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%285)
      %286 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2179072, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%286)
      %287 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2195456, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%287)
      %288 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2228224xui8>, 2211840, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%288)
      aiex.dma_await_task(%264)
      aiex.dma_await_task(%265)
      aiex.dma_await_task(%266)
      aiex.dma_await_task(%267)
      aiex.dma_await_task(%268)
      aiex.dma_await_task(%269)
      aiex.dma_await_task(%270)
      aiex.dma_await_task(%271)
      aiex.dma_free_task(%255)
      aiex.dma_free_task(%256)
      aiex.dma_free_task(%257)
      aiex.dma_free_task(%258)
      aiex.dma_free_task(%259)
      aiex.dma_free_task(%260)
      aiex.dma_free_task(%261)
      aiex.dma_free_task(%262)
      aiex.dma_free_task(%263)
      aiex.dma_await_task(%281)
      aiex.dma_await_task(%282)
      aiex.dma_await_task(%283)
      aiex.dma_await_task(%284)
      aiex.dma_await_task(%285)
      aiex.dma_await_task(%286)
      aiex.dma_await_task(%287)
      aiex.dma_await_task(%288)
      aiex.dma_free_task(%272)
      aiex.dma_free_task(%273)
      aiex.dma_free_task(%274)
      aiex.dma_free_task(%275)
      aiex.dma_free_task(%276)
      aiex.dma_free_task(%277)
      aiex.dma_free_task(%278)
      aiex.dma_free_task(%279)
      aiex.dma_free_task(%280)
    }
    aie.shim_dma_allocation @A_L3L2_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb15)
    ^bb7:  // 2 preds: ^bb6, ^bb14
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_0_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 25 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%B_L3L2_0_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%B_L3L2_0_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 27 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%B_L3L2_0_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%B_L3L2_0_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%B_L3L2_0_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 31 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb7
    ^bb15:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 33 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%B_L3L2_0_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L3L2_0_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(MM2S, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%B_L3L2_0_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_L3L2_0_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 35 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(MM2S, 4, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%B_L3L2_0_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%B_L3L2_0_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %7 = aie.dma_start(S2MM, 2, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%C_L2L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %8 = aie.dma_start(S2MM, 3, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%C_L2L3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 36 : i32, next_bd_id = 37 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 37 : i32, next_bd_id = 36 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %9 = aie.dma_start(S2MM, 4, ^bb34, ^bb36)
    ^bb34:  // 2 preds: ^bb33, ^bb35
      aie.use_lock(%C_L2L3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%C_L2L3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb34
    ^bb36:  // pred: ^bb33
      %10 = aie.dma_start(S2MM, 5, ^bb37, ^bb39)
    ^bb37:  // 2 preds: ^bb36, ^bb38
      aie.use_lock(%C_L2L3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 38 : i32, next_bd_id = 39 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb38
    ^bb38:  // pred: ^bb37
      aie.use_lock(%C_L2L3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 39 : i32, next_bd_id = 38 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb37
    ^bb39:  // pred: ^bb36
      %11 = aie.dma_start(MM2S, 5, ^bb40, ^bb48)
    ^bb40:  // 2 preds: ^bb39, ^bb47
      aie.use_lock(%C_L2L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 40 : i32, next_bd_id = 41 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb41
    ^bb41:  // pred: ^bb40
      aie.use_lock(%C_L2L3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 41 : i32, next_bd_id = 42 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb42
    ^bb42:  // pred: ^bb41
      aie.use_lock(%C_L2L3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 42 : i32, next_bd_id = 43 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb43
    ^bb43:  // pred: ^bb42
      aie.use_lock(%C_L2L3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 43 : i32, next_bd_id = 44 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb44
    ^bb44:  // pred: ^bb43
      aie.use_lock(%C_L2L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 44 : i32, next_bd_id = 45 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb45
    ^bb45:  // pred: ^bb44
      aie.use_lock(%C_L2L3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 45 : i32, next_bd_id = 46 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb46
    ^bb46:  // pred: ^bb45
      aie.use_lock(%C_L2L3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 46 : i32, next_bd_id = 47 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb47
    ^bb47:  // pred: ^bb46
      aie.use_lock(%C_L2L3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 47 : i32, next_bd_id = 40 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb40
    ^bb48:  // pred: ^bb39
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_0_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_0_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_0_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_0_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_0_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_0_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_0_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_0_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_0_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_0_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_0_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_0_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_4_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_4_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_0_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_0_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_0_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_5_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_5_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_0_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_0_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_0_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_6_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_6_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_0_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_0_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_0_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_7_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_7_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_0_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_0_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_0_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_8_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_8_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_8_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_8_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_8_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_8_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_1_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_1_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_1_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_9_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_9_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_9_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_9_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_9_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_9_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_1_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_1_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_1_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_10_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_10_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_10_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_10_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_10_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_10_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_1_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_1_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_1_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_11_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_11_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_11_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_11_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_11_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_11_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_1_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_1_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_1_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_3 = aie.mem(%tile_4_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_12_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_12_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_12_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_12_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_12_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_12_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_4_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_1_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_4_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_4_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_1_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_4_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_1_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_3 = aie.mem(%tile_5_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_13_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_13_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_13_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_13_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_13_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_13_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_5_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_1_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_5_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_5_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_1_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_5_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_1_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_3 = aie.mem(%tile_6_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_14_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_14_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_14_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_14_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_14_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_14_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_6_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_1_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_6_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_6_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_1_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_6_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_1_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_3 = aie.mem(%tile_7_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_15_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_15_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_15_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_15_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_15_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_15_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_7_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_1_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_7_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_7_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_1_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_7_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_1_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_7_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_16_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_16_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_16_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_16_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_16_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_16_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_2_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_2_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_2_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_17_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_17_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_17_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_17_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_17_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_17_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_2_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_2_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_2_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_18_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_18_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_18_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_18_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_18_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_18_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_2_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_2_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_2_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_19_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_19_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_19_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_19_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_19_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_19_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_2_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_2_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_2_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_4 = aie.mem(%tile_4_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_20_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_20_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_20_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_20_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_20_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_20_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_4_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_2_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_4_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_4_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_2_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_4_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_2_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_4 = aie.mem(%tile_5_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_21_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_21_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_21_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_21_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_21_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_21_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_5_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_2_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_5_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_5_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_2_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_5_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_2_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_4 = aie.mem(%tile_6_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_22_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_22_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_22_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_22_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_22_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_22_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_6_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_2_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_6_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_6_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_2_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_6_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_2_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_4 = aie.mem(%tile_7_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_23_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_23_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_23_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_23_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_23_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_23_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_7_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_2_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_7_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_7_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_2_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_7_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_2_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_7_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_24_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_24_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_24_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_24_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_24_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_24_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_3_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_3_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_3_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_25_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_25_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_25_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_25_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_25_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_25_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_3_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_3_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_3_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_26_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_26_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_26_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_26_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_26_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_26_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_3_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_3_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_3_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_27_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_27_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_27_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_27_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_27_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_27_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_3_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_3_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_3_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_5 = aie.mem(%tile_4_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_28_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_28_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_28_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_28_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_28_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_28_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_4_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_3_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_4_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_4_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_3_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_4_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_3_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_5 = aie.mem(%tile_5_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_29_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_29_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_29_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_29_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_29_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_29_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_5_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_3_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_5_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_5_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_3_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_5_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_3_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_5 = aie.mem(%tile_6_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_30_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_30_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_30_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_30_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_30_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_30_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_6_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_3_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_6_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_6_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_3_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_6_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_3_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_5 = aie.mem(%tile_7_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_31_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_31_cons_buff_0 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_31_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_31_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_31_cons_buff_1 : memref<16x8x!aiex.bfp<"v8bfp16ebs8">>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_31_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_7_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_3_cons_buff_0 : memref<1536xui8>, 0, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_7_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_7_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_3_cons_buff_1 : memref<1536xui8>, 0, 1536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_7_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_3_buff_0 : memref<4096xui8>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_7_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @C_L2L3_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @B_L3L2_1_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_1_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%B_L3L2_1_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L2_1_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L3L2_1_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_L3L2_1_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_1_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%B_L3L2_1_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%B_L3L2_1_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%B_L3L2_1_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_L3L2_1_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%B_L3L2_1_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L3L2_1_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(S2MM, 1, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 2, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%C_L2L3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %7 = aie.dma_start(S2MM, 3, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%C_L2L3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %8 = aie.dma_start(S2MM, 4, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%C_L2L3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 15 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %9 = aie.dma_start(MM2S, 4, ^bb34, ^bb42)
    ^bb34:  // 2 preds: ^bb33, ^bb41
      aie.use_lock(%C_L2L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%C_L2L3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb36
    ^bb36:  // pred: ^bb35
      aie.use_lock(%C_L2L3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb37
    ^bb37:  // pred: ^bb36
      aie.use_lock(%C_L2L3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb38
    ^bb38:  // pred: ^bb37
      aie.use_lock(%C_L2L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb39
    ^bb39:  // pred: ^bb38
      aie.use_lock(%C_L2L3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb40
    ^bb40:  // pred: ^bb39
      aie.use_lock(%C_L2L3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb41
    ^bb41:  // pred: ^bb40
      aie.use_lock(%C_L2L3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 23 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb34
    ^bb42:  // pred: ^bb33
      aie.end
    }
    aie.shim_dma_allocation @C_L2L3_1_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @B_L3L2_2_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_2_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%B_L3L2_2_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L2_2_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L3L2_2_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_L3L2_2_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_2_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%B_L3L2_2_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%B_L3L2_2_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%B_L3L2_2_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_L3L2_2_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%B_L3L2_2_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L3L2_2_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(S2MM, 1, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 2, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%C_L2L3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %7 = aie.dma_start(S2MM, 3, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%C_L2L3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %8 = aie.dma_start(S2MM, 4, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%C_L2L3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 15 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %9 = aie.dma_start(MM2S, 4, ^bb34, ^bb42)
    ^bb34:  // 2 preds: ^bb33, ^bb41
      aie.use_lock(%C_L2L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%C_L2L3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb36
    ^bb36:  // pred: ^bb35
      aie.use_lock(%C_L2L3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb37
    ^bb37:  // pred: ^bb36
      aie.use_lock(%C_L2L3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb38
    ^bb38:  // pred: ^bb37
      aie.use_lock(%C_L2L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb39
    ^bb39:  // pred: ^bb38
      aie.use_lock(%C_L2L3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb40
    ^bb40:  // pred: ^bb39
      aie.use_lock(%C_L2L3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb41
    ^bb41:  // pred: ^bb40
      aie.use_lock(%C_L2L3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 23 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb34
    ^bb42:  // pred: ^bb33
      aie.end
    }
    aie.shim_dma_allocation @C_L2L3_2_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @B_L3L2_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_3_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%B_L3L2_3_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L2_3_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L3L2_3_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_L3L2_3_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_3_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%B_L3L2_3_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%B_L3L2_3_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%B_L3L2_3_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_L3L2_3_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%B_L3L2_3_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L3L2_3_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(S2MM, 1, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 2, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%C_L2L3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %7 = aie.dma_start(S2MM, 3, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%C_L2L3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %8 = aie.dma_start(S2MM, 4, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%C_L2L3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 15 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %9 = aie.dma_start(MM2S, 4, ^bb34, ^bb42)
    ^bb34:  // 2 preds: ^bb33, ^bb41
      aie.use_lock(%C_L2L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%C_L2L3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb36
    ^bb36:  // pred: ^bb35
      aie.use_lock(%C_L2L3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb37
    ^bb37:  // pred: ^bb36
      aie.use_lock(%C_L2L3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb38
    ^bb38:  // pred: ^bb37
      aie.use_lock(%C_L2L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb39
    ^bb39:  // pred: ^bb38
      aie.use_lock(%C_L2L3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb40
    ^bb40:  // pred: ^bb39
      aie.use_lock(%C_L2L3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb41
    ^bb41:  // pred: ^bb40
      aie.use_lock(%C_L2L3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 23 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb34
    ^bb42:  // pred: ^bb33
      aie.end
    }
    aie.shim_dma_allocation @C_L2L3_3_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @B_L3L2_4_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_4_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%B_L3L2_4_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L2_4_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L3L2_4_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_L3L2_4_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_4_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%B_L3L2_4_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%B_L3L2_4_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%B_L3L2_4_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_L3L2_4_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%B_L3L2_4_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L3L2_4_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(S2MM, 1, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_4_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_4_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 2, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%C_L2L3_4_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_1, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_4_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %7 = aie.dma_start(S2MM, 3, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%C_L2L3_4_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_2, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_4_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %8 = aie.dma_start(S2MM, 4, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%C_L2L3_4_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_3, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_4_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 15 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_3, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %9 = aie.dma_start(MM2S, 4, ^bb34, ^bb42)
    ^bb34:  // 2 preds: ^bb33, ^bb41
      aie.use_lock(%C_L2L3_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%C_L2L3_4_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_1, Release, 1)
      aie.next_bd ^bb36
    ^bb36:  // pred: ^bb35
      aie.use_lock(%C_L2L3_4_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_2, Release, 1)
      aie.next_bd ^bb37
    ^bb37:  // pred: ^bb36
      aie.use_lock(%C_L2L3_4_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_3, Release, 1)
      aie.next_bd ^bb38
    ^bb38:  // pred: ^bb37
      aie.use_lock(%C_L2L3_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb39
    ^bb39:  // pred: ^bb38
      aie.use_lock(%C_L2L3_4_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_1, Release, 1)
      aie.next_bd ^bb40
    ^bb40:  // pred: ^bb39
      aie.use_lock(%C_L2L3_4_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_2, Release, 1)
      aie.next_bd ^bb41
    ^bb41:  // pred: ^bb40
      aie.use_lock(%C_L2L3_4_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 23 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_3, Release, 1)
      aie.next_bd ^bb34
    ^bb42:  // pred: ^bb33
      aie.end
    }
    aie.shim_dma_allocation @C_L2L3_4_shim_alloc(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @B_L3L2_5_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_5_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%B_L3L2_5_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L2_5_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L3L2_5_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_L3L2_5_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_5_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%B_L3L2_5_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%B_L3L2_5_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%B_L3L2_5_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_L3L2_5_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%B_L3L2_5_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L3L2_5_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(S2MM, 1, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_5_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_5_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 2, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%C_L2L3_5_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_1, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_5_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %7 = aie.dma_start(S2MM, 3, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%C_L2L3_5_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_2, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_5_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %8 = aie.dma_start(S2MM, 4, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%C_L2L3_5_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_3, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_5_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 15 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_3, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %9 = aie.dma_start(MM2S, 4, ^bb34, ^bb42)
    ^bb34:  // 2 preds: ^bb33, ^bb41
      aie.use_lock(%C_L2L3_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%C_L2L3_5_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_1, Release, 1)
      aie.next_bd ^bb36
    ^bb36:  // pred: ^bb35
      aie.use_lock(%C_L2L3_5_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_2, Release, 1)
      aie.next_bd ^bb37
    ^bb37:  // pred: ^bb36
      aie.use_lock(%C_L2L3_5_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_3, Release, 1)
      aie.next_bd ^bb38
    ^bb38:  // pred: ^bb37
      aie.use_lock(%C_L2L3_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb39
    ^bb39:  // pred: ^bb38
      aie.use_lock(%C_L2L3_5_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_1, Release, 1)
      aie.next_bd ^bb40
    ^bb40:  // pred: ^bb39
      aie.use_lock(%C_L2L3_5_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_2, Release, 1)
      aie.next_bd ^bb41
    ^bb41:  // pred: ^bb40
      aie.use_lock(%C_L2L3_5_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 23 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_3, Release, 1)
      aie.next_bd ^bb34
    ^bb42:  // pred: ^bb33
      aie.end
    }
    aie.shim_dma_allocation @C_L2L3_5_shim_alloc(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @B_L3L2_6_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_6_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%B_L3L2_6_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L2_6_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L3L2_6_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_L3L2_6_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_6_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%B_L3L2_6_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%B_L3L2_6_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%B_L3L2_6_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_L3L2_6_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%B_L3L2_6_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L3L2_6_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(S2MM, 1, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_6_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_6_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 2, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%C_L2L3_6_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_1, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_6_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %7 = aie.dma_start(S2MM, 3, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%C_L2L3_6_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_2, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_6_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %8 = aie.dma_start(S2MM, 4, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%C_L2L3_6_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_3, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_6_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 15 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_3, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %9 = aie.dma_start(MM2S, 4, ^bb34, ^bb42)
    ^bb34:  // 2 preds: ^bb33, ^bb41
      aie.use_lock(%C_L2L3_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%C_L2L3_6_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_1, Release, 1)
      aie.next_bd ^bb36
    ^bb36:  // pred: ^bb35
      aie.use_lock(%C_L2L3_6_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_2, Release, 1)
      aie.next_bd ^bb37
    ^bb37:  // pred: ^bb36
      aie.use_lock(%C_L2L3_6_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_3, Release, 1)
      aie.next_bd ^bb38
    ^bb38:  // pred: ^bb37
      aie.use_lock(%C_L2L3_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb39
    ^bb39:  // pred: ^bb38
      aie.use_lock(%C_L2L3_6_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_1, Release, 1)
      aie.next_bd ^bb40
    ^bb40:  // pred: ^bb39
      aie.use_lock(%C_L2L3_6_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_2, Release, 1)
      aie.next_bd ^bb41
    ^bb41:  // pred: ^bb40
      aie.use_lock(%C_L2L3_6_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 23 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_3, Release, 1)
      aie.next_bd ^bb34
    ^bb42:  // pred: ^bb33
      aie.end
    }
    aie.shim_dma_allocation @C_L2L3_6_shim_alloc(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @B_L3L2_7_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_7_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%B_L3L2_7_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L2_7_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L3L2_7_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_L3L2_7_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_7_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<6144xui8>, 0, 1536) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<6144xui8>, 0, 1536) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%B_L3L2_7_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<6144xui8>, 1536, 1536) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%B_L3L2_7_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<6144xui8>, 1536, 1536) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%B_L3L2_7_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<6144xui8>, 3072, 1536) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_L3L2_7_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<6144xui8>, 3072, 1536) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%B_L3L2_7_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<6144xui8>, 4608, 1536) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L3L2_7_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<6144xui8>, 4608, 1536) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(S2MM, 1, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_7_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_7_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 2, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%C_L2L3_7_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_1, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_7_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %7 = aie.dma_start(S2MM, 3, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%C_L2L3_7_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_2, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_7_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %8 = aie.dma_start(S2MM, 4, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%C_L2L3_7_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_3, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_7_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 15 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_3, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %9 = aie.dma_start(MM2S, 4, ^bb34, ^bb42)
    ^bb34:  // 2 preds: ^bb33, ^bb41
      aie.use_lock(%C_L2L3_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<16384xui8>, 0, 4096) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%C_L2L3_7_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<16384xui8>, 4096, 4096) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_1, Release, 1)
      aie.next_bd ^bb36
    ^bb36:  // pred: ^bb35
      aie.use_lock(%C_L2L3_7_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<16384xui8>, 8192, 4096) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_2, Release, 1)
      aie.next_bd ^bb37
    ^bb37:  // pred: ^bb36
      aie.use_lock(%C_L2L3_7_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<16384xui8>, 12288, 4096) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_3, Release, 1)
      aie.next_bd ^bb38
    ^bb38:  // pred: ^bb37
      aie.use_lock(%C_L2L3_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<16384xui8>, 0, 4096) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb39
    ^bb39:  // pred: ^bb38
      aie.use_lock(%C_L2L3_7_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<16384xui8>, 4096, 4096) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_1, Release, 1)
      aie.next_bd ^bb40
    ^bb40:  // pred: ^bb39
      aie.use_lock(%C_L2L3_7_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<16384xui8>, 8192, 4096) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_2, Release, 1)
      aie.next_bd ^bb41
    ^bb41:  // pred: ^bb40
      aie.use_lock(%C_L2L3_7_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<16384xui8>, 12288, 4096) {bd_id = 23 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_3, Release, 1)
      aie.next_bd ^bb34
    ^bb42:  // pred: ^bb33
      aie.end
    }
    aie.shim_dma_allocation @C_L2L3_7_shim_alloc(%shim_noc_tile_7_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
      aie.connect<DMA : 4, North : 2>
      aie.connect<North : 1, DMA : 2>
      aie.connect<North : 2, DMA : 3>
      aie.connect<North : 3, DMA : 4>
      aie.connect<North : 0, DMA : 5>
      aie.connect<DMA : 5, South : 2>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, East : 2>
      aie.connect<South : 1, North : 1>
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<South : 0, North : 5>
      aie.connect<South : 3, North : 4>
      aie.connect<South : 2, North : 2>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 3, South : 2>
      aie.connect<North : 1, South : 3>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 1, North : 0>
      aie.connect<South : 1, East : 0>
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<South : 4, North : 2>
      aie.connect<South : 2, North : 4>
      aie.connect<DMA : 0, South : 3>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 2, DMA : 1>
      aie.connect<South : 4, North : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<South : 0, East : 0>
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<West : 2, East : 1>
      aie.connect<West : 2, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 0, North : 2>
      aie.connect<South : 3, North : 0>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 0, South : 3>
      aie.connect<North : 2, South : 2>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<West : 0, North : 0>
      aie.connect<West : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 2, North : 3>
      aie.connect<South : 0, North : 5>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 2>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<South : 5, North : 3>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<West : 0, East : 3>
      aie.connect<West : 0, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<West : 1, East : 2>
      aie.connect<West : 1, North : 0>
      aie.connect<West : 1, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 0, North : 2>
      aie.connect<South : 3, North : 3>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 2, South : 3>
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<South : 0, North : 4>
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 2, North : 3>
      aie.connect<South : 3, North : 2>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 2>
    }
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<South : 4, East : 1>
      aie.connect<South : 4, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<South : 2, North : 0>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<West : 3, East : 2>
      aie.connect<West : 3, DMA : 0>
      aie.connect<South : 0, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<West : 2, East : 1>
      aie.connect<West : 2, North : 5>
      aie.connect<West : 2, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 0, North : 2>
      aie.connect<South : 3, North : 0>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, South : 2>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 2, North : 3>
      aie.connect<South : 0, North : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 3>
      aie.connect<North : 3, South : 0>
    }
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<West : 1, East : 3>
      aie.connect<West : 1, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<South : 1, North : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %switchbox_3_5 = aie.switchbox(%tile_3_5) {
      aie.connect<West : 2, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      aie.connect<West : 1, East : 0>
      aie.connect<West : 1, North : 5>
      aie.connect<West : 1, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 0, North : 2>
      aie.connect<South : 3, North : 0>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 2, South : 3>
    }
    %switchbox_4_3 = aie.switchbox(%tile_4_3) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 2, North : 3>
      aie.connect<South : 0, North : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 2>
    }
    %switchbox_4_4 = aie.switchbox(%tile_4_4) {
      aie.connect<West : 3, North : 1>
      aie.connect<West : 3, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<South : 1, North : 0>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %switchbox_4_5 = aie.switchbox(%tile_4_5) {
      aie.connect<South : 1, East : 3>
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 0, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
      aie.connect<West : 0, East : 2>
      aie.connect<West : 0, North : 3>
      aie.connect<West : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 0, North : 2>
      aie.connect<South : 3, North : 0>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 2, South : 3>
    }
    %switchbox_5_3 = aie.switchbox(%tile_5_3) {
      aie.connect<South : 3, North : 2>
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 2, North : 3>
      aie.connect<South : 0, North : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 2>
    }
    %switchbox_5_4 = aie.switchbox(%tile_5_4) {
      aie.connect<South : 2, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<South : 1, North : 3>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %switchbox_5_5 = aie.switchbox(%tile_5_5) {
      aie.connect<West : 3, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
      aie.connect<West : 2, North : 5>
      aie.connect<West : 2, East : 0>
      aie.connect<West : 2, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 0, North : 2>
      aie.connect<South : 3, North : 0>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 3>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 2, South : 0>
    }
    %switchbox_6_3 = aie.switchbox(%tile_6_3) {
      aie.connect<South : 5, North : 4>
      aie.connect<South : 5, East : 3>
      aie.connect<South : 5, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 2, North : 3>
      aie.connect<South : 0, North : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 2>
    }
    %switchbox_6_4 = aie.switchbox(%tile_6_4) {
      aie.connect<South : 4, North : 1>
      aie.connect<South : 4, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<South : 1, North : 3>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %switchbox_6_5 = aie.switchbox(%tile_6_5) {
      aie.connect<South : 1, East : 1>
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
      aie.connect<West : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 5, North : 4>
      aie.connect<South : 0, North : 1>
      aie.connect<South : 3, North : 2>
      aie.connect<DMA : 0, South : 3>
      aie.connect<North : 3, South : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 0>
    }
    %switchbox_7_3 = aie.switchbox(%tile_7_3) {
      aie.connect<West : 3, North : 2>
      aie.connect<West : 3, DMA : 0>
      aie.connect<South : 4, DMA : 1>
      aie.connect<South : 1, North : 0>
      aie.connect<South : 2, North : 1>
      aie.connect<DMA : 0, South : 3>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 1>
    }
    %switchbox_7_4 = aie.switchbox(%tile_7_4) {
      aie.connect<South : 2, DMA : 0>
      aie.connect<South : 0, DMA : 1>
      aie.connect<South : 1, North : 4>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %switchbox_7_5 = aie.switchbox(%tile_7_5) {
      aie.connect<West : 1, DMA : 0>
      aie.connect<South : 4, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
      aie.connect<North : 1, DMA : 1>
      aie.connect<North : 0, DMA : 2>
      aie.connect<North : 3, DMA : 3>
      aie.connect<North : 2, DMA : 4>
      aie.connect<DMA : 4, South : 2>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
      aie.connect<North : 1, DMA : 1>
      aie.connect<North : 0, DMA : 2>
      aie.connect<North : 2, DMA : 3>
      aie.connect<North : 3, DMA : 4>
      aie.connect<DMA : 4, South : 2>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
      aie.connect<North : 1, DMA : 1>
      aie.connect<North : 0, DMA : 2>
      aie.connect<North : 3, DMA : 3>
      aie.connect<North : 2, DMA : 4>
      aie.connect<DMA : 4, South : 2>
    }
    %switchbox_4_0 = aie.switchbox(%shim_noc_tile_4_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_4_0 = aie.shim_mux(%shim_noc_tile_4_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_4_1 = aie.switchbox(%mem_tile_4_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
      aie.connect<North : 1, DMA : 1>
      aie.connect<North : 0, DMA : 2>
      aie.connect<North : 2, DMA : 3>
      aie.connect<North : 3, DMA : 4>
      aie.connect<DMA : 4, South : 2>
    }
    %switchbox_5_0 = aie.switchbox(%shim_noc_tile_5_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_5_0 = aie.shim_mux(%shim_noc_tile_5_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_5_1 = aie.switchbox(%mem_tile_5_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
      aie.connect<North : 1, DMA : 1>
      aie.connect<North : 0, DMA : 2>
      aie.connect<North : 2, DMA : 3>
      aie.connect<North : 3, DMA : 4>
      aie.connect<DMA : 4, South : 2>
    }
    %switchbox_6_0 = aie.switchbox(%shim_noc_tile_6_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_6_0 = aie.shim_mux(%shim_noc_tile_6_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_6_1 = aie.switchbox(%mem_tile_6_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
      aie.connect<North : 1, DMA : 1>
      aie.connect<North : 3, DMA : 2>
      aie.connect<North : 2, DMA : 3>
      aie.connect<North : 0, DMA : 4>
      aie.connect<DMA : 4, South : 2>
    }
    %switchbox_7_0 = aie.switchbox(%shim_noc_tile_7_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_7_0 = aie.shim_mux(%shim_noc_tile_7_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_7_1 = aie.switchbox(%mem_tile_7_1) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
      aie.connect<North : 3, DMA : 1>
      aie.connect<North : 1, DMA : 2>
      aie.connect<North : 2, DMA : 3>
      aie.connect<North : 0, DMA : 4>
      aie.connect<DMA : 4, South : 2>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_0_5 : East, %switchbox_1_5 : West)
    aie.wire(%tile_1_5 : Core, %switchbox_1_5 : Core)
    aie.wire(%tile_1_5 : DMA, %switchbox_1_5 : DMA)
    aie.wire(%switchbox_1_4 : North, %switchbox_1_5 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
    aie.wire(%switchbox_1_5 : East, %switchbox_2_5 : West)
    aie.wire(%tile_2_5 : Core, %switchbox_2_5 : Core)
    aie.wire(%tile_2_5 : DMA, %switchbox_2_5 : DMA)
    aie.wire(%switchbox_2_4 : North, %switchbox_2_5 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%shim_noc_tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%mem_tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%mem_tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
    aie.wire(%switchbox_2_3 : East, %switchbox_3_3 : West)
    aie.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
    aie.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
    aie.wire(%switchbox_3_2 : North, %switchbox_3_3 : South)
    aie.wire(%switchbox_2_4 : East, %switchbox_3_4 : West)
    aie.wire(%tile_3_4 : Core, %switchbox_3_4 : Core)
    aie.wire(%tile_3_4 : DMA, %switchbox_3_4 : DMA)
    aie.wire(%switchbox_3_3 : North, %switchbox_3_4 : South)
    aie.wire(%switchbox_2_5 : East, %switchbox_3_5 : West)
    aie.wire(%tile_3_5 : Core, %switchbox_3_5 : Core)
    aie.wire(%tile_3_5 : DMA, %switchbox_3_5 : DMA)
    aie.wire(%switchbox_3_4 : North, %switchbox_3_5 : South)
    aie.wire(%switchbox_3_0 : East, %switchbox_4_0 : West)
    aie.wire(%shim_mux_4_0 : North, %switchbox_4_0 : South)
    aie.wire(%shim_noc_tile_4_0 : DMA, %shim_mux_4_0 : DMA)
    aie.wire(%switchbox_3_1 : East, %switchbox_4_1 : West)
    aie.wire(%mem_tile_4_1 : Core, %switchbox_4_1 : Core)
    aie.wire(%mem_tile_4_1 : DMA, %switchbox_4_1 : DMA)
    aie.wire(%switchbox_4_0 : North, %switchbox_4_1 : South)
    aie.wire(%switchbox_3_2 : East, %switchbox_4_2 : West)
    aie.wire(%tile_4_2 : Core, %switchbox_4_2 : Core)
    aie.wire(%tile_4_2 : DMA, %switchbox_4_2 : DMA)
    aie.wire(%switchbox_4_1 : North, %switchbox_4_2 : South)
    aie.wire(%switchbox_3_3 : East, %switchbox_4_3 : West)
    aie.wire(%tile_4_3 : Core, %switchbox_4_3 : Core)
    aie.wire(%tile_4_3 : DMA, %switchbox_4_3 : DMA)
    aie.wire(%switchbox_4_2 : North, %switchbox_4_3 : South)
    aie.wire(%switchbox_3_4 : East, %switchbox_4_4 : West)
    aie.wire(%tile_4_4 : Core, %switchbox_4_4 : Core)
    aie.wire(%tile_4_4 : DMA, %switchbox_4_4 : DMA)
    aie.wire(%switchbox_4_3 : North, %switchbox_4_4 : South)
    aie.wire(%switchbox_3_5 : East, %switchbox_4_5 : West)
    aie.wire(%tile_4_5 : Core, %switchbox_4_5 : Core)
    aie.wire(%tile_4_5 : DMA, %switchbox_4_5 : DMA)
    aie.wire(%switchbox_4_4 : North, %switchbox_4_5 : South)
    aie.wire(%switchbox_4_0 : East, %switchbox_5_0 : West)
    aie.wire(%shim_mux_5_0 : North, %switchbox_5_0 : South)
    aie.wire(%shim_noc_tile_5_0 : DMA, %shim_mux_5_0 : DMA)
    aie.wire(%switchbox_4_1 : East, %switchbox_5_1 : West)
    aie.wire(%mem_tile_5_1 : Core, %switchbox_5_1 : Core)
    aie.wire(%mem_tile_5_1 : DMA, %switchbox_5_1 : DMA)
    aie.wire(%switchbox_5_0 : North, %switchbox_5_1 : South)
    aie.wire(%switchbox_4_2 : East, %switchbox_5_2 : West)
    aie.wire(%tile_5_2 : Core, %switchbox_5_2 : Core)
    aie.wire(%tile_5_2 : DMA, %switchbox_5_2 : DMA)
    aie.wire(%switchbox_5_1 : North, %switchbox_5_2 : South)
    aie.wire(%switchbox_4_3 : East, %switchbox_5_3 : West)
    aie.wire(%tile_5_3 : Core, %switchbox_5_3 : Core)
    aie.wire(%tile_5_3 : DMA, %switchbox_5_3 : DMA)
    aie.wire(%switchbox_5_2 : North, %switchbox_5_3 : South)
    aie.wire(%switchbox_4_4 : East, %switchbox_5_4 : West)
    aie.wire(%tile_5_4 : Core, %switchbox_5_4 : Core)
    aie.wire(%tile_5_4 : DMA, %switchbox_5_4 : DMA)
    aie.wire(%switchbox_5_3 : North, %switchbox_5_4 : South)
    aie.wire(%switchbox_4_5 : East, %switchbox_5_5 : West)
    aie.wire(%tile_5_5 : Core, %switchbox_5_5 : Core)
    aie.wire(%tile_5_5 : DMA, %switchbox_5_5 : DMA)
    aie.wire(%switchbox_5_4 : North, %switchbox_5_5 : South)
    aie.wire(%switchbox_5_0 : East, %switchbox_6_0 : West)
    aie.wire(%shim_mux_6_0 : North, %switchbox_6_0 : South)
    aie.wire(%shim_noc_tile_6_0 : DMA, %shim_mux_6_0 : DMA)
    aie.wire(%switchbox_5_1 : East, %switchbox_6_1 : West)
    aie.wire(%mem_tile_6_1 : Core, %switchbox_6_1 : Core)
    aie.wire(%mem_tile_6_1 : DMA, %switchbox_6_1 : DMA)
    aie.wire(%switchbox_6_0 : North, %switchbox_6_1 : South)
    aie.wire(%switchbox_5_2 : East, %switchbox_6_2 : West)
    aie.wire(%tile_6_2 : Core, %switchbox_6_2 : Core)
    aie.wire(%tile_6_2 : DMA, %switchbox_6_2 : DMA)
    aie.wire(%switchbox_6_1 : North, %switchbox_6_2 : South)
    aie.wire(%switchbox_5_3 : East, %switchbox_6_3 : West)
    aie.wire(%tile_6_3 : Core, %switchbox_6_3 : Core)
    aie.wire(%tile_6_3 : DMA, %switchbox_6_3 : DMA)
    aie.wire(%switchbox_6_2 : North, %switchbox_6_3 : South)
    aie.wire(%switchbox_5_4 : East, %switchbox_6_4 : West)
    aie.wire(%tile_6_4 : Core, %switchbox_6_4 : Core)
    aie.wire(%tile_6_4 : DMA, %switchbox_6_4 : DMA)
    aie.wire(%switchbox_6_3 : North, %switchbox_6_4 : South)
    aie.wire(%switchbox_5_5 : East, %switchbox_6_5 : West)
    aie.wire(%tile_6_5 : Core, %switchbox_6_5 : Core)
    aie.wire(%tile_6_5 : DMA, %switchbox_6_5 : DMA)
    aie.wire(%switchbox_6_4 : North, %switchbox_6_5 : South)
    aie.wire(%switchbox_6_0 : East, %switchbox_7_0 : West)
    aie.wire(%shim_mux_7_0 : North, %switchbox_7_0 : South)
    aie.wire(%shim_noc_tile_7_0 : DMA, %shim_mux_7_0 : DMA)
    aie.wire(%switchbox_6_1 : East, %switchbox_7_1 : West)
    aie.wire(%mem_tile_7_1 : Core, %switchbox_7_1 : Core)
    aie.wire(%mem_tile_7_1 : DMA, %switchbox_7_1 : DMA)
    aie.wire(%switchbox_7_0 : North, %switchbox_7_1 : South)
    aie.wire(%switchbox_6_2 : East, %switchbox_7_2 : West)
    aie.wire(%tile_7_2 : Core, %switchbox_7_2 : Core)
    aie.wire(%tile_7_2 : DMA, %switchbox_7_2 : DMA)
    aie.wire(%switchbox_7_1 : North, %switchbox_7_2 : South)
    aie.wire(%switchbox_6_3 : East, %switchbox_7_3 : West)
    aie.wire(%tile_7_3 : Core, %switchbox_7_3 : Core)
    aie.wire(%tile_7_3 : DMA, %switchbox_7_3 : DMA)
    aie.wire(%switchbox_7_2 : North, %switchbox_7_3 : South)
    aie.wire(%switchbox_6_4 : East, %switchbox_7_4 : West)
    aie.wire(%tile_7_4 : Core, %switchbox_7_4 : Core)
    aie.wire(%tile_7_4 : DMA, %switchbox_7_4 : DMA)
    aie.wire(%switchbox_7_3 : North, %switchbox_7_4 : South)
    aie.wire(%switchbox_6_5 : East, %switchbox_7_5 : West)
    aie.wire(%tile_7_5 : Core, %switchbox_7_5 : Core)
    aie.wire(%tile_7_5 : DMA, %switchbox_7_5 : DMA)
    aie.wire(%switchbox_7_4 : North, %switchbox_7_5 : South)
  }
}
