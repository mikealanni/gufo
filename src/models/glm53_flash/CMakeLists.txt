# GLM-5.3-Flash (`glm5-next`). The CPU side owns metadata, the tensor manifest
# and the tokenizer-adjacent contracts; GPU kernels live under kernels/rocm and
# are added once the layer-0 reference matches.
add_library(gufo_glm53_flash STATIC
  runtime/config.cpp
  runtime/manifest.cpp
  runtime/layout.cpp
  chat_template.cpp
  reference.cpp)

target_include_directories(gufo_glm53_flash PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gufo_glm53_flash PUBLIC gufo_core)

# Host-only artifact check: parses the pinned `glm5-next` metadata and rejects
# anything this build cannot run. Not part of the production package.
add_executable(gufo_glm53_probe EXCLUDE_FROM_ALL probe_main.cpp)
target_link_libraries(gufo_glm53_probe PRIVATE gufo_glm53_flash)

if(ENGINE_ENABLE_HIP)
  # Decode matvec over the vendored DS4 mmvq tier (covers Q6_K, Q8_0, IQ2_XXS
  # and IQ2_S). Same dialect and feature macros as the tier it links.
  add_library(gufo_glm53_flash_rocm STATIC kernels/rocm/matvec.hip.cpp
    kernels/rocm/mhc.hip.cpp
    kernels/rocm/kda.hip.cpp
    kernels/rocm/dsa.hip.cpp
    kernels/rocm/moe.hip.cpp
    kernels/rocm/npu_decode.hip.cpp
    kernels/rocm/router.hip.cpp
    kernels/rocm/embed.hip.cpp
    kernels/rocm/mtp.hip.cpp)
  set_source_files_properties(kernels/rocm/matvec.hip.cpp
    kernels/rocm/mhc.hip.cpp
    kernels/rocm/kda.hip.cpp
    kernels/rocm/dsa.hip.cpp
    kernels/rocm/moe.hip.cpp
    kernels/rocm/npu_decode.hip.cpp
    kernels/rocm/router.hip.cpp
    kernels/rocm/embed.hip.cpp
    kernels/rocm/mtp.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_glm53_flash_rocm PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_glm53_flash_rocm PRIVATE
    GGML_USE_HIP
    GGML_HIP_NO_VMM
    DS4_ROCM_WMMA_MMQ_Y=64)
  target_compile_options(gufo_glm53_flash_rocm PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_glm53_flash_rocm PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(gufo_glm53_flash_rocm PUBLIC
    gufo_glm53_flash gufo_deepseek_v4_flash_mmq)

  # Host-side engine; plain C++20 (the kernel library keeps the C++17 dialect
  # of the vendored mmq headers).
  add_library(gufo_glm53_flash_engine STATIC kernels/rocm/executor.cpp
    engine.cpp)
  target_include_directories(gufo_glm53_flash_engine PUBLIC
    ${CMAKE_SOURCE_DIR})
  target_link_libraries(gufo_glm53_flash_engine PUBLIC
    gufo_glm53_flash_rocm gufo_qwen38_flash_next gufo_core hip::host
    roc::hipblas)
endif()
