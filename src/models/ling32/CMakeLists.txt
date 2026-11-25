# Ling-3.2-Air (`bailingmoe3`). The CPU side owns metadata, the tensor
# manifest and the correctness-first reference; GPU kernels are added once the
# reference matches upstream numerics.
add_library(gufo_ling32 STATIC
  chat_template.cpp
  runtime/config.cpp
  runtime/manifest.cpp
  reference.cpp)

target_include_directories(gufo_ling32 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gufo_ling32 PUBLIC gufo_core)

# Host-only artifact check: parses the pinned `bailingmoe3` metadata, validates
# every tensor shape, and optionally runs the CPU reference. Not part of the
# production package.
add_executable(gufo_ling32_probe EXCLUDE_FROM_ALL probe_main.cpp)
target_link_libraries(gufo_ling32_probe PRIVATE gufo_ling32)

if(ENGINE_ENABLE_HIP)
  set(LING32_GPU_SOURCES gpu/engine.hip.cpp gpu/kernels.hip.cpp)
  add_library(gufo_ling32_gpu STATIC ${LING32_GPU_SOURCES})
  set_source_files_properties(${LING32_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_ling32_gpu PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_options(gufo_ling32_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O2;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_ling32_gpu PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(gufo_ling32_gpu PUBLIC
    gufo_ling32 gufo_qwen38_flash_next hip::host roc::hipblas)

  add_executable(gufo_ling32_gpu_check EXCLUDE_FROM_ALL gpu_check.cpp)
  target_link_libraries(gufo_ling32_gpu_check PRIVATE gufo_ling32_gpu)
endif()
