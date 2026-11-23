# Inkling 2: CPU reference plus the ROCm engine when HIP is enabled.
add_library(gufo_inkling STATIC
  runtime/config.cpp
  chat_template.cpp
  reference.cpp)
target_include_directories(gufo_inkling PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_inkling PUBLIC gufo_core)
add_executable(gufo_inkling_ref_check reference_check.cpp)
target_link_libraries(gufo_inkling_ref_check PRIVATE gufo_inkling)

if(ENGINE_ENABLE_HIP)
  target_sources(gufo_inkling PRIVATE gpu/engine.hip.cpp gpu/iu4_experts.hip.cpp)
  set_source_files_properties(gpu/engine.hip.cpp gpu/iu4_experts.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_inkling PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_inkling PUBLIC hip::host gufo_qwen38_flash_next gufo_qwen38_flash_next_mmq)
  add_executable(gufo_inkling_gpu_check gpu_check.cpp)
  target_link_libraries(gufo_inkling_gpu_check PRIVATE gufo_inkling)
endif()

if(ENGINE_ENABLE_HIP AND GUFO_ENABLE_NPU)
  add_executable(gufo_inkling_npu_probe npu_probe.cpp)
  target_link_libraries(gufo_inkling_npu_probe PRIVATE gufo_inkling gufo_qwen38_flash_next gufo_core)
endif()
