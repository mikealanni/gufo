# Raon-1 mini: CPU reference plus the ROCm decode engine.
add_library(gufo_raon1 STATIC
  runtime/config.cpp
  reference.cpp)
target_include_directories(gufo_raon1 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_raon1 PUBLIC gufo_core)

add_executable(gufo_raon1_ref_check reference_check.cpp)
target_link_libraries(gufo_raon1_ref_check PRIVATE gufo_raon1)

if(ENGINE_ENABLE_HIP)
  add_library(gufo_raon1_rocm STATIC kernels/rocm/raon.hip.cpp)
  set_source_files_properties(kernels/rocm/raon.hip.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_raon1_rocm PROPERTIES HIP_STANDARD 20)
  target_compile_options(gufo_raon1_rocm PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_raon1_rocm PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_raon1_rocm PUBLIC gufo_raon1 hip::host)

  add_library(gufo_raon1_engine STATIC engine.cpp kernels/rocm/executor.cpp)
  target_include_directories(gufo_raon1_engine PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_raon1_engine PUBLIC
    gufo_raon1_rocm gufo_glm53_flash_rocm gufo_qwen38_flash_next gufo_core hip::host)

  add_executable(gufo_raon1_gpu_check EXCLUDE_FROM_ALL gpu_check.cpp)
  target_link_libraries(gufo_raon1_gpu_check PRIVATE gufo_raon1_engine)
endif()
