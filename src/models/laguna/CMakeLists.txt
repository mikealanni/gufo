# Laguna: CPU reference plus the ROCm engine when HIP is enabled.
add_library(gufo_laguna STATIC
  runtime/config.cpp
  chat_template.cpp
  reference.cpp)
target_include_directories(gufo_laguna PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_laguna PUBLIC gufo_core)

if(ENGINE_ENABLE_HIP)
  target_sources(gufo_laguna PRIVATE
    gpu/device_model.cpp
    gpu/engine.cpp
    gpu/kernels/kernels.hip.cpp)
  set_source_files_properties(gpu/kernels/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_laguna PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_laguna PUBLIC
    gufo_qwen38_flash_next gufo_qwen38_flash_next_mmq hip::host)
else()
  target_sources(gufo_laguna PRIVATE gpu/engine_stub.cpp)
endif()
