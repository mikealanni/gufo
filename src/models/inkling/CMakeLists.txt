# Inkling 2: CPU reference plus the ROCm engine when HIP is enabled.
add_library(gufo_inkling STATIC
  runtime/config.cpp)
target_include_directories(gufo_inkling PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_inkling PUBLIC gufo_core)
