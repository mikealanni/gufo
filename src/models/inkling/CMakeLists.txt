# Inkling 2: CPU reference plus the ROCm engine when HIP is enabled.
add_library(gufo_inkling STATIC
  runtime/config.cpp
  reference.cpp)
target_include_directories(gufo_inkling PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_inkling PUBLIC gufo_core)
add_executable(gufo_inkling_ref_check reference_check.cpp)
target_link_libraries(gufo_inkling_ref_check PRIVATE gufo_inkling)
