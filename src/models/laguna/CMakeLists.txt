# Laguna: CPU reference first; the ROCm engine follows.
add_library(gufo_laguna STATIC
  runtime/config.cpp
  chat_template.cpp
  reference.cpp)
target_include_directories(gufo_laguna PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_laguna PUBLIC gufo_core)
