add_library(gufo_lfm3 STATIC config.cpp weights.cpp tokenizer.cpp chat_template.cpp)
target_include_directories(gufo_lfm3 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_lfm3 PUBLIC gufo_core)
add_library(gufo_lfm3_reference STATIC EXCLUDE_FROM_ALL reference.cpp decision_forward.cpp)
target_link_libraries(gufo_lfm3_reference PUBLIC gufo_lfm3)
target_compile_options(gufo_lfm3_reference PRIVATE -O2)
add_executable(gufo_lfm3_decision_forward_check EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/models/lfm3/decision_forward_check.cpp)
target_link_libraries(gufo_lfm3_decision_forward_check PRIVATE gufo_lfm3_reference)
add_executable(gufo_lfm3_generate_check EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/models/lfm3/generate_check.cpp)
target_link_libraries(gufo_lfm3_generate_check PRIVATE gufo_lfm3_reference)
