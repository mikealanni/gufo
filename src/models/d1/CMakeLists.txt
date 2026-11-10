add_library(gufo_d1_prompt STATIC prompt.cpp packing.cpp judge.cpp)
target_include_directories(gufo_d1_prompt PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_d1_prompt PUBLIC gufo_lfm3)
add_executable(gufo_d1_decide_check EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/models/d1/decide_check.cpp)
target_link_libraries(gufo_d1_decide_check PRIVATE gufo_d1_prompt gufo_lfm3_reference)
add_executable(gufo_d1_judge_check EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/models/d1/judge_check.cpp)
target_link_libraries(gufo_d1_judge_check PRIVATE gufo_d1_prompt gufo_lfm3_reference)
