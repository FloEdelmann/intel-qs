add_executable(basic_code_for_scaling.exe basic_code_for_scaling.cpp)
add_executable(specv2_bench.exe specv2_bench.cpp)
add_executable(single_node_performance.exe single_node_performance.cpp)
target_link_libraries(basic_code_for_scaling.exe PRIVATE iqs)
target_link_libraries(specv2_bench.exe PRIVATE iqs)
target_link_libraries(single_node_performance.exe PRIVATE iqs)

set_target_properties( 
    basic_code_for_scaling.exe
    specv2_bench.exe
    single_node_performance.exe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
