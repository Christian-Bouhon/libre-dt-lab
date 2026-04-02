include_directories("${CMAKE_CURRENT_BINARY_DIR}/../")

add_executable(darktable-test-variables variables.c)
set_target_properties(darktable-test-variables PROPERTIES OUTPUT_NAME libre-dt-lab-test-variables)
target_link_libraries(darktable-test-variables lib_darktable)

if(WIN32)
    # This tester sets up a Libre DT-Lab instance (of sorts). Hence it expects libraries at ../lib/libre-dt-lab
    # Easiest way to comply with this on Windows: Put tester executable in same directory as Libre DT-Lab executable
    set_target_properties(darktable-test-variables PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${DARKTABLE_BINDIR}
    )
endif(WIN32)

add_subdirectory(unittests)
