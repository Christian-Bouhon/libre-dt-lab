add_cmocka_test(test_ai_backend
                SOURCES test_ai_backend.c dt_stubs.c
                LINK_LIBRARIES darktable_ai cmocka::cmocka)

target_compile_definitions(test_ai_backend PRIVATE
  TEST_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")

# ONNX Runtime is dlopen'ed by path, so point the test at the build-tree copy.
# Keep discovery hermetic: an isolated XDG data dir prevents models installed
# by the user from leaking into the fixtures.
set(_ai_test_xdg "${CMAKE_CURRENT_BINARY_DIR}/xdg-data")
file(MAKE_DIRECTORY "${_ai_test_xdg}")
set_tests_properties(test_ai_backend PROPERTIES
  ENVIRONMENT "XDG_DATA_HOME=${_ai_test_xdg};DT_ORT_LIBRARY=${ONNXRuntime_LIB_DIR}/libonnxruntime.so")
