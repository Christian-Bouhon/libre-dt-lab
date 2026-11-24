add_cmocka_test(test_assistant_endpoint
                SOURCES test_assistant_endpoint.c
                LINK_LIBRARIES lib_darktable cmocka)

# Windows: libs have to be copied next to the executable
if(WIN32)
    _copy_required_library(test_assistant_endpoint lib_darktable)
endif(WIN32)
