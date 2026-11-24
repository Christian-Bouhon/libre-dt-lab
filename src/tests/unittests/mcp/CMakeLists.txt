# White-box tests for the MCP bridge's path and name handling. Following
# test_filmicrgb.c, dt_bridge.c is #included directly for access to its static
# helpers, so the test needs the same json-glib flags the mcp target uses.
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0)

add_cmocka_test(test_mcp_paths
                SOURCES test_mcp_paths.c
                LINK_LIBRARIES lib_darktable whereami cmocka ${JSON_GLIB_LIBRARIES})

target_include_directories(test_mcp_paths PRIVATE ${JSON_GLIB_INCLUDE_DIRS})
target_link_directories(test_mcp_paths PRIVATE ${JSON_GLIB_LIBRARY_DIRS})

if(WIN32)
    _copy_required_library(test_mcp_paths lib_darktable)
endif(WIN32)
