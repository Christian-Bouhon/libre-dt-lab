include_directories(${DARKTABLE_BINDIR})

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0)

add_executable(libre-dt-lab-mcp
  main.c
  mcp_jsonrpc.c
  mcp_tools.c
  dt_bridge.c)

set_target_properties(libre-dt-lab-mcp PROPERTIES LINKER_LANGUAGE C)

target_include_directories(libre-dt-lab-mcp PRIVATE ${JSON_GLIB_INCLUDE_DIRS})
target_link_directories(libre-dt-lab-mcp PRIVATE ${JSON_GLIB_LIBRARY_DIRS})
target_link_libraries(libre-dt-lab-mcp lib_darktable whereami ${JSON_GLIB_LIBRARIES})

if(WIN32)
  _detach_debuginfo(libre-dt-lab-mcp bin)
else()
  set_target_properties(libre-dt-lab-mcp
                        PROPERTIES
                        INSTALL_RPATH ${RPATH_ORIGIN}/${REL_BIN_TO_LIBDIR}
                        RUNTIME_OUTPUT_DIRECTORY ${DARKTABLE_BINDIR})
endif(WIN32)

install(TARGETS libre-dt-lab-mcp DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT DTApplication)
