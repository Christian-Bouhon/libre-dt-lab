include_directories(${DARKTABLE_BINDIR})
add_executable(libre-dt-lab-cli main.c)

set_target_properties(libre-dt-lab-cli PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(libre-dt-lab-cli lib_darktable whereami)

if (WIN32)
  _detach_debuginfo (libre-dt-lab-cli bin)
else()
    set_target_properties(libre-dt-lab-cli
                          PROPERTIES
                          INSTALL_RPATH ${RPATH_ORIGIN}/${REL_BIN_TO_LIBDIR}
                          RUNTIME_OUTPUT_DIRECTORY ${DARKTABLE_BINDIR})
endif(WIN32)

install(TARGETS libre-dt-lab-cli DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT DTApplication)
