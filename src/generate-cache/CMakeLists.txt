include_directories(${DARKTABLE_BINDIR})
add_executable(libre-dt-lab-generate-cache main.c)

set_target_properties(libre-dt-lab-generate-cache PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(libre-dt-lab-generate-cache lib_darktable whereami)

if (WIN32)
  _detach_debuginfo (libre-dt-lab-generate-cache bin)
else()
    set_target_properties(libre-dt-lab-generate-cache
                          PROPERTIES
                          INSTALL_RPATH ${RPATH_ORIGIN}/${REL_BIN_TO_LIBDIR}
                          RUNTIME_OUTPUT_DIRECTORY ${DARKTABLE_BINDIR})
endif(WIN32)

install(TARGETS libre-dt-lab-generate-cache DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT DTApplication)
