include_directories(${DARKTABLE_BINDIR})
add_executable(libre-dt-lab-cltest main.c)

set_target_properties(libre-dt-lab-cltest PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(libre-dt-lab-cltest lib_darktable)

if (WIN32)
  _detach_debuginfo (libre-dt-lab-cltest bin)
else()
    set_target_properties(libre-dt-lab-cltest
                          PROPERTIES
                          INSTALL_RPATH ${RPATH_ORIGIN}/${REL_BIN_TO_LIBDIR}
                          RUNTIME_OUTPUT_DIRECTORY ${DARKTABLE_BINDIR})
endif(WIN32)

install(TARGETS libre-dt-lab-cltest DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT DTApplication)
