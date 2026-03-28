add_executable(libre-dt-lab-chart
               main.c
               colorchart.c
               deltaE.c
               ../common/pfm.c
               dtcairo.c
               common.c
               thinplate.c
               tonecurve.c)

target_include_directories(libre-dt-lab-chart
                           PRIVATE
                              ${CMAKE_CURRENT_BINARY_DIR}/..)

target_compile_options(libre-dt-lab-chart
                       PRIVATE
                           ${LCMS2_DEFINITIONS}
                           ${GTK3_DEFINITIONS})

target_link_libraries(libre-dt-lab-chart
                      PRIVATE
                        ${LCMS2_LIBRARIES}
                        ${GTK3_LIBRARIES}
                        lib_darktable
                        m)
if (WIN32)
  set_target_properties(libre-dt-lab-chart PROPERTIES LINK_FLAGS "-mwindows -Wl,-subsystem,windows")
  _detach_debuginfo (libre-dt-lab-chart bin)
else()
    set_target_properties(libre-dt-lab-chart
                          PROPERTIES
                          INSTALL_RPATH ${RPATH_ORIGIN}/${REL_BIN_TO_LIBDIR})
endif(WIN32)

install(TARGETS libre-dt-lab-chart DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT DTApplication)
