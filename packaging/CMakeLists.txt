if(WIN32 AND NOT BUILD_MSYS2_INSTALL)

  set(plugin_dest_dir lib/darktable/plugins)
  set(APPS "\${CMAKE_INSTALL_PREFIX}/bin/libre-dt-lab.exe") # paths to executables
  get_filename_component(MINGW_PATH ${CMAKE_CXX_COMPILER} PATH)

  # Copy all required DLLs from MSYS2 to the install prefix
  install(
    CODE "
    # Copy all DLLs from MINGW_PATH to install prefix
    file(GLOB MINGW_DLLS \"${MINGW_PATH}/*.dll\")
    foreach(dll \${MINGW_DLLS})
      get_filename_component(dllname \"\${dll}\" NAME)
      if(NOT EXISTS \"\${CMAKE_INSTALL_PREFIX}/bin/\${dllname}\")
        execute_process(COMMAND \${CMAKE_COMMAND} -E copy \"\${dll}\" \"\${CMAKE_INSTALL_PREFIX}/bin/\")
      endif()
    endforeach()
    "
    COMPONENT DTApplication)

endif()
