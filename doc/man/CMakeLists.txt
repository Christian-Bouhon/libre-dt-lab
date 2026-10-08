###################################
# ADD NEW MANPAGES HERE
###################################
set(MANPAGES
  "darktable.pod"
  "darktable-cli.pod"
  "darktable-generate-cache.pod"
  "darktable-cltest.pod"
  "darktable-cmstest.pod"
)
###################################
# THAT'S IT, NO MORE CHANGES NEEDED
###################################

# Reproducible date for the man pages: SOURCE_DATE_EPOCH if set, otherwise
# the date of the last git commit. Empty falls back to pod2man's default.
set(DT_MAN_DATE "")
if(DEFINED ENV{SOURCE_DATE_EPOCH})
  execute_process(
    COMMAND date -u -d "@$ENV{SOURCE_DATE_EPOCH}" +%Y-%m-%d
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DT_MAN_DATE
    ERROR_QUIET
  )
endif()
if(NOT DT_MAN_DATE)
  execute_process(
    COMMAND git log -n1 --format=%cd --date=short
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DT_MAN_DATE
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ERROR_QUIET
  )
endif()




add_custom_target(manpages ALL)
set(MANPAGE_SOURCES)

macro(add_manpage pod)
  list(APPEND MANPAGE_SOURCES ${pod})
  string(REPLACE ".pod" ".1" manpage ${pod})
  # install the manpages under the Libre DT-Lab name so that they do not
  # clash with the official darktable package
  string(REPLACE "darktable" "libre-dt-lab" manpage ${manpage})
  set(source ${CMAKE_CURRENT_SOURCE_DIR}/${pod})
  set(target ${CMAKE_CURRENT_BINARY_DIR}/${manpage})
  add_custom_command(
    OUTPUT ${target}
    COMMAND sh ${CMAKE_SOURCE_DIR}/tools/makeman.sh ${source} ${CMAKE_CURRENT_SOURCE_DIR}/../../AUTHORS ${target} "${PROJECT_VERSION}" "${DT_MAN_DATE}"
    DEPENDS ${source} ${CMAKE_CURRENT_SOURCE_DIR}/../../AUTHORS ${CMAKE_SOURCE_DIR}/tools/makeman.sh
  )
  add_custom_target(manpage_${manpage} ALL DEPENDS ${target})
  add_dependencies(manpages manpage_${manpage})
  install(FILES ${target} DESTINATION ${CMAKE_INSTALL_MANDIR}/man1 COMPONENT DTDocuments)
endmacro (add_manpage)

foreach(manpage ${MANPAGES})
  add_manpage(${manpage})
endforeach(manpage)


# we also want to have translated manpages
find_program(po4a-translate_BIN po4a-translate)

# create and install translated manpages
if(${po4a-translate_BIN} STREQUAL "po4a-translate_BIN-NOTFOUND")
  message("Missing po4a-translate. Can NOT create translated manpages")
else()
  file(STRINGS "po/LINGUAS" LANGUAGES)
  foreach(language ${LANGUAGES})
    set(pofile "po/${language}.po")
    add_custom_target(manpage-${language} ALL)
    add_dependencies(manpages manpage-${language})
    foreach(pod ${MANPAGE_SOURCES})
      string(REPLACE ".pod" ".1" manpage ${pod})
      # install under the Libre DT-Lab name (see add_manpage above)
      string(REPLACE "darktable" "libre-dt-lab" manpage ${manpage})
      set(target "${CMAKE_CURRENT_BINARY_DIR}/${language}/${manpage}")
      set(tmppod "${CMAKE_CURRENT_BINARY_DIR}/${language}/${pod}")
      # create translated .pod
      add_custom_command(
        OUTPUT ${tmppod}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${language}
        COMMAND ${po4a-translate_BIN} -f pod
                -m ${CMAKE_CURRENT_SOURCE_DIR}/${pod}
                -p ${CMAKE_CURRENT_SOURCE_DIR}/${pofile}
                -l ${tmppod}
                -k 0
        DEPENDS ${pofile} ${pod}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
      # turn the .pod into a manpage
      add_custom_command(
        OUTPUT ${target}
        COMMAND sh ${CMAKE_SOURCE_DIR}/tools/makeman.sh ${tmppod} ${CMAKE_CURRENT_SOURCE_DIR}/../../AUTHORS ${target} "${PROJECT_VERSION}" "${DT_MAN_DATE}"
        DEPENDS ${tmppod} ${CMAKE_CURRENT_SOURCE_DIR}/../../AUTHORS ${CMAKE_SOURCE_DIR}/tools/makeman.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
      add_custom_target(manpage-${language}-${manpage} ALL DEPENDS ${target})
      add_dependencies(manpage-${language} manpage-${language}-${manpage})
      install(FILES ${target} DESTINATION ${CMAKE_INSTALL_MANDIR}/${language}/man1 COMPONENT DTDocuments)
    endforeach(pod)
  endforeach(language)
endif()

# we have to split the update targets to a different directory to prevent cmake from calling it all the time :-(
add_subdirectory(po)
