add_custom_target(doc DEPENDS ${CMAKE_BINARY_DIR}/Doxyfile deann)

if(DOXYGEN_FOUND)
  file(READ ${PROJECT_SOURCE_DIR}/docs/Doxyfile doxyfilevar)
  string(REGEX REPLACE "INPUT                  =" "INPUT                  = ${PROJECT_SOURCE_DIR}/src/" doxyfilevarreplaced ${doxyfilevar})
  file(WRITE ${CMAKE_BINARY_DIR}/Doxyfile ${doxyfilevarreplaced})
  add_custom_command(TARGET doc
    POST_BUILD
    COMMAND
    Doxygen::doxygen ${CMAKE_BINARY_DIR}/Doxyfile
    )
endif()

if (SPHINX_FOUND)
  file(COPY ${PROJECT_SOURCE_DIR}/docs/conf.py ${PROJECT_SOURCE_DIR}/docs/index.rst ${PROJECT_SOURCE_DIR}/docs/deann.rst DESTINATION ${CMAKE_BINARY_DIR})
  add_custom_command(TARGET doc
    POST_BUILD
    COMMAND
    ${SPHINX_EXECUTABLE} -b html ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/docs/doc/deann)
endif()
