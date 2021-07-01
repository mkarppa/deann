if (Catch2_FOUND)
  add_executable(KdeTests KdeTests.cpp)
  target_link_libraries(KdeTests Catch2::Catch2 ${MKL_LIBRARIES})
  target_include_directories(KdeTests PRIVATE ${PROJECT_SOURCE_DIR}/src)
else()
  message(STATUS "Catch2 not found. Not building unit tests.")
endif()

add_custom_target(testpy ALL DEPENDS ${PROJECT_SOURCE_DIR}/test/test.py)
add_custom_command(TARGET testpy
  POST_BUILD 
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/test/test.py ${CMAKE_BINARY_DIR}/test.py
)
add_custom_command(TARGET testpy
  POST_BUILD 
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/extern/base.py ${CMAKE_BINARY_DIR}/extern/base.py
)
add_custom_command(TARGET testpy
  POST_BUILD 
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/extern/faiss.py ${CMAKE_BINARY_DIR}/extern/faiss.py
)
add_custom_command(TARGET testpy
  POST_BUILD 
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/extern/annoy.py ${CMAKE_BINARY_DIR}/extern/annoy.py
)
add_custom_command(TARGET testpy
  POST_BUILD 
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/extern/brute.py ${CMAKE_BINARY_DIR}/extern/brute.py
)
