cmake_minimum_required(VERSION 3.16)
project(deann LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors")
  set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_SHARED_LINKER_FLAGS "-shared-libasan")
    set(CMAKE_MODULE_LINKER_FLAGS "-shared-libasan")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors")
  set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_SHARED_LINKER_FLAGS "-shared-libasan")
    set(CMAKE_MODULE_LINKER_FLAGS "-shared-libasan")
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors")
  set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
else()
  message(WARNING "Unknown compiler ``${CMAKE_CXX_COMPILER_ID}'' detected; compiler flags are not set")
endif()


find_package(pybind11 REQUIRED)
find_package(Doxygen QUIET)
find_package(Sphinx QUIET)
find_package(Catch2 QUIET)

if (NOT DEFINED MKL_ROOT)
  if (DEFINED ENV{MKL_ROOT})
    set(MKL_ROOT $ENV{MKL_ROOT} CACHE PATH "Root of MKL")
  else()
    message(FATAL_ERROR "Please set the MKL_ROOT to point to MKL")
  endif()
endif()
find_path(MKL_INCLUDE_DIR mkl_cblas.h HINTS ${MKL_ROOT}/include)
include_directories(${MKL_INCLUDE_DIR})
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
if (APPLE)
  find_library(MKL_LIBRARIES libmkl_rt.dylib HINTS ${MKL_ROOT}/lib/intel64/ ${MKL_ROOT}/lib)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  find_library(MKL_RT libmkl_rt.so HINTS ${MKL_ROOT}/lib/intel64/ ${MKL_ROOT}/lib)
  set(MKL_LIBRARIES ${MKL_RT} -Wl,--no-as-needed)
else()
  message(WARNING "Unsupported operating system; MKL detection may fail")
  find_library(MKL_LIBRARIES libmkl_rt.so HINTS ${MKL_ROOT}/lib/intel64/ ${MKL_ROOT}/lib)
endif()

# this definition can be enabled for some debuggery purposes
# if you don't know why you should enable it, please ignore it
# add_compile_definitions(DEANN_ENABLE_DEBUG_ACCESSORS)

pybind11_add_module(deann deann.cpp)
target_link_libraries(deann PRIVATE ${MKL_LIBRARIES})

if (Catch2_FOUND)
  add_executable(KdeTests KdeTests.cpp)
  target_link_libraries(KdeTests Catch2::Catch2 ${MKL_LIBRARIES})
else()
  message(STATUS "Catch2 not found. Not building unit tests.")
endif()

add_custom_target(testpy ALL DEPENDS ${PROJECT_SOURCE_DIR}/test.py)
add_custom_command(TARGET testpy
  POST_BUILD 
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/test.py ${CMAKE_BINARY_DIR}/test.py
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
add_custom_target(doc DEPENDS ${CMAKE_BINARY_DIR}/Doxyfile deann)

if(DOXYGEN_FOUND)
  file(READ ${PROJECT_SOURCE_DIR}/Doxyfile doxyfilevar)
  string(REGEX REPLACE "INPUT                  =" "INPUT                  = ${PROJECT_SOURCE_DIR}" doxyfilevarreplaced ${doxyfilevar})
  file(WRITE ${CMAKE_BINARY_DIR}/Doxyfile ${doxyfilevarreplaced})
  add_custom_command(TARGET doc
    POST_BUILD
    COMMAND
    Doxygen::doxygen ${CMAKE_BINARY_DIR}/Doxyfile
    )
endif()

if (SPHINX_FOUND)
  file(COPY ${PROJECT_SOURCE_DIR}/conf.py ${PROJECT_SOURCE_DIR}/index.rst ${PROJECT_SOURCE_DIR}/deann.rst DESTINATION ${CMAKE_BINARY_DIR})
  add_custom_command(TARGET doc
    POST_BUILD
    COMMAND
    ${SPHINX_EXECUTABLE} -b html ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/doc/deann)
endif()
