pybind11_add_module(deann deann.cpp)
target_link_libraries(deann PRIVATE ${MKL_LIBRARIES})

execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

install(TARGETS deann 
  DESTINATION "${PYTHON_SITE_PACKAGES}/deann"
)
