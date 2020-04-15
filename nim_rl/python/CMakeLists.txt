if(Python_TARGET_VERSION STREQUAL "")
    find_package(Python COMPONENTS Development)
    include_directories(SYSTEM ${Python_INCLUDE_DIRS})
elseif(Python_TARGET_VERSION VERSION_LESS "3.0.0")
    message(FATAL_ERROR
            "Only Python 3 is supported, not ${Python_TARGET_VERSION}")
elseif(Python_TARGET_VERSION VERSION_GREATER_EQUAL "3.0.0")
    # Look for the Interpreter component so that the linker uses the correct
    # libpython when a virtual environment is loaded
    find_package(Python3 COMPONENTS Interpreter Development)
    include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
    if(Python3_VERSION VERSION_LESS Python_TARGET_VERSION)
        message(FATAL_ERROR
                "Python found ${Python3_VERSION} < ${Python_TARGET_VERSION}")
    endif()
endif()

add_library(pynim MODULE pybind11/pynim.cpp ${NIM_RL_OBJECTS})

set_target_properties(pynim PROPERTIES PREFIX "")
