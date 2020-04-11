cmake_minimum_required(VERSION 3.12)
project(nim_rl)

set (Python_TARGET_VERSION "" CACHE STRING "Specify a target Python version. \
Any Python library found with CMake modules FindPython2/FindPython3 that \
agrees  with the major version and is at least as high for minor version and \
patch number is accepted. If the variable is not set, we use the FindPython \
module which favours Python 3 over Python 2 if both are available.")
set(CMAKE_CXX_STANDARD 14)
if(APPLE)
    # On MacOS:
    #   -undefined dynamic_lookup is necessary for pybind11 linking
    set (CMAKE_CXX_FLAGS "-Wno-everything -w -undefined dynamic_lookup")

    # On MacOS, we need this so that CMake will use the right Python if the user
    # has a virtual environment active
    set (CMAKE_FIND_FRAMEWORK LAST)
else()
    set (CMAKE_CXX_FLAGS "-Wno-everything")
endif()

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set (NIM_RL_CORE_FILES game.h game.cpp agent.h agent.cpp state.h state.cpp action.h action.cpp)

add_library(nim_rl_core OBJECT ${NIM_RL_CORE_FILES})
target_include_directories (nim_rl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set (NIM_RL_CORE_OBJECTS $<TARGET_OBJECTS:nim_rl_core>)

set (NIM_RL_OBJECTS $<TARGET_OBJECTS:nim_rl_core>)

include_directories(..)

add_subdirectory (python)

add_executable(nim_rl main.cpp)
target_link_libraries(nim_rl nim_rl_core)
