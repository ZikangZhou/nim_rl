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

set (NIM_RL_CORE_FILES
        action/action.h
        action/action.cpp
        agent/agent.h
        agent/agent.cpp
        agent/dp_agent.h
        agent/dp_agent.cpp
        agent/human_agent.h
        agent/human_agent.cpp
        agent/monte_carlo_agent.h
        agent/monte_carlo_agent.cpp
        agent/n_step_bootstrapping_agent.h
        agent/n_step_bootstrapping_agent.cpp
        agent/optimal_agent.h
        agent/optimal_agent.cpp
        agent/random_agent.h
        agent/random_agent.cpp
        agent/rl_agent.h
        agent/rl_agent.cpp
        agent/td_agent.h
        agent/td_agent.cpp
        environment/game.h
        environment/game.cpp
        exploration/exploration.h
        state/state.h
        state/state.cpp)

add_library(nim_rl_core OBJECT ${NIM_RL_CORE_FILES})
target_include_directories (nim_rl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set (NIM_RL_CORE_OBJECTS $<TARGET_OBJECTS:nim_rl_core>)

set (NIM_RL_OBJECTS $<TARGET_OBJECTS:nim_rl_core>)

include_directories(..)

add_subdirectory (python)

add_executable(nim_rl test/test.cpp)
target_link_libraries(nim_rl nim_rl_core)
