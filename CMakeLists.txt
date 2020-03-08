cmake_minimum_required(VERSION 3.12)
project(Nim)

set(CMAKE_CXX_STANDARD 14)

add_executable(Nim main.cpp game.h agent.h game.cpp state.h state.cpp agent.cpp action.h action.cpp)