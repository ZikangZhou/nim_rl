cmake_minimum_required(VERSION 3.12)
project(Nim)

set(CMAKE_CXX_STANDARD 14)

add_executable(Nim main.cpp game.h player.h game.cpp state.h state.cpp player.cpp)