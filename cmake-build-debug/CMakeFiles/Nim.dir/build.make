# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zhouzikang/CLionProjects/Nim

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zhouzikang/CLionProjects/Nim/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Nim.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Nim.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Nim.dir/flags.make

CMakeFiles/Nim.dir/main.cpp.o: CMakeFiles/Nim.dir/flags.make
CMakeFiles/Nim.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhouzikang/CLionProjects/Nim/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Nim.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Nim.dir/main.cpp.o -c /Users/zhouzikang/CLionProjects/Nim/main.cpp

CMakeFiles/Nim.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Nim.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zhouzikang/CLionProjects/Nim/main.cpp > CMakeFiles/Nim.dir/main.cpp.i

CMakeFiles/Nim.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Nim.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zhouzikang/CLionProjects/Nim/main.cpp -o CMakeFiles/Nim.dir/main.cpp.s

CMakeFiles/Nim.dir/game.cpp.o: CMakeFiles/Nim.dir/flags.make
CMakeFiles/Nim.dir/game.cpp.o: ../game.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhouzikang/CLionProjects/Nim/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Nim.dir/game.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Nim.dir/game.cpp.o -c /Users/zhouzikang/CLionProjects/Nim/game.cpp

CMakeFiles/Nim.dir/game.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Nim.dir/game.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zhouzikang/CLionProjects/Nim/game.cpp > CMakeFiles/Nim.dir/game.cpp.i

CMakeFiles/Nim.dir/game.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Nim.dir/game.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zhouzikang/CLionProjects/Nim/game.cpp -o CMakeFiles/Nim.dir/game.cpp.s

CMakeFiles/Nim.dir/state.cpp.o: CMakeFiles/Nim.dir/flags.make
CMakeFiles/Nim.dir/state.cpp.o: ../state.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhouzikang/CLionProjects/Nim/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Nim.dir/state.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Nim.dir/state.cpp.o -c /Users/zhouzikang/CLionProjects/Nim/state.cpp

CMakeFiles/Nim.dir/state.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Nim.dir/state.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zhouzikang/CLionProjects/Nim/state.cpp > CMakeFiles/Nim.dir/state.cpp.i

CMakeFiles/Nim.dir/state.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Nim.dir/state.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zhouzikang/CLionProjects/Nim/state.cpp -o CMakeFiles/Nim.dir/state.cpp.s

CMakeFiles/Nim.dir/agent.cpp.o: CMakeFiles/Nim.dir/flags.make
CMakeFiles/Nim.dir/agent.cpp.o: ../agent.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhouzikang/CLionProjects/Nim/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Nim.dir/agent.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Nim.dir/agent.cpp.o -c /Users/zhouzikang/CLionProjects/Nim/agent.cpp

CMakeFiles/Nim.dir/agent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Nim.dir/agent.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zhouzikang/CLionProjects/Nim/agent.cpp > CMakeFiles/Nim.dir/agent.cpp.i

CMakeFiles/Nim.dir/agent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Nim.dir/agent.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zhouzikang/CLionProjects/Nim/agent.cpp -o CMakeFiles/Nim.dir/agent.cpp.s

CMakeFiles/Nim.dir/action.cpp.o: CMakeFiles/Nim.dir/flags.make
CMakeFiles/Nim.dir/action.cpp.o: ../action.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhouzikang/CLionProjects/Nim/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/Nim.dir/action.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Nim.dir/action.cpp.o -c /Users/zhouzikang/CLionProjects/Nim/action.cpp

CMakeFiles/Nim.dir/action.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Nim.dir/action.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zhouzikang/CLionProjects/Nim/action.cpp > CMakeFiles/Nim.dir/action.cpp.i

CMakeFiles/Nim.dir/action.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Nim.dir/action.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zhouzikang/CLionProjects/Nim/action.cpp -o CMakeFiles/Nim.dir/action.cpp.s

# Object files for target Nim
Nim_OBJECTS = \
"CMakeFiles/Nim.dir/main.cpp.o" \
"CMakeFiles/Nim.dir/game.cpp.o" \
"CMakeFiles/Nim.dir/state.cpp.o" \
"CMakeFiles/Nim.dir/agent.cpp.o" \
"CMakeFiles/Nim.dir/action.cpp.o"

# External object files for target Nim
Nim_EXTERNAL_OBJECTS =

Nim: CMakeFiles/Nim.dir/main.cpp.o
Nim: CMakeFiles/Nim.dir/game.cpp.o
Nim: CMakeFiles/Nim.dir/state.cpp.o
Nim: CMakeFiles/Nim.dir/agent.cpp.o
Nim: CMakeFiles/Nim.dir/action.cpp.o
Nim: CMakeFiles/Nim.dir/build.make
Nim: CMakeFiles/Nim.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zhouzikang/CLionProjects/Nim/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable Nim"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Nim.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Nim.dir/build: Nim

.PHONY : CMakeFiles/Nim.dir/build

CMakeFiles/Nim.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Nim.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Nim.dir/clean

CMakeFiles/Nim.dir/depend:
	cd /Users/zhouzikang/CLionProjects/Nim/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zhouzikang/CLionProjects/Nim /Users/zhouzikang/CLionProjects/Nim /Users/zhouzikang/CLionProjects/Nim/cmake-build-debug /Users/zhouzikang/CLionProjects/Nim/cmake-build-debug /Users/zhouzikang/CLionProjects/Nim/cmake-build-debug/CMakeFiles/Nim.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Nim.dir/depend

