# Installation
The instructions here are for Linux and MacOS.
## Install the development environment on your system
### Prerequisites
* Clang
* CMake 3.12 or later
* Python 3.7
* pip 20.0.2 or later
* virtualenv 20.0.16 or later

Check if your environment is already configured:
```shell script
$ clang --version
$ cmake --version
$ python --version
$ pip3 --version
$ virtualenv --version
```
If these packages are already installed, skip to the next step.
Otherwise, install them:

for Linux users,
```shell script
$ sudo apt update
$ sudo apt install clang cmake python3-dev python3-pip
$ sudo pip3 install -U virtualenv  # system-wide install
```
for mac OS users, Clang is the default compiler for Mac OS X since Xcode 4.2.
If Clang is not found, please install or upgrade Xcode and run the command-line 
developer tools. The following instructions are for installing the other 
packages:
```shell script
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" 
$ export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
$ brew update
$ brew install cmake python  # Python 3
$ sudo pip3 install -U virtualenv  # system-wide install
```
## Create a virtual environment
Python virtual environments are used to isolate package installation from the 
system.
```shell script
$ virtualenv -p python3 /<path_to_venv>/venv  # path to venv can be anywhere you want  
```
Activate the virtual environment using a shell-specific command:
```shell script
$ source /<path_to_venv>/venv/bin/activate  # sh, bash, ksh, or zsh
```
When virtualenv is active, your shell prompt is prefixed with (venv).

Install packages within a virtual environment without affecting the host system 
setup. Start by upgrading pip:
```shell script
(venv) $ pip install --upgrade pip
(venv) $ pip list  # show packages installed within the virtual environment
```
Any time when you want to exit virtualenv later:
```shell script
(venv) $ deactivate
```
Download the NimRL source code and pull in the source dependency 
[pybind11](https://github.com/pybind/pybind11):
```shell script
(venv) $ git clone https://github.com/ZikangZhou/nim_rl
(venv) $ cd nim_rl
(venv) $ git clone https://github.com/pybind/pybind11.git
```
## Build NimRL
Now it's time to build NimRL:
```shell script
(venv) $ mkdir -p build
(venv) $ cd build
(venv) $ CXX=clang++ cmake -DPython_TARGET_VERSION=3.7 -DCMAKE_CXX_COMPILER=${CXX} ../nim_rl
(venv) $ make -j$(nproc)
(venv) $ python3 -m pip install ..
```
## Setting Your PYTHONPATH environment variable
The following should be added to `./venv/bin/activate`:
```
export PYTHONPATH=$PYTHONPATH:/<path_to_nim_rl>
export PYTHONPATH=$PYTHONPATH:/<path_to_nim_rl>/build/python
```
Exit virtualenv and Activate it again:
```shell script
(venv) $ deactivate
$ source /<path_to_venv>/venv/bin/activate  # sh, bash, ksh, or zsh
```
Now it's able to import the Python code (both the C++ binding `pynim` and the 
rest) from any location. Verify it:
```shell script
(venv) $ python -c "from pynim import *; import nim_rl.python"
```
`pynim` and the rest python modules has been successfully installed if no error 
occurs. 
## (optional) Run test.py
```shell script
(venv) $ python /<path_to_nim_rl>/python/tests/test.py
```