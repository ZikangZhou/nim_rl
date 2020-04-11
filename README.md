#### Installation
```
cd nim_rl
git clone https://github.com/pybind/pybind11.git
mkdir -p build
cd build
CXX=clang++ cmake -DPython_TARGET_VERSION=3.7 -DCMAKE_CXX_COMPILER=${CXX} ../nim_rl
make -j$(nproc)
```

#### Setting Your PYTHONPATH environment variable

