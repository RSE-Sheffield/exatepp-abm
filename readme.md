# ExaTEPP ABM 

A GPU accelerated Agent-Based Model (ABM) of infection disease spreading within a population. 

Implemented using [FLAMEGPU/FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2), with model functionality inspired by [https://github.com/BDI-pathogens/OpenABM-Covid19](BDI-pathogens/OpenABM-Covid19).

## Requirements / Dependencies

This has the same requirements/dependencies as FLAME GPU 2

+ [CMake](https://cmake.org/download/) `>= 3.18`
  + `>= 3.20` if building python bindings using a multi-config generator (Visual Studio, Eclipse or Ninja Multi-Config)
+ [CUDA](https://developer.nvidia.com/cuda-downloads) `>= 11.0` and a [Compute Capability](https://developer.nvidia.com/cuda-gpus) `>= 3.5` NVIDIA GPU.
+ C++17 capable C++ compiler (host), compatible with the installed CUDA version
  + [Microsoft Visual Studio 2019 or 2022](https://visualstudio.microsoft.com/) (Windows)
  + [make](https://www.gnu.org/software/make/) and [GCC](https://gcc.gnu.org/) `>= 8.1` (Linux)
+ [git](https://git-scm.com/)

Optionally:

+ [cpplint](https://github.com/cpplint/cpplint) for linting code
+ [Doxygen](http://www.doxygen.nl/) to build the FLAME GPU 2 documentation
+ [FLAMEGPU2-visualiser](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser) dependencies (fetched if possible)
  + [SDL](https://www.libsdl.org/)
  + [GLM](http://glm.g-truc.net/) *(consistent C++/GLSL vector maths functionality)*
  + [GLEW](http://glew.sourceforge.net/) *(GL extension loader)*
  + [FreeType](http://www.freetype.org/)  *(font loading)*
  + [DevIL](http://openil.sourceforge.net/)  *(image loading)*
  + [Fontconfig](https://www.fontconfig.org/)  *(Linux only, font detection)*

## Building the model

Building via CMake is a three step process:

1. Create a build directory for an out-of tree build
2. Configure CMake into the build directory, using the CLI or GUI to specify configuration options such as GPU architecture
3. Build compilation targets using the configured build system

i.e. To configure a Release build targeting Volta GPUs under linux or WSL, and build using 8 threads

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . --target exatepp_abm -j 8
```

## Usage

@todo.

```bash
./build/Release/exatepp_abm --help
```
