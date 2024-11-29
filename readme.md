# ExaTEPP ABM

A GPU accelerated Agent-Based Model (ABM) of infection disease spread within a population.

Implemented using [FLAMEGPU/FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2), with model functionality inspired by [https://github.com/BDI-pathogens/OpenABM-Covid19](BDI-pathogens/OpenABM-Covid19).

## Requirements / Dependencies

This has the same requirements/dependencies as FLAME GPU 2, except for CUDA which instead requires >= 11.5 currently (due to use of `constexpr __device__`);

+ [CMake](https://cmake.org/download/) `>= 3.18`
  + `>= 3.20` if building python bindings using a multi-config generator (Visual Studio, Eclipse or Ninja Multi-Config)
+ [CUDA](https://developer.nvidia.com/cuda-downloads) `>= 11.5` and a [Compute Capability](https://developer.nvidia.com/cuda-gpus) `>= 3.5` NVIDIA GPU.
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

i.e. To configure a Release build targeting Ampere (`SM_80`, i.e. A100) GPUs under linux or WSL, and build using 8 threads

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . --target exatepp_abm -j 8
```

### CMake Configuration Options

| Option                            | Default |  Description |
|-----------------------------------|---------|--------------|
| `CMAKE_CUDA_ARCHITECTURES`        | `"50;60;70;80;90"` | ` Specify which CUDA [Compute Capability](https://developer.nvidia.com/cuda-gpus) Architectures to build for |
| `CMAKE_BUILD_TYPE`                | `Release` | CMake build configuration, setting optimisation levels etc. Choose from [`Release`, `RelWithDebInfo`, `MinSizeRel`, `Debug`] |
| `BUILD_TESTING`                   | `OFF` | Enable / disable the test suite |
| `FLAMEGPU_VISUALISATION`          | `OFF` | If FLAME GPU's 3D interactive visualisation should be enabled. Requires OpenGL and local execution. |
| `FLAMEGPU_SEATBELTS`              | `ON` | Enable / Disable additional runtime checks which harm performance but increase usability |
| `FLAMEGPU_SHARE_USAGE_STATISTICS` | `ON` | Enable / Disable FLAME GPU 2 telemetry which helps evidence use/impact of FLAME GPU 2. See the [FLAME GPU 2 user guide for more information](https://docs.flamegpu.com/guide/telemetry/) |

For a list of all available CMake configuration options, run the following from the `build` directory:

```bash
cmake -LH ..
```

### CMake targets

Additional CMake targets can be used for other build processes:

| Target | Description |
|--------|-------------|
| `exatepp_abm`        | The primary executable target for the simulator              |
| `tests`              | The unit test suite target (if enabled)                      |
| `help`               | List all CMake targets (including fetched dependencies)      |
| `all`                | Builds all non-excluded targets                              |
| `clean`              | Deletes files produced by CMake (i.e. compiled objects)      |
| `lint`               | Lints source code within this repository. Requires `cpplint` |
| `all_lint`           | A synonym of `lint`                                          |
| `lint_exatepp_abm`   | Just lint the `exatepp_abm` target's source files            |
| `exatepp_abm_lib`    | The underlying static library which enables the test suite   |
| `flamegpu`           | The FLAME GPU static library target.                         |

## Usage

The produced binary will be located in `bin/<CMAKE_BUILD_TYPE>/` within the build directory.
It can be launched taking a  number of command line arguments, the details of which can be seen by passing `-h`/`--help`.

```bash
./bin/Release/exatepp_abm --help
```

Typical usage involves:

+ `-i <path/to/file.csv>` / `--input-file <path/to/file.csv>` - a csv input file, see `data/inputs` for examples
+ `-n <int>` / `--param-number <int>` - the row of the input parameter CSV to use (uses the first row if not specified)
+ `-o <path/to/outdir/` / `--output-dir <path/to/oudir` - path to a directory for file output  

e.g. from the `build` directory of a `Release` build  

```bash
./bin/Release/exatepp_abm -i ../data/inputs/sample.csv -n 1 -o path/to/outdir/ 
```

## Tests

> **Note**: The test suite is currently very sparse, and should be expanded in the future

A (currently very sparse) set of integration and unit tests can optionally be enabled during CMake Configuration using `-DBUILD_TESTING=ON`.

This enables the use of `CTest` and the creation of the `tests` CMake target.

Tests can be executed from a build directory using `ctest`:

```bash
cd build
ctest .
```

## Linting

Source code linting required `cpplint` to be installed and on your path at CMake Configuration time.

Linting can be initiated via cmake, i.e from a build directory:

```bash
cmake --build . --target lint`
```

## Documentation

> @todo - documentation to follow

## License

This project is distributed under the [MIT Licence](./LICENSE.md).


