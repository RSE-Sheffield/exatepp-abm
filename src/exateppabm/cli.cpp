#include "exateppabm/cli.h"

#include <fmt/core.h>

#include <memory>

namespace exateppabm {

namespace cli {

void setup(CLI::App& app, std::shared_ptr<exateppabm::cli::params> params) {
    app.add_option("-d, --device", params->device, "CUDA Device ordinal (GPU) to use (0 indexed)");
    app.add_flag("-v, --verbose", params->verbosity, "Verbosity of simulation output, forwarded to FLAME GPU 2");
    app.add_option("-i, --input-file", params->inputParamFile, "Path to input parameters file");
    app.add_option("-n, --param-number", params->inputParamLine, "The line from the parameters file to use. 1 indexed (assuming there is a header)");
    app.add_option("-o, --output-dir", params->outputDir, "Path to output directory");
    app.add_flag("--individual-file", params->individualFile, "Enable the creation of the per individual file");
}

void print(const exateppabm::cli::params params) {
    fmt::print("params {{\n");
    fmt::print("  device = {}\n", params.device);
    fmt::print("  verbosity = {}\n", params.verbosity);
    fmt::print("  inputParamFile = {}\n", params.inputParamFile);
    fmt::print("  outputDir = {}\n", params.outputDir);
    fmt::print("}}\n");
}

}  // namespace cli
}  // namespace exateppabm
