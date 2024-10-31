#pragma once
#include <string>
#include <memory>
#include <filesystem>
#include "CLI/App.hpp"

namespace exateppabm {

namespace cli {

/**
 * Structure containing values from the CLI
 * 
 * This is intended for per-execution parameters, such as input configuration file path, GPU and output file path.
 * 
 * @todo - this may or may not include the ability to override certain configuration parameters compared to the file on disk, but that significantly complicates input parsing/setup.
 * @todo - check for file overwrite unless --force / --yes passed?
 */
struct params {
    /**
     * GPU device ordinal for use at runtime (0 indexed)
     */
    int device = 0;
    /**
     * Verbosity of simulation execution
     */
    int verbosity = 0;
    /**
     * Path to the input parameter file to read model parameters from
     */
    std::string inputParamFile;
    /**
     * Path to directory for file output
     */
    std::string outputDir = std::filesystem::current_path();
};

/**
 * Setup CLI11 for CLI parsing, which will store data into the params object
 * 
 * @param app CLI11 App object
 * @param params struct where parsed values will be stored
 */
void setup(CLI::App& app, std::shared_ptr<exateppabm::cli::params> params);

/**
 * Print the params struct out for testing / debugging purposes.
 * 
 * @param params parameters struct
 */
void print(const exateppabm::cli::params params);

}  // namespace cli
}  // namespace exateppabm
