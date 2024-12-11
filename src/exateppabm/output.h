#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <memory>

#include "flamegpu/flamegpu.h"

#include "output/OutputFile.h"
#include "output/TimeSeriesFile.h"
#include "output/PerIndividualFile.h"

namespace exateppabm {
namespace output {

/**
 * Define FLAME GPU model functions related to output of data to disk. 
 * 
 * I.e. Step and Exit functions (maybe host layer) which collect data through the simulation and output it to disk.
 * 
 * @todo this will likely find a better home once multiple output types are added.
 * 
 * @param model flamegpu2 model description object to mutate
 * @param outputDirectory path to directory for file output
 * @param individualFile if the per individual file should be written or not
 */
void define(flamegpu::ModelDescription& model, std::filesystem::path outputDirectory, bool individualFile);

}  // namespace output
}  // namespace exateppabm
