#include "PerformanceFile.h"

#include <fmt/core.h>

#include <vector>

namespace exateppabm {
namespace output {

PerformanceFile::PerformanceFile(std::filesystem::path directory) : OutputFile(directory / PerformanceFile::DEFAULT_FILENAME) { }

PerformanceFile::~PerformanceFile() { }

void PerformanceFile::reset() {
    this->metadata = {};
    this->timers = {};
}

bool PerformanceFile::write() {
    if (!this->_handle) {
        this->open();
    }

    // Print to the file handle
    fmt::print(_handle, "{{\n");

    fmt::print(_handle, "  \"build_type\": \"{}\",\n", this->metadata.build_type);
    fmt::print(_handle, "  \"flamegpu_seatbelts\": {},\n", this->metadata.flamegpu_seatbelts);
    fmt::print(_handle, "  \"device\": {},\n", this->metadata.device);
    fmt::print(_handle, "  \"device_name\": \"{}\",\n", this->metadata.device_name);
    fmt::print(_handle, "  \"device_sm_count\": {},\n", this->metadata.device_sm_count);
    fmt::print(_handle, "  \"device_memory\": {},\n", this->metadata.device_memory);
    fmt::print(_handle, "  \"parameter_path\": \"{}\",\n", this->metadata.parameter_path.c_str());
    fmt::print(_handle, "  \"duration\": {},\n", this->metadata.duration);
    fmt::print(_handle, "  \"n_total\": {},\n", this->metadata.n_total);
    fmt::print(_handle, "  \"totalProgram\": {},\n", this->timers.totalProgram.getElapsedSeconds());
    fmt::print(_handle, "  \"configParsing\": {},\n", this->timers.configParsing.getElapsedSeconds());
    fmt::print(_handle, "  \"preSimulate\": {},\n", this->timers.preSimulate.getElapsedSeconds());
    fmt::print(_handle, "  \"simulate\": {},\n", this->timers.simulate.getElapsedSeconds());
    fmt::print(_handle, "  \"postSimulate\": {},\n", this->timers.postSimulate.getElapsedSeconds());
    fmt::print(_handle, "  \"flamegpuRTCElapsed\": {},\n", this->timers.flamegpuRTCElapsed);
    fmt::print(_handle, "  \"flamegpuInitElapsed\": {},\n", this->timers.flamegpuInitElapsed);
    fmt::print(_handle, "  \"flamegpuExitElapsed\": {},\n", this->timers.flamegpuExitElapsed);
    fmt::print(_handle, "  \"flamegpuSimulateElapsed\": {}\n", this->timers.flamegpuSimulateElapsed);

    fmt::print(_handle, "}}\n");


    fmt::print("Performance data written to {}\n", std::filesystem::absolute(this->_filepath).c_str());
    return true;
}

}  // namespace output
}  // namespace exateppabm
