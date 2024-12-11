#include "exateppabm/demographics.h"

#include <array>
#include <cstdint>

#include "flamegpu/flamegpu.h"
#include "exateppabm/input.h"
#include "exateppabm/util.h"

namespace exateppabm {
namespace demographics {

void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params) {
    // Get a handle to the model environment description object
    flamegpu::EnvironmentDescription env = model.Environment();

    // Define a new environment property array variable, with the per-demographic infection susceptibility modifier
    // @todo - store this environment property variable name somewhere.
    env.newProperty<float, demographics::AGE_COUNT>("relative_susceptibility_per_demographic", {
        params.relative_susceptibility_0_9,
        params.relative_susceptibility_10_19,
        params.relative_susceptibility_20_29,
        params.relative_susceptibility_30_39,
        params.relative_susceptibility_40_49,
        params.relative_susceptibility_50_59,
        params.relative_susceptibility_60_69,
        params.relative_susceptibility_70_79,
        params.relative_susceptibility_80
    }, true);
}

std::array<demographics::Age, demographics::AGE_COUNT> getAllAgeDemographics() {
    std::array<demographics::Age, demographics::AGE_COUNT> all = {{
        demographics::Age::AGE_0_9,
        demographics::Age::AGE_10_19,
        demographics::Age::AGE_20_29,
        demographics::Age::AGE_30_39,
        demographics::Age::AGE_40_49,
        demographics::Age::AGE_50_59,
        demographics::Age::AGE_60_69,
        demographics::Age::AGE_70_79,
        demographics::Age::AGE_80
    }};
    return all;
}

std::array<float, demographics::AGE_COUNT> getAgeDemographicCumulativeProbabilityArray(const exateppabm::input::config& params) {
    // Prepare a probability matrix for selecting an age demographic for the agent based on the ratio from the configuration.
    // @todo - this could probably be cleaned up
    std::uint64_t configDemographicSum = params.population_0_9 + params.population_10_19 + params.population_20_29 + params.population_30_39 + params.population_40_49 + params.population_50_59 + params.population_60_69 + params.population_70_79 + params.population_80;

    std::array<float, demographics::AGE_COUNT> demographicProbabilties =  {{
        params.population_0_9 / static_cast<float>(configDemographicSum),
        params.population_10_19 / static_cast<float>(configDemographicSum),
        params.population_20_29 / static_cast<float>(configDemographicSum),
        params.population_30_39 / static_cast<float>(configDemographicSum),
        params.population_40_49 / static_cast<float>(configDemographicSum),
        params.population_50_59 / static_cast<float>(configDemographicSum),
        params.population_60_69 / static_cast<float>(configDemographicSum),
        params.population_70_79 / static_cast<float>(configDemographicSum),
        params.population_80 / static_cast<float>(configDemographicSum)
    }};
    // Perform an inclusive scan to convert to cumulative probability
    // Using a local method which supports inclusive scans in old libstc++
    exateppabm::util::inclusive_scan(demographicProbabilties.begin(), demographicProbabilties.end(), demographicProbabilties.begin());
    return demographicProbabilties;
}

}  // namespace demographics
}  // namespace exateppabm
