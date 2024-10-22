#include "population.h"

#include <random>
#include <numeric>
#include <algorithm>

#include <fmt/core.h>

#include "person.h"
#include "input.h"

namespace exateppabm {

namespace population {

std::unique_ptr<flamegpu::AgentVector> generate(flamegpu::ModelDescription& model, const exateppabm::input::config config, const float env_width, const float interactionRadius) {

    fmt::print("@todo - validate config inputs when generated agents (pop size, initial infected count etc)\n");

    // @todo - assert that the requested initial population is non zero.
    auto pop = std::make_unique<flamegpu::AgentVector>(model.Agent(exateppabm::person::NAME), config.n_total);

    std::uint64_t sq_width = static_cast<std::uint64_t>(env_width);
    // float expectedNeighbours = interactionRadius * interactionRadius;
    // fmt::print("sq_width {} interactionRadius {} expectedNeighbours {}\n", sq_width, interactionRadius, expectedNeighbours);

    // seed host rng for population generation.
    // @todo - does this want to be a separate seed from the config file?
    std::mt19937_64 rng(config.rng_seed);


    // Need to initialise a fixed number of individuals as infected.
    // This not very scalable way of doing it, is to create a vector with one element per individual in the simulation, initialised to false
    // set the first n_seed_infection elements to true/1
    // Shuffle the vector,  and query at agent creation time
    // RNG sampling in-loop would be more memory efficient, but harder to guarantee that exactly enough are created. This will likely be replaced anyway, so quick and dirty is fine.
    std::vector<bool> infected_vector(config.n_total);
    std::fill(infected_vector.begin(), infected_vector.begin() + std::min(config.n_total, config.n_seed_infection), true);
    std::shuffle(infected_vector.begin(), infected_vector.end(), rng);

    // Prepare a probability matrix for selecting an age demographic for the agent based on the ratio from the configuration.
    // @todo abstract this into class/methods.
    // @todo - this hardcoded 9 is a bit grim. Maybe enums can help?
    std::uint64_t configDemographicSum = config.population_0_9 + config.population_10_19 + config.population_20_29 + config.population_30_39 + config.population_40_49 + config.population_50_59 + config.population_60_69 + config.population_70_79 + config.population_80;
    // @todo - map might be more readable than an array (incase the underlying class enum values are ever changed to be a different order?)
    std::array<float, exateppabm::person::DEMOGRAPHIC_COUNT> demographicProbabilties =  {{
        config.population_0_9 / static_cast<float>(configDemographicSum),
        config.population_10_19 / static_cast<float>(configDemographicSum),
        config.population_20_29 / static_cast<float>(configDemographicSum),
        config.population_30_39 / static_cast<float>(configDemographicSum),
        config.population_40_49 / static_cast<float>(configDemographicSum),
        config.population_50_59 / static_cast<float>(configDemographicSum),
        config.population_60_69 / static_cast<float>(configDemographicSum),
        config.population_70_79 / static_cast<float>(configDemographicSum),
        config.population_80 / static_cast<float>(configDemographicSum)
    }};
    // Perform an inclusive scan to convert to cumulative probability
    std::inclusive_scan(demographicProbabilties.begin(), demographicProbabilties.end(), demographicProbabilties.begin());
    // std::array<std::uint8_t, exateppabm::person::DEMOGRAPHIC_COUNT> allDemographics = {{0, 1, 2, 3, 4, 5, 6, 7, 8}};
    std::array<exateppabm::person::Demographic, exateppabm::person::DEMOGRAPHIC_COUNT> allDemographics = {{
        exateppabm::person::Demographic::AGE_0_9,
        exateppabm::person::Demographic::AGE_10_19,
        exateppabm::person::Demographic::AGE_20_29,
        exateppabm::person::Demographic::AGE_30_39,
        exateppabm::person::Demographic::AGE_40_49,
        exateppabm::person::Demographic::AGE_50_59,
        exateppabm::person::Demographic::AGE_60_69,
        exateppabm::person::Demographic::AGE_70_79,
        exateppabm::person::Demographic::AGE_80
    }};

    // per demo total is not an output in time series.
    // Alternately, we need to initialise the exact number of each age band, not RNG, and just scale it down accordingly. Will look at in "realistic" population generation
    std::array<std::uint64_t, exateppabm::person::DEMOGRAPHIC_COUNT> createdPerDemographic =  {{0, 0, 0, 0, 0, 0, 0, 0, 0}};
    std::uniform_real_distribution<float> demo_dist(0.0f, 1.0f);

    unsigned idx = 0;
    for (auto person : *pop) {
        // Infections status
        bool infected = infected_vector.at(idx);
        person.setVariable<std::uint32_t>(exateppabm::person::v::INFECTED, infected);

        // Demographic
        // @todo - this is a bit grim, enum class aren't as nice as hoped.
        float demo_random = demo_dist(rng);
        // @todo - abstract this into a method.
        exateppabm::person::Demographic demo = exateppabm::person::Demographic::AGE_0_9;
        for(std::uint8_t i = 0; i < exateppabm::person::DEMOGRAPHIC_COUNT; i++) {
            if(demo_random < demographicProbabilties[i]){
                demo = allDemographics[i];
                createdPerDemographic[i]++;
                break;
            }
        }
        person.setVariable<std::uint8_t>(exateppabm::person::v::DEMOGRAPHIC, static_cast<uint8_t>(demo));

        // Location in 3D space (temp/vis)
        unsigned row = idx / sq_width;
        unsigned col = idx % sq_width;
        person.setVariable<float>(exateppabm::person::v::x, col);  // @todo temp
        person.setVariable<float>(exateppabm::person::v::y, row);  // @todo -temp
        person.setVariable<float>(exateppabm::person::v::z, 0);  // @todo -temp

        // Inc counter
        ++idx;
    }

    // Print a summary of population creation for now.
    fmt::print("Created {} people with {} infected.\n", config.n_total, config.n_seed_infection);
    fmt::print("Demographics {{\n");
    fmt::print("   0- 9 = {}\n", createdPerDemographic[0]);
    fmt::print("  10-19 = {}\n", createdPerDemographic[1]);
    fmt::print("  20-29 = {}\n", createdPerDemographic[2]);
    fmt::print("  30-39 = {}\n", createdPerDemographic[3]);
    fmt::print("  40-49 = {}\n", createdPerDemographic[4]);
    fmt::print("  50-59 = {}\n", createdPerDemographic[5]);
    fmt::print("  60-69 = {}\n", createdPerDemographic[6]);
    fmt::print("  70-79 = {}\n", createdPerDemographic[7]);
    fmt::print("  80+   = {}\n", createdPerDemographic[8]);
    fmt::print("}}\n");

    // @todo - move this.
    // Also set related environment properties. This does not really belong here, but its best fit (for now).
    flamegpu::EnvironmentDescription env = model.Environment();
    env.setProperty<float>("INFECTION_INTERACTION_RADIUS", interactionRadius);

    return pop;
}


}  // namespsace population

} // namespace exateppabm