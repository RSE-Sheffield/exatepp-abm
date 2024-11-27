#include "exateppabm/visualisation.h"
#include <fmt/core.h>

#include <array>
#include <memory>
#include <vector>
#include "flamegpu/flamegpu.h"
#include "exateppabm/person.h"
#include "exateppabm/disease/SEIR.h"

namespace exateppabm {
namespace visualisation {

namespace {
#if defined(FLAMEGPU_VISUALISATION)
// namespace scoped pointer to visualiastion object for use in the two separate methods here, without needing to add macros to the pseudo main method
std::unique_ptr<flamegpu::visualiser::ModelVis> _modelVis = nullptr;
#endif  //  FLAMEGPU_VISUALISATION

//. flag indicating if the visualiser was enabled (for cli control so a vis build can  be used without visualisation)
bool _enabled = false;

}  // anon namespace

void setup(bool enabled, flamegpu::ModelDescription& model, flamegpu::CUDASimulation& simulation, bool paused, unsigned simulationSpeed) {
#if defined(FLAMEGPU_VISUALISATION)
    // forward on if vis is enabled. do nothing else if not enabled.
    _enabled = enabled;
    if (!_enabled) {
        return;
    }
    // do nothing if already called
    if (_modelVis != nullptr) {
        return;
    }

    // Otherwise we define the visualisation
    _modelVis = std::make_unique<flamegpu::visualiser::ModelVis>(simulation.getVisualisation());
    // set sim rate and start paused
    _modelVis->setSimulationSpeed(simulationSpeed);
    _modelVis->setBeginPaused(paused);

    // Use orthographic projection for now, currently a 2D model


    // Specify the default camera location
    // @todo - pass in a value which helps define this at runtime. For now hardcoded?
    constexpr float center = 32 / 2.0f;
    _modelVis->setInitialCameraLocation(center, center, center * 2);
    _modelVis->setInitialCameraTarget(center, center, 0);

    _modelVis->setCameraSpeed(0.001f * center);
    _modelVis->setViewClips(0.0001f, 1000);

    // Start in orthographic projection?
    constexpr bool ortho = true;
    if (ortho) {
        _modelVis->setOrthographic(true);
        _modelVis->setOrthographicZoomModifier(0.05f);
    }

    // define how the agents shoudl be visualised
    auto personVis = _modelVis->addAgent(exateppabm::person::NAME);
    personVis.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
    personVis.setModelScale(0.5f);

    // Set the colour to be based on the agents infected status
    // personVis.setColor(flamegpu::visualiser::DiscreteColor(
    //     exateppabm::person::v::INFECTED,
    //     flamegpu::visualiser::Stock::Palette::DARK2,
    //     flamegpu::visualiser::Stock::Colors::WHITE
    // ));

    auto infectedPalette = flamegpu::visualiser::DiscreteColor<exateppabm::disease::SEIR::InfectionStateUnderlyingType>(exateppabm::person::v::INFECTION_STATE, flamegpu::visualiser::Color{ "#0000FF" });
    infectedPalette[disease::SEIR::InfectionState::Susceptible] = "#0000FF";
    infectedPalette[disease::SEIR::InfectionState::Exposed] = "#00FF00";
    infectedPalette[disease::SEIR::InfectionState::Infected] = "#FF0000";
    infectedPalette[disease::SEIR::InfectionState::Recovered] = "#FFFFFF";
    personVis.setColor(infectedPalette);

    // Enable the visualisation
    _modelVis->activate();

    if (paused) {
        fmt::print(stdout, "Press 'p' to start the simulation.\n");
    }
#endif  // FLAMEGPU_VISUALISATION
}

void join() {
#if defined(FLAMEGPU_VISUALISATION)
    // Wait for the visualisation window to be closed if needed
    if (_modelVis && _enabled) {
        _modelVis->join();
    }
#endif  // FLAMEGPU_VISUALISATION
}

void initialiseAgentPopulation(const flamegpu::ModelDescription& model, const exateppabm::input::config config, std::unique_ptr<flamegpu::AgentVector> & pop, const std::uint32_t householdCount) {
#if defined(FLAMEGPU_VISUALISATION)
    // Use the number of households to figure out the size of a 2D grid for visualisation purposes
    std::uint64_t visHouseholdGridwidth = static_cast<std::uint64_t>(std::ceil(std::sqrt(static_cast<double>(householdCount))));
    // Prep a vector of integers to find the location within a household for each individual
    auto visAssignedHouseholdCount = std::vector<std::uint8_t>(householdCount, 0);
    // pre-calculate spatial offset per individual within household, for upto 6 individuals per household (current hardcoded upper limit)
    constexpr float OFFSET_SF = 0.7f;
    std::array<float, 6> visHouseholdOffsetX = {{0.f
        , OFFSET_SF * static_cast<float>(std::sin(0 * M_PI / 180.0))
        , OFFSET_SF * static_cast<float>(std::sin(72 * M_PI / 180.0))
        , OFFSET_SF * static_cast<float>(std::sin(144 * M_PI / 180.0))
        , OFFSET_SF * static_cast<float>(std::sin(216 * M_PI / 180.0))
        , OFFSET_SF * static_cast<float>(std::sin(288 * M_PI / 180.0))}};
    std::array<float, 6> visHouseholdOffsetY = {{0.f
        , OFFSET_SF * static_cast<float>(std::cos(0 * M_PI / 180.0))
        , OFFSET_SF * static_cast<float>(std::cos(72 * M_PI / 180.0))
        , OFFSET_SF * static_cast<float>(std::cos(144 * M_PI / 180.0))
        , OFFSET_SF * static_cast<float>(std::cos(216 * M_PI / 180.0))
        , OFFSET_SF * static_cast<float>(std::cos(288 * M_PI / 180.0))}};

    // Iterate the population, setting the agent's x, y and z values based on their household and index within their household
    unsigned idx = 0;
    for (auto person : *pop) {
        // Get the agent's household index
        auto householdIdx = person.getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
        // Get the center point of their household
        std::uint32_t visHouseholdRow = householdIdx / visHouseholdGridwidth;
        std::uint32_t visHouseholdCol = householdIdx % visHouseholdGridwidth;
        // Get their index within their household [0-5]
        std::uint8_t idxInHouse = visAssignedHouseholdCount[householdIdx];
        visAssignedHouseholdCount[householdIdx]++;
        // Get their arbitrary offset, given a vector of offsets (6 potential values)
        constexpr float VIS_HOUSE_GRID_SPACING = 2.5f;
        float visX = (visHouseholdCol * VIS_HOUSE_GRID_SPACING) + visHouseholdOffsetX[idxInHouse % visHouseholdOffsetX.size()];
        float visY = (visHouseholdRow * VIS_HOUSE_GRID_SPACING) + visHouseholdOffsetY[idxInHouse % visHouseholdOffsetY.size()];
        float visZ = 0;
        // Set the x,y and z in agent data. These must be floats.
        person.setVariable<float>(exateppabm::person::v::x, visX);
        person.setVariable<float>(exateppabm::person::v::y, visY);
        person.setVariable<float>(exateppabm::person::v::z, visZ);
        // Increment the agent index
        ++idx;
    }
#endif  // FLAMEGPU_VISUALISATION
}

}  // namespace visualisation
}  // namespace exateppabm
