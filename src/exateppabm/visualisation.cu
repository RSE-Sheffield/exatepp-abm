#include "exateppabm/visualisation.h"

#include <memory>
#include <fmt/core.h>
#include "flamegpu/flamegpu.h"
#include "person.h"

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
    if (_modelVis != nullptr){
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
    if(ortho) {
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

    auto infectedPalette = flamegpu::visualiser::DiscreteColor<uint32_t>(exateppabm::person::v::INFECTED, flamegpu::visualiser::Color{ "#0000FF" });
    infectedPalette[0] = "#0000FF";
    infectedPalette[1] = "#FF0000";
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

}  // namespace visualisation
}  // namespace exateppabm
