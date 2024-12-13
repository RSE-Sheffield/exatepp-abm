#include "exateppabm/population.h"

#include <fmt/core.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <memory>
#include <random>
#include <array>
#include <vector>

#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/input.h"
#include "exateppabm/household.h"
#include "exateppabm/network.h"
#include "exateppabm/person.h"
#include "exateppabm/util.h"
#include "exateppabm/visualisation.h"
#include "exateppabm/workplace.h"

namespace exateppabm {
namespace population {

namespace {

// File-scoped array  containing the number of infected agents per demographic from population initialisation. This needs to be made accessible to a FLAME GPU Init func due to macro environment property limitations.
std::array<std::uint64_t, demographics::AGE_COUNT> _infectedPerDemographic = {};

// File-scoped copy of the model parameters struct
exateppabm::input::config _params = {};

// File-scoped boolean for if generation should be verbose or not.
bool _verbose = false;

// Store the ID of each individual for each workplace, to form the node labels of the small world network
std::array<std::vector<flamegpu::id_t>, workplace::WORKPLACE_COUNT> workplaceMembers = {{}};

}  // namespace

// Struct used to build per-person data during multi-pass population generation.
// This is required due to bugs within FLAME GPU 2.0.0rc.2, which have since been fixed in by https://github.com/FLAMEGPU/FLAMEGPU2/pull/1270, so can be resolved in the future.
struct HostPerson {
    flamegpu::id_t id = flamegpu::ID_NOT_SET;
    disease::SEIR::InfectionStateUnderlyingType infectionStatus;
    float infectionStateDuration = 1.f;
    std::uint32_t infectionCount = 0;
    demographics::AgeUnderlyingType ageDemographic;
    std::uint32_t householdIdx = 0;
    std::uint8_t householdSize = 0;
    workplace::WorkplaceUnderlyingType workplaceIdx = 0;
    std::uint32_t workplaceOutDegree = 0;
    std::uint32_t randomInteractionTarget = 0;
};

/**
 * FLAME GPU init function which generates a population of person agents for the current simulation
 *
 * This function has been split into multiple init functions, due to errors encountered when attempting to do multiple passes over a newly created set of agents. There should be a working way to do this...
 */
FLAMEGPU_INIT_FUNCTION(generatePopulation) {
    fmt::print("@todo - validate params inputs when generated agents (pop size, initial infected count etc)\n");
    // n_total must be less than uint max
    // n_total must be more than 0.
    // initial infected count must be more than 0, less than full pop.

    // Get a handle on the environment.
    auto env = FLAMEGPU->environment;

    // Get a handle to the host agent api for Person agents
    // auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::DEFAULT);

    // Do nothing if no one to create?
    if (_params.n_total == 0) {
        return;
    }

    // Workaround: Generating agents in FLAME GPU 2 in init functions does not allow iterating the data structure multiple times, this appears to be a bug (which may have a fix in place, currently untested pr)

    // Instead, we must generate data on the host for the "first pass", storing it in non-flame gpu storage
    // Then set it on the latter pass.
    std::vector<HostPerson> hostPersonData;
    hostPersonData.resize(_params.n_total);

    // seed host rng for population generation.
    // @todo - does this want to be a separate seed from the config file?
    std::mt19937_64 rng(FLAMEGPU->getSimulationConfig().random_seed);

    // Need to initialise a fixed number of individuals as infected.
    // This not very scalable way of doing it, is to create a vector with one element per individual in the simulation, initialised to false
    // set the first n_seed_infection elements to true/1
    // Shuffle the vector,  and query at agent creation time
    // RNG sampling in-loop would be more memory efficient, but harder to guarantee that exactly enough are created. This will likely be replaced anyway, so quick and dirty is fine.
    std::vector<bool> infected_vector(_params.n_total);
    std::fill(infected_vector.begin(), infected_vector.begin() + std::min(_params.n_total, _params.n_seed_infection), true);
    std::shuffle(infected_vector.begin(), infected_vector.end(), rng);

    // Get the number of individuals per house and their age demographics
    auto households = household::generateHouseholdStructures(_params, rng, _verbose);

    // per demo total is not an output in time series.
    // Alternately, we need to initialise the exact number of each age band, not RNG, and just scale it down accordingly. Will look at in "realistic" population generation
    std::array<std::uint64_t, demographics::AGE_COUNT> createdPerDemographic = {};
    for (const auto & household : households) {
        for (demographics::AgeUnderlyingType i = 0; i < demographics::AGE_COUNT; i++) {
            createdPerDemographic[i] += household.sizePerAge[i];
        }
    }

    // reset per demographic count of the number initialised agents in each infection state.
    // This is used for the initial value in time series data, without having to iterate all agent data again, but may not be thread safe (i.e. probably need to change for ensembles)
    _infectedPerDemographic = {{0, 0, 0, 0, 0, 0, 0, 0, 0}};

    // Given the household and ages are known, we can compute the workplace assignment probabilities for adults
    auto p_adultPerWorkNetwork = workplace::getAdultWorkplaceCumulativeProbabilityArray(_params.child_network_adults, _params.elderly_network_adults, createdPerDemographic);
    // // Store the ID of each individual for each workplace, to form the node labels of the small world network
    // std::array<std::vector<flamegpu::id_t>, workplace::WORKPLACE_COUNT> workplaceMembers = {{}};

    // Counter for random interaction count. This will be 2x the number of interactions
    std::uint64_t randomInteractionCountSum = 0u;
    // Max/min trackers for random interaction targets, for verbose output.
    std::uint32_t randomInteractionMax = 0u;
    std::uint32_t randomInteractionMin = std::numeric_limits<std::uint32_t>::max();

    // Populate agent data, by iterating households
    std::uint32_t personIdx = 0;
    for (std::uint32_t householdIdx = 0; householdIdx < households.size(); householdIdx++) {
        auto household = households.at(householdIdx);
        // For each individual in the household
        for (household::HouseholdSizeType householdMemberIdx = 0; householdMemberIdx < household.size; householdMemberIdx++) {
            // assert that the household structure is complete
            assert(household.size == household.agePerPerson.size());

            // Get the flamegpu person object for the individual
            // auto person = personAgent.newAgent();  @temp disabled due to bug/workaround in place for multi-pass

            // Set the manual ID, as getID() isn't guaranteed to actually start at 1.
            flamegpu::id_t personID = personIdx + 1;
            // person.setVariable<flamegpu::id_t>(person::v::ID, personID);
            hostPersonData[personIdx].id = personID;

            // Set the individuals infection status. @todo - refactor into seir.cu?
            disease::SEIR::InfectionState infectionStatus = infected_vector.at(personIdx) ? disease::SEIR::InfectionState::Infected : disease::SEIR::InfectionState::Susceptible;
            // person.setVariable<disease::SEIR::InfectionStateUnderlyingType>(exateppabm::person::v::INFECTION_STATE, infectionStatus);
            hostPersonData[personIdx].infectionStatus = infectionStatus;
            // Also set the initial infection duration. @todo - stochastic.
            float infectionStateDuration = infectionStatus == disease::SEIR::InfectionState::Infected ? _params.mean_time_to_recovered: 0;
            // person.setVariable<float>(exateppabm::person::v::INFECTION_STATE_DURATION, infectionStateDuration);
            hostPersonData[personIdx].infectionStateDuration = infectionStateDuration;

            // Set the individuals age and household properties
            demographics::Age age = household.agePerPerson[householdMemberIdx];
            // person.setVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, static_cast<demographics::AgeUnderlyingType>(age));
            hostPersonData[personIdx].ageDemographic = static_cast<demographics::AgeUnderlyingType>(age);
            // person.setVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX, householdIdx);
            hostPersonData[personIdx].householdIdx = householdIdx;
            // person.setVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE, household.size);
            hostPersonData[personIdx].householdSize = household.size;

            // initialise the agents infection count
            if (infectionStatus == disease::SEIR::Infected) {
                // person.setVariable<std::uint32_t>(exateppabm::person::v::INFECTION_COUNT, 1u);
                hostPersonData[personIdx].infectionCount = 1u;
                // Increment the per-age demographic initial agent count. @todo refactor elsewhere?
                _infectedPerDemographic[age]++;
            }

            // Assign the workplace based on age band
            auto workplaceIdx = workplace::generateWorkplaceForIndividual(age, p_adultPerWorkNetwork, rng);

            // Store the agent's manually set ID, cannot rely on autogenerated to be useful indices
            workplaceMembers[workplaceIdx].push_back(personID);
            // Store the assigned network in the agent data structure
            // person.setVariable<workplace::WorkplaceUnderlyingType>(person::v::WORKPLACE_IDX, workplaceIdx);
            hostPersonData[personIdx].workplaceIdx = workplaceIdx;

            // Generate the (target) number of random interactions this individual will be involved in per day. Not all combinations will be possible on all days, hence target.

            // @todo - optionally allow binomial distributions
            // @todo - decide if non non-binomial should be a mean or not, maybe allow non fixed normal dists?
            // @todo - refactor and test.
            double meanRandomInteractions = _params.mean_random_interactions_20_69;
            double sdRandomInteractions = _params.sd_random_interactions_20_69;
            if (age == demographics::Age::AGE_0_9 || age == demographics::Age::AGE_10_19) {
                meanRandomInteractions = _params.mean_random_interactions_0_19;
                sdRandomInteractions = _params.sd_random_interactions_0_19;
            } else if (age == demographics::Age::AGE_70_79 || age == demographics::Age::AGE_80) {
                meanRandomInteractions = _params.mean_random_interactions_70plus;
                sdRandomInteractions = _params.sd_random_interactions_70plus;
            }

            // Sample a normal distribution (of integers, so we can clamp to >= 0)
            std::normal_distribution<double> randomInteractionDist{meanRandomInteractions, sdRandomInteractions};
            // Sample from the distribution
            double randomInteractionsRaw = randomInteractionDist(rng);
            // Clamp to be between 0 and the population size, and cast to uint
            std::uint32_t randomInteractionTarget = static_cast<std::uint32_t>(std::clamp(randomInteractionsRaw, 0.0, static_cast<double>(_params.n_total)));

            // If the max was over the compile time upper limit due to flamegpu limitations, emit a warning and exit.
            if (randomInteractionTarget > person::MAX_RANDOM_DAILY_INTERACTIONS) {
                fmt::print(stderr, "Fatal Error: Random Interaction Target {} exceeds fixed limit MAX_RANDOM_DAILY_INTERACTIONS {}. Please rebuild with a higher value @todo\n", randomInteractionTarget, person::MAX_RANDOM_DAILY_INTERACTIONS);
                exit(EXIT_FAILURE);
            }

            // Update the min and max tracking for output to stdout
            randomInteractionMin = std::min(randomInteractionMin, randomInteractionTarget);
            randomInteractionMax = std::max(randomInteractionMax, randomInteractionTarget);
            // Set for the agent
            // person.setVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT_TARGET, randomInteractionTarget);
            hostPersonData[personIdx].randomInteractionTarget = randomInteractionTarget;
            // Track the sum of target interaction counts
            randomInteractionCountSum += randomInteractionTarget;

            // Increment the person index
            ++personIdx;
        }
    }

    if (_verbose) {
        fmt::print("Random Interactions: min={}, max={}, sum={}\n", randomInteractionMin, randomInteractionMax, randomInteractionCountSum);
    }

    // Set the sum of per agent target interaction counts. This is not the total number of interactions that will occur, as not all configurations are possible.
    // This only works if this method is called prior to simulation construction!
    env.setProperty<std::uint64_t>("RANDOM_INTERACTION_COUNT_SUM", randomInteractionCountSum);
    // Warn if the number target number of interactions is odd.
    if (randomInteractionCountSum % 2 == 1) {
        fmt::print(stderr, "Warning: Total the sum of per-agent random interactions is odd ({})\n", randomInteractionCountSum);
    }

    // Now individuals have been assigned to each workplace network, we can generate the small world networks for workplace interactions
    std::array<double, workplace::WORKPLACE_COUNT> meanInteractionsPerNetwork = {{
        _params.mean_work_interactions_child / _params.daily_fraction_work,
        _params.mean_work_interactions_child / _params.daily_fraction_work,
        _params.mean_work_interactions_adult / _params.daily_fraction_work,
        _params.mean_work_interactions_elderly / _params.daily_fraction_work,
        _params.mean_work_interactions_elderly / _params.daily_fraction_work}};

    // agent count + 1 elements, as 1 indexed
    std::uint32_t totalVertices = _params.n_total;
    // Count the total number of edges
    std::uint32_t totalUndirectedEdges = 0u;
    std::array<network::UndirectedGraph, workplace::WORKPLACE_COUNT> workplaceNetworks = {};
    for (workplace::WorkplaceUnderlyingType widx = 0u; widx < workplace::WORKPLACE_COUNT; ++widx) {
        const std::uint64_t N = workplaceMembers[widx].size();
        // Round to nearest even number. This might want to be always round up instead @todo
        std::uint32_t meanInteractions = static_cast<std::uint32_t>(std::nearbyint(meanInteractionsPerNetwork[widx] / 2.0) * 2.0);
        // @todo - test / cleaner handling of different N and K values.
        if (meanInteractions >= N) {
            // @todo - this feels brittle.
            meanInteractions = N - 1;
        }

        // Generate the small world network, unless N is too small, or meanInteractions is too small
        if (N > 2 && meanInteractions > 1) {
            // Shuffle the network, to avoid people working with others in their same household a disproportionate amount for low values of work_network_rewire
            std::shuffle(workplaceMembers[widx].begin(), workplaceMembers[widx].end(), rng);
            // Generate the small world network
            workplaceNetworks[widx] = network::generateSmallWorldUndirectedGraph(workplaceMembers[widx], meanInteractions, _params.work_network_rewire, rng);
        } else {
            workplaceNetworks[widx] = network::generateFullyConnectedUndirectedGraph(workplaceMembers[widx]);
        }

        totalUndirectedEdges += workplaceNetworks[widx].getNumEdges();
    }

    // Get a handle to the FLAME GPU workplace directed graph
    flamegpu::HostEnvironmentDirectedGraph workplaceDigraph = FLAMEGPU->environment.getDirectedGraph("WORKPLACE_DIGRAPH");
    // This will contain all workplace information, i.e. a single graph data structure containing 5 unconnected graphs
    // The custom 1-indexed agent's ID will be used as the vertex keys
    // This will then allow agents to simply lookup their neighbours and
    // FLAME GPU's graphs are accessed by their string name, but agents can't have string properties, hence using a single graph.
    // Representing a undirected graph using a directed graph will consume twice as much device memory as required, but only digraphs are currently implemented in FLAME GPU 2.
    std::uint32_t totalDirectedEdges = 2 * totalUndirectedEdges;
    workplaceDigraph.setVertexCount(totalVertices);
    workplaceDigraph.setEdgeCount(totalDirectedEdges);

    // Set vertex properties
    flamegpu::HostEnvironmentDirectedGraph::VertexMap vertices = workplaceDigraph.vertices();
    for (std::uint32_t v = 1; v <= totalVertices; ++v) {
        flamegpu::HostEnvironmentDirectedGraph::VertexMap::Vertex vertex = vertices[v];
        // vertex.setProperty<flamegpu::id_t>(person::v::ID, v);
    }

    // Iterate each workplace network, adding it's both directions of each undirected edge to the graph
    flamegpu::HostEnvironmentDirectedGraph::EdgeMap edges = workplaceDigraph.edges();
    for (auto& undirectedNetwork : workplaceNetworks) {
        // For each undirected edge which contains indexes within the network, create the 2 directed edges between agent IDs
        for (const auto & undirectedEdge : undirectedNetwork.getEdges()) {
            flamegpu::id_t a = undirectedNetwork.getVertexLabel(undirectedEdge.source);
            flamegpu::id_t b = undirectedNetwork.getVertexLabel(undirectedEdge.dest);
            auto abEdge = edges[{a, b}];
            abEdge.setSourceDestinationVertexID(a, b);
            auto baEdge = edges[{b, a}];
            baEdge.setSourceDestinationVertexID(b, a);
        }
    }

    // flamegpu::DeviceAgentVector population = personAgent.getPopulationData();
    // Update the population with additional data.
    // This is (potentially) suboptimal in terms of host-device memcpy, but need to pass agents multiple times unfortunately.

    // flamegpu::DeviceAgentVector pop = FLAMEGPU->agent("person", "default").getPopulationData();
    // std::uint32_t personIdx = 0;
    // for (auto person : pop) {
    for (std::uint32_t personIdx = 0; personIdx < hostPersonData.size(); ++personIdx) {
        // Get the agents id
        // std::uint32_t agentId = person.getVariable<flamegpu::id_t>(person::v::ID);
        std::uint32_t agentId = hostPersonData[personIdx].id;
        // get the assigned workplace
        // workplace::WorkplaceUnderlyingType workplaceIdx = person.getVariable<workplace::WorkplaceUnderlyingType>(person::v::WORKPLACE_IDX);
        auto workplaceIdx = hostPersonData[personIdx].workplaceIdx;
        // For this individual in their small world network, get the (out)degree, i.e. the max number of interactions per day.
        network::UndirectedGraph& workplaceGraph = workplaceNetworks[workplaceIdx];
        auto vertexLabels = workplaceGraph.getVertexLabels();
        auto it = std::find(vertexLabels.begin(), vertexLabels.end(), agentId);
        if (it != vertexLabels.end()) {
            std::uint32_t vertexIdx = std::distance(vertexLabels.begin(), it);
            // Get the outdegree for this vertex index
            std::uint32_t degree = workplaceGraph.degree(vertexIdx);
            // person.setVariable<std::uint32_t>(person::v::WORKPLACE_OUT_DEGREE, degree);
            hostPersonData[personIdx].workplaceOutDegree = degree;
        } else {
            throw std::runtime_error("@todo - could not find agent in workplace");
        }
        // ++personIdx;
    }

    // Finally, in another pass create the actual agent instances, finishing the multi-pass init function (and split init function workaround)
    auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::DEFAULT);
    for (std::uint32_t personIdx = 0; personIdx < hostPersonData.size(); ++personIdx) {
        const auto& hostPerson = hostPersonData[personIdx];
        // Generate the new agent
        auto person = personAgent.newAgent();
        // Set each property on the agent.
        person.setVariable<flamegpu::id_t>(person::v::ID, hostPerson.id);
        person.setVariable<disease::SEIR::InfectionStateUnderlyingType>(exateppabm::person::v::INFECTION_STATE, hostPerson.infectionStatus);
        person.setVariable<float>(exateppabm::person::v::INFECTION_STATE_DURATION, hostPerson.infectionStateDuration);
        person.setVariable<std::uint32_t>(exateppabm::person::v::INFECTION_COUNT, hostPerson.infectionCount);
        person.setVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, hostPerson.ageDemographic);
        person.setVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX, hostPerson.householdIdx);
        person.setVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE, hostPerson.householdSize);
        person.setVariable<workplace::WorkplaceUnderlyingType>(person::v::WORKPLACE_IDX, hostPerson.workplaceIdx);
        person.setVariable<std::uint32_t>(person::v::WORKPLACE_OUT_DEGREE, hostPerson.workplaceOutDegree);
        person.setVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT_TARGET, hostPerson.randomInteractionTarget);


        // If this is a visualisation enabled build, set their x/y/z
#if defined(FLAMEGPU_VISUALISATION)
        auto [visX, visY, visZ] = exateppabm::visualisation::getAgentXYZ(static_cast<std::uint32_t>(households.size()), hostPerson.householdIdx, 0);
        person.setVariable<float>(exateppabm::person::v::x, visX);
        person.setVariable<float>(exateppabm::person::v::y, visY);
        person.setVariable<float>(exateppabm::person::v::z, visZ);
#endif  // defined(FLAMEGPU_VISUALISATION)
    }

    // Do the verbose output
    if (_verbose) {
        // Print a summary of population creation for now.
        fmt::print("Created {} people with {} infected.\n", _params.n_total, _params.n_seed_infection);
        fmt::print("Households: {}\n", households.size());
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
        fmt::print("Workplaces {{\n");
        fmt::print("  00-09: {}, 20_69 {}\n", createdPerDemographic[0],  workplaceMembers[workplace::Workplace::WORKPLACE_SCHOOL_0_9].size() - createdPerDemographic[demographics::Age::AGE_0_9]);
        fmt::print("  10-19: {}, 20_69 {}\n", createdPerDemographic[demographics::Age::AGE_10_19],  workplaceMembers[workplace::Workplace::WORKPLACE_SCHOOL_0_9].size() - createdPerDemographic[demographics::Age::AGE_10_19]);
        fmt::print("  20_69 {}\n", workplaceMembers[workplace::Workplace::WORKPLACE_ADULT].size());
        fmt::print("  20_69 {}, 70_79 {}\n", workplaceMembers[workplace::Workplace::WORKPLACE_70_79].size() - createdPerDemographic[demographics::Age::AGE_70_79], createdPerDemographic[demographics::Age::AGE_70_79]);
        fmt::print("  20_69 {}, 80+ {}\n", workplaceMembers[workplace::Workplace::WORKPLACE_80_PLUS].size() -  - createdPerDemographic[demographics::Age::AGE_80], createdPerDemographic[demographics::Age::AGE_80]);
        fmt::print("}}\n");
    }
}

void define(flamegpu::ModelDescription& model, const exateppabm::input::config params, const bool verbose) {
    // Store passed in parameters in file-scoped variables
    _params = params;
    _verbose = verbose;
    // Define the init function which will generate the population for the parameters struct stored in the anon namespace
    model.addInitFunction(generatePopulation);
}

std::array<std::uint64_t, demographics::AGE_COUNT> getPerDemographicInitialInfectionCount() {
    return _infectedPerDemographic;
}

}  // namespace population
}  // namespace exateppabm
