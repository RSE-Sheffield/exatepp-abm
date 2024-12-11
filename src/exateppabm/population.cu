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
#include "exateppabm/network.h"
#include "exateppabm/person.h"
#include "exateppabm/util.h"
#include "exateppabm/visualisation.h"

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
std::array<std::vector<flamegpu::id_t>, WORKPLACE_COUNT> workplaceMembers = {{}};

}  // namespace


// Workaround struct representing a person used during host generation
struct HostPerson {
    flamegpu::id_t id;
    disease::SEIR::InfectionStateUnderlyingType infectionStatus;
    float infectionStateDuration;
    std::uint32_t infectionCount = 0;
    demographics::AgeUnderlyingType ageDemographic;
    std::uint32_t householdIdx;
    std::uint8_t householdSize;
    std::uint32_t workplaceIdx;
    std::uint32_t randomInteractionTarget;
    std::uint32_t workplaceOutDegree;
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
    auto households = generateHouseholdStructures(_params, rng, _verbose);

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
    std::array<double, WORKPLACE_COUNT> p_adultPerWorkNetwork = getAdultWorkplaceCumulativeProbabilityArray(_params.child_network_adults, _params.elderly_network_adults, createdPerDemographic);
    // // Store the ID of each individual for each workplace, to form the node labels of the small world network
    // std::array<std::vector<flamegpu::id_t>, WORKPLACE_COUNT> workplaceMembers = {{}};

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
        for (population::HouseholdSizeType householdMemberIdx = 0; householdMemberIdx < household.size; householdMemberIdx++) {
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
            WorkplaceUnderlyingType workplaceIdx = generateWorkplaceForIndividual(age, p_adultPerWorkNetwork, rng);

            // Store the agent's manually set ID, cannot rely on autogenerated to be useful indices
            workplaceMembers[workplaceIdx].push_back(personID);
            // Store the assigned network in the agent data structure
            // person.setVariable<std::uint32_t>(person::v::WORKPLACE_IDX, workplaceIdx);
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
    std::array<double, population::WORKPLACE_COUNT> meanInteractionsPerNetwork = {{
        _params.mean_work_interactions_child / _params.daily_fraction_work,
        _params.mean_work_interactions_child / _params.daily_fraction_work,
        _params.mean_work_interactions_adult / _params.daily_fraction_work,
        _params.mean_work_interactions_elderly / _params.daily_fraction_work,
        _params.mean_work_interactions_elderly / _params.daily_fraction_work}};

    // agent count + 1 elements, as 1 indexed
    std::uint32_t totalVertices = _params.n_total;
    // Count the total number of edges
    std::uint32_t totalUndirectedEdges = 0u;
    std::array<network::UndirectedGraph, population::WORKPLACE_COUNT> workplaceNetworks = {};
    for (population::WorkplaceUnderlyingType widx = 0; widx < population::WORKPLACE_COUNT; ++widx) {
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
        // std::uint32_t workplaceIdx = person.getVariable<std::uint32_t>(person::v::WORKPLACE_IDX);
        std::uint32_t workplaceIdx = hostPersonData[personIdx].workplaceIdx;
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
        person.setVariable<std::uint32_t>(person::v::WORKPLACE_IDX, hostPerson.workplaceIdx);
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
        fmt::print("  00-09: {}, 20_69 {}\n", createdPerDemographic[0],  workplaceMembers[Workplace::WORKPLACE_SCHOOL_0_9].size() - createdPerDemographic[demographics::Age::AGE_0_9]);
        fmt::print("  10-19: {}, 20_69 {}\n", createdPerDemographic[demographics::Age::AGE_10_19],  workplaceMembers[Workplace::WORKPLACE_SCHOOL_0_9].size() - createdPerDemographic[demographics::Age::AGE_10_19]);
        fmt::print("  20_69 {}\n", workplaceMembers[Workplace::WORKPLACE_ADULT].size());
        fmt::print("  20_69 {}, 70_79 {}\n", workplaceMembers[Workplace::WORKPLACE_70_79].size() - createdPerDemographic[demographics::Age::AGE_70_79], createdPerDemographic[demographics::Age::AGE_70_79]);
        fmt::print("  20_69 {}, 80+ {}\n", workplaceMembers[Workplace::WORKPLACE_80_PLUS].size() -  - createdPerDemographic[demographics::Age::AGE_80], createdPerDemographic[demographics::Age::AGE_80]);
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

double getReferenceMeanHouseholdSize(const exateppabm::input::config& params) {
    std::vector<std::uint64_t> countPerSize = {{params.household_size_1, params.household_size_2, params.household_size_3, params.household_size_4, params.household_size_5, params.household_size_6}};
    std::uint64_t refPeople = 0u;
    std::uint64_t refHouses = 0u;
    for (std::size_t idx = 0; idx < countPerSize.size(); idx++) {
        refPeople += (idx + 1) * countPerSize[idx];
        refHouses += countPerSize[idx];
    }
    double refMeanHouseholdSize = refPeople / static_cast<double>(refHouses);
    return refMeanHouseholdSize;
}

std::vector<double> getHouseholdSizeCumulativeProbabilityVector(const exateppabm::input::config& params) {
    // Initialise vector with each config household size
    std::vector<std::uint64_t> countPerSize = {{params.household_size_1, params.household_size_2, params.household_size_3, params.household_size_4, params.household_size_5, params.household_size_6}};
    // get the sum, to find relative proportions
    std::uint64_t sumConfigHouseholdSizes = exateppabm::util::reduce(countPerSize.begin(), countPerSize.end(), 0ull);
    // Get the number of people in each household band for the reference size
    // Find the number of people that the reference household sizes can account for
    std::vector<std::uint64_t> peoplePerHouseSize = countPerSize;
    for (std::size_t idx = 0; idx < peoplePerHouseSize.size(); idx++) {
        peoplePerHouseSize[idx] = (idx + 1) * peoplePerHouseSize[idx];
    }
    std::uint64_t sumConfigPeoplePerHouseSize = exateppabm::util::reduce(peoplePerHouseSize.begin(), peoplePerHouseSize.end(), 0ull);
    double configMeanPeoplePerHouseSize = sumConfigPeoplePerHouseSize / static_cast<double>(sumConfigHouseholdSizes);

    // Build a list of household sizes, by random sampling from a uniform distribution using probabilities from the reference house size counts.
    std::vector<double> householdSizeProbability(countPerSize.size());
    for (std::size_t idx = 0; idx < householdSizeProbability.size(); idx++) {
        householdSizeProbability[idx] = countPerSize[idx] / static_cast<double>(sumConfigHouseholdSizes);
    }
    // Perform an inclusive scan to convert to cumulative probability
    exateppabm::util::inclusive_scan(householdSizeProbability.begin(), householdSizeProbability.end(), householdSizeProbability.begin());

    return householdSizeProbability;
}

std::vector<HouseholdStructure> generateHouseholdStructures(const exateppabm::input::config params, std::mt19937_64 & rng, const bool verbose) {
    /*
    @todo This method will want refactoring for realistic household generation.
    Current structure is:
    1. Get the cumulative probability distribution of house sizes based on reference data
    2. Generate household sizes randomly, using based on reference house size data
    3. For each house, generate the age per person within the household using probabilities based on global age demographic target data
    */

    // Get the vector of household size cumulative probability
    auto householdSizeProbabilityVector =  getHouseholdSizeCumulativeProbabilityVector(params);

    // Get the array of age demographic cumulative probability and reverse enum map
    auto ageDemographicProbabilities = demographics::getAgeDemographicCumulativeProbabilityArray(params);
    auto allAgeDemographics = demographics::getAllAgeDemographics();


    // Specify the rng distribution to sample from, [0, 1.0)
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // initialise a value with the number of people left to generate
    std::int64_t remainingPeople = static_cast<std::int64_t>(params.n_total);
    // estimate the number of houses, based on the cumulative probability distribution
    double refMeanHouseSize = getReferenceMeanHouseholdSize(params);
    std::uint64_t householdCountEstimate = static_cast<std::uint64_t>(std::ceil(remainingPeople / refMeanHouseSize));

    // Create a vector of house structures, and reserve enough room for the estimated number of houses
    std::vector<HouseholdStructure> households = {};
    households.reserve(householdCountEstimate);

    // create enough households for the whole population using the uniform distribution and cumulative probability vector. Ensure the last household is not too large.
    while (remainingPeople > 0) {
        double r_houseSize = dist(rng);
        for (std::size_t idx = 0; idx < static_cast<HouseholdSizeType>(householdSizeProbabilityVector.size()); idx++) {
            if (r_houseSize < householdSizeProbabilityVector[idx]) {
                HouseholdStructure household = {};
                household.size = static_cast<HouseholdSizeType>(idx + 1) <= remainingPeople ? static_cast<HouseholdSizeType>(idx + 1) : remainingPeople;
                household.agePerPerson.reserve(household.size);
                // Generate ages for members of the household
                for (HouseholdSizeType pidx = 0; pidx < household.size; ++pidx) {
                    float r_age = dist(rng);
                    // @todo - abstract this into a method.
                    demographics::Age age = demographics::Age::AGE_0_9;
                    for (demographics::AgeUnderlyingType i = 0; i < demographics::AGE_COUNT; i++) {
                        if (r_age < ageDemographicProbabilities[i]) {
                            age = allAgeDemographics[i];
                            break;
                        }
                    }
                    household.agePerPerson.push_back(age);
                    household.sizePerAge[static_cast<demographics::AgeUnderlyingType>(age)]++;
                }
                households.push_back(household);
                remainingPeople -= household.size;
                break;
            }
        }
    }
    // potentially shrink the vector, in case the reservation was too large
    households.shrink_to_fit();

    if (verbose) {
        // Get the count of created per house size and print it.
        std::vector<std::uint64_t> generatedHouseSizeDistribution(householdSizeProbabilityVector.size());
        for (const auto& household : households) {
            generatedHouseSizeDistribution[household.size-1]++;
        }
        fmt::print("generated households per household size (total {}) {{\n", households.size());
        for (const auto & v : generatedHouseSizeDistribution) {
            fmt::print("  {},\n", v);
        }
        fmt::print("}}\n");
        // Sum the number of people per household
        std::uint64_t sumPeoplePerHouse = std::accumulate(households.begin(), households.end(), 0ull, [](std::uint64_t tot, HouseholdStructure& h) {return tot + h.size;});
        // std::uint64_t sumPeoplePerHouse = exateppabm::util::reduce(households.begin(), households.end(), 0ull);
        // Check the mean still agrees.
        double generatedMeanPeoplePerHouseSize = sumPeoplePerHouse / static_cast<double>(households.size());
        fmt::print("generated mean household size {}\n", generatedMeanPeoplePerHouseSize);
    }
    return households;
}

std::array<double, WORKPLACE_COUNT> getAdultWorkplaceCumulativeProbabilityArray(const double child_network_adults, const double elderly_network_adults, std::array<std::uint64_t, demographics::AGE_COUNT> n_per_age) {
    // Adults are assigned to a work place network randomly, using a probability distribution which (for a large enough sample) will match the target ratios of adults to other workplace members, based on the child_network_adults and elderly_network_adults parameters
    std::uint64_t n_0_9 = n_per_age[demographics::Age::AGE_0_9];
    std::uint64_t n_10_19 = n_per_age[demographics::Age::AGE_10_19];
    std::uint64_t n_70_79 = n_per_age[demographics::Age::AGE_70_79];
    std::uint64_t n_80_plus = n_per_age[demographics::Age::AGE_80];
    std::uint64_t n_adult = n_per_age[demographics::Age::AGE_20_29] + n_per_age[demographics::Age::AGE_30_39] + n_per_age[demographics::Age::AGE_40_49] + n_per_age[demographics::Age::AGE_50_59] + n_per_age[demographics::Age::AGE_60_69];

    // Initialise with the target number of adults in each network
    std::array<double, WORKPLACE_COUNT> p_adultPerWorkNetwork = {0};
    // p of each category is target number / number adults
    p_adultPerWorkNetwork[Workplace::WORKPLACE_SCHOOL_0_9] = (child_network_adults * n_0_9) / n_adult;
    p_adultPerWorkNetwork[Workplace::WORKPLACE_SCHOOL_10_19] = (child_network_adults * n_10_19) / n_adult;
    p_adultPerWorkNetwork[Workplace::WORKPLACE_70_79] = (elderly_network_adults * n_70_79) / n_adult;
    p_adultPerWorkNetwork[Workplace::WORKPLACE_80_PLUS] = (elderly_network_adults * n_80_plus) / n_adult;

    // p of being adult is then the remaining probability
    p_adultPerWorkNetwork[Workplace::WORKPLACE_ADULT] = 1.0 - std::accumulate(p_adultPerWorkNetwork.begin(), p_adultPerWorkNetwork.end(), 0.0);

    // Then convert to cumulative probability with an inclusive scan
    exateppabm::util::inclusive_scan(p_adultPerWorkNetwork.begin(), p_adultPerWorkNetwork.end(), p_adultPerWorkNetwork.begin());

    // make sure the top bracket ends in a value in case of floating point / rounding >= 1.0
    p_adultPerWorkNetwork[Workplace::WORKPLACE_80_PLUS] = 1.0;

    return p_adultPerWorkNetwork;
}

WorkplaceUnderlyingType generateWorkplaceForIndividual(const demographics::Age age, std::array<double, WORKPLACE_COUNT> p_adult_workplace, std::mt19937_64 & rng) {
    // Children, retired and elderly are assigned a network based on their age
    if (age == demographics::Age::AGE_0_9) {
        return Workplace::WORKPLACE_SCHOOL_0_9;
    } else if (age == demographics::Age::AGE_10_19) {
        return Workplace::WORKPLACE_SCHOOL_10_19;
    } else if (age == demographics::Age::AGE_70_79) {
        return Workplace::WORKPLACE_70_79;
    } else if (age == demographics::Age::AGE_80) {
        return Workplace::WORKPLACE_80_PLUS;
    } else {
        // Adults randomly sample using a cumulative probability distribution computed from population counts and model parameters
        std::uniform_real_distribution<double> work_network_dist(0.0, 1.0);
        float r = work_network_dist(rng);
        for (std::uint32_t i = 0; i < p_adult_workplace.size(); i++) {
            if (r < p_adult_workplace[i]) {
                return i;
            }
        }
        throw std::runtime_error("@todo - invalid cumulative probability distribution for p_adult_workplace?");
    }
}

}  // namespace population
}  // namespace exateppabm
