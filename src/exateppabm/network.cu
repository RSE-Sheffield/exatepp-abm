#include "exateppabm/network.h"

#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <random>

#include "flamegpu/flamegpu.h"
#include "fmt/core.h"

namespace exateppabm {
namespace network {


UndirectedNetwork generateFullyConnectedUndirectedNetwork(std::vector<flamegpu::id_t> nodes) {
    UndirectedNetwork network = {};
    network.nodes = nodes;
    if (nodes.size() <= 1) {
        return network;
    }
    std::uint32_t N = static_cast<std::int32_t>(nodes.size());
    std::uint32_t E = (N * (N - 1)) / 2;
    network.edges.reserve(E);
    // Generate edges in ascending order for the undirected graph.
    for (std::uint32_t i = 0; i < N; ++i) {
        for (std::uint32_t j = i + 1; j < N; ++j) {
            network.edges.push_back({i, j});
        }
    }
    return network;
}

// @todo - should the initial lattice be randomised rather than sequential?
// @todo - should the graph be directed or undirected? Data structure is directed, so do pairs need rewiring together?
UndirectedNetwork generateSmallWorldUndirectedNetwork(std::vector<flamegpu::id_t> nodes, std::uint32_t K, double p_rewire, std::mt19937_64 rng) {
    // fmt::print("@todo - check if small world network should be directed or undirected\n");
    UndirectedNetwork network = {};
    network.nodes = nodes;

    // Return early with an empty network if 0 or 1 nodes, so no room for non-self edges
    if (network.nodes.size() <= 1) {
        return network;
    }

    // If K is 0 (or 1), no edges so do nothing
    if (K <= 1) {
        // throw std::runtime_error("@todo values - small world network K (" + std::to_string(K) + ") must be > 1");
        return network;
    } else if (K == network.nodes.size()) {
        // If K == Node count, the graph is fully connected, so return a fully connected graph
        return generateFullyConnectedUndirectedNetwork(nodes);
    } else if (K >= network.nodes.size()) {
        // Raise an exception if K is too large
        throw std::runtime_error(std::string("@todo values - small world network K(") + std::to_string(K) + ") must be less than |N| (" + std::to_string(network.nodes.size()) + ")");
    }
    // If K is odd, use K-1
    if (K % 2 == 1) {
        K = K - 1;
    }

    // p_rewire must be between 0 and 1.
    if (p_rewire < 0 || p_rewire > 1) {
        throw std::runtime_error("generateSmallWorldNetwork p must be in [0, 1]. @todo include value");
    }

    // Initialise the edges as a lattice network, going K edges in each direction.
    // Use signed integers to make modulo possible
    std::int32_t N = static_cast<std::int32_t>(nodes.size());
    std::int32_t E = (N * K) / 2;
    network.edges.reserve(E);
    for (std::int32_t i = 0; i < N; ++i) {
        for (std::int32_t j = 1; j <= static_cast<std::int32_t>(K / 2); ++j) {
            // only add the positive undirected edges
            std::uint32_t s = static_cast<std::uint32_t>(i);
            std::uint32_t d = static_cast<std::uint32_t>((i + j) % N);
            // Ensure that the edge source node is lower than the destination, to simplify avoiding duplicate edges
            if (s > d) {
                std::swap(s, d);
            }
            network.edges.push_back({s, d});
            // network.edges.push_back({static_cast<std::uint32_t>(i), static_cast<std::uint32_t>((i + j) % N)});
            // network.edges.push_back({static_cast<std::uint32_t>(i), static_cast<std::uint32_t>((i - j + N) % N)});
        }
    }

    // If the network is fully connected (enough edges for each node to be connected to every other node), rewiring is not needed, so adjust the probability to prevent infinite loops.
    std::int32_t MAX_E = (N * (N - 1)) / 2;
    if (E >= MAX_E) {
        p_rewire = 0.0;
    }

    // If p_rewire is 0, no need to loop over the edges
    if (p_rewire >= 0.0) {
        // Get a uniform dist [0, 1)
        std::uniform_real_distribution<double> p_dist(0.0, 1.0);
        // Get a uniform integer distribution from [0, N) for generating new edge indices
        std::uniform_int_distribution<std::uint32_t> dest_dist(0, N-1);
        // Randomly rewire edges
        for (auto& edge : network.edges) {
            if (p_dist(rng) < p_rewire) {
                std::uint32_t newDest = dest_dist(rng);
                // Repeat the generation until the edge is not a self-edge, nor duplicate
                // Not using do while so the search for the edge is ok.
                // @note - this will be a challenge to parallelise efficiently without atomics (i.e. stable matching)
                // @note - this will be expensive for large networks, an edge matrix would be cheaper (at the cost of N*N memory)
                // @todo - need to avoid directed edge duplicates.
                while (newDest == edge.source || std::find(network.edges.begin(), network.edges.end(), UndirectedNetwork::Edge{edge.source, newDest}) != network.edges.end()) {
                    newDest = dest_dist(rng);
                }
                edge.dest = newDest;
            }
        }
    }

    return network;
}

}  // namespace network
}  // namespace exateppabm
