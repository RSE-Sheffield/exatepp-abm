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


UndirectedGraph generateFullyConnectedUndirectedGraph(std::vector<flamegpu::id_t> nodes) {
    UndirectedGraph network(nodes);
    if (nodes.size() <= 1) {
        return network;
    }
    std::uint32_t N = static_cast<std::int32_t>(nodes.size());
    std::uint32_t E = (N * (N - 1)) / 2;
    // network.edges.reserve(E);
    // Generate edges in ascending order for the undirected graph.
    for (std::uint32_t i = 0; i < N; ++i) {
        for (std::uint32_t j = i + 1; j < N; ++j) {
            network.addEdge(i, j);
        }
    }
    return network;
}

// @todo - should the initial lattice be randomised rather than sequential?
// @todo - should the graph be directed or undirected? Data structure is directed, so do pairs need rewiring together?
UndirectedGraph generateSmallWorldUndirectedGraph(std::vector<flamegpu::id_t> nodes, std::uint32_t K, double p_rewire, std::mt19937_64 rng) {
    UndirectedGraph network(nodes);

    // Return early with an empty network if 0 or 1 nodes, so no room for non-self edges
    if (network.getNumVertices() <= 1) {
        return network;
    }

    // If K is 0 (or 1), no edges so do nothing
    if (K <= 1) {
        // throw std::runtime_error("@todo values - small world network K (" + std::to_string(K) + ") must be > 1");
        return network;
    } else if (K == network.getNumVertices()) {
        // If K == Node count, the graph is fully connected, so return a fully connected graph
        return generateFullyConnectedUndirectedGraph(nodes);
    } else if (K >= network.getNumVertices()) {
        // Raise an exception if K is too large
        throw std::runtime_error(std::string("@todo values - small world network K(") + std::to_string(K) + ") must be less than |N| (" + std::to_string(network.getNumVertices()) + ")");
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

    // network.edges.reserve(E);
    for (std::int32_t i = 0; i < N; ++i) {
        for (std::int32_t j = 1; j <= static_cast<std::int32_t>(K / 2); ++j) {
            // only add the positive undirected edges
            std::uint32_t s = static_cast<std::uint32_t>(i);
            std::uint32_t d = static_cast<std::uint32_t>((i + j) % N);
            network.addEdge(s, d);
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

        // Take a copy of the network edges, to ensure we avoid iterator invalidation
        // Only the current iterator should be removed in the undirected graph, so this is probably ok.
        std::vector<network::Edge> copyOfEdges(network.getEdges().begin(), network.getEdges().end());
        for (auto& edge : copyOfEdges) {
            if (p_dist(rng) < p_rewire) {
                // If the source vertex is full, do not attempt to rewire it, it would loop forever so check the next original edge
                if (network.degree(edge.source) >= network.getNumVertices() - 1) {
                    continue;
                }

                // Repeatedly generate a new destination vertex until a new edge has been found.
                std::uint32_t newDest = dest_dist(rng);
                // While the new dest is the source, or already in the graph, try again.
                while (newDest == edge.source || network.contains(edge.source, newDest)) {
                    newDest = dest_dist(rng);
                }
                // Remove the old edge
                network.removeEdge(edge.source, edge.dest);
                // Add the new edge
                network.addEdge(edge.source, newDest);
            }
        }
    }

    return network;
}

}  // namespace network
}  // namespace exateppabm
