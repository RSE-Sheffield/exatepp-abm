#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "flamegpu/flamegpu.h"

namespace exateppabm {
namespace network {

/**
 * Struct representing a network.
 * 
 * @todo - re-use flamegpu's networks? Not sure this is viable yet.
 * @todo - make it a class
 * @todo - template the index type? Map indices to ID so a sparse matrix would make sense? 
 */
struct UndirectedNetwork {
    /**
     * Struct representing an Edge
     */
    struct Edge {
        std::uint32_t source;
        std::uint32_t dest;
        /**
         * const operator== comparison for std::find
         */
        bool operator==(const Edge& other) const {
            return source == other.source && dest == other.dest;
        }
        /**
         * operator< for use in a set
         */
        bool operator<(const Edge& other) const {
        if (source != other.source) {
            return source < other.source;
        } else {
            return dest < other.dest;
        }
    }
    };

    /**
     * Vector of nodes, which provides the mapping from the network-index based edges back to agent IDs
     */
    std::vector<flamegpu::id_t> nodes;
    /**
     * Vector of edge objects, each containing the directed source-dest pair. These are 0 indexed within this network.
     */
    std::vector<Edge> edges;
};

/**
 * Generate a fully connected undirected network given a vector of node labels
 * @param nodes vector of nodes (agent Id's) to include in this network. IDs may not be sequential and should be one indexed.
 * @return network struct for a fully connected network
 */
UndirectedNetwork generateFullyConnectedUndirectedNetwork(std::vector<flamegpu::id_t> nodes);

/**
 * Generate a watts strogatz small world  (undirected) network for a given set of nodes (agent ID's, 1 indexed, not sequential), K & p.
 * 
 * If k is odd, then k-1 neighboours are created
 * 
 * @param nodes vector of nodes (i.e. agent id's) to include in this network. IDs may not be sequential and should be 1 indexed.
 * @param K the degree of each node
 * @param p rewire probability
 * @return a network struct for the generate small world network
 */
UndirectedNetwork generateSmallWorldUndirectedNetwork(std::vector<flamegpu::id_t> nodes, std::uint32_t K, double p, std::mt19937_64 rng);


}  // namespace network
}  // namespace exateppabm
