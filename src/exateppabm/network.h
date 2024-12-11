#pragma once

#include <cstdint>
#include <exception>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

#include "flamegpu/flamegpu.h"

namespace exateppabm {
namespace network {


/**
 * Struct representing an Edge
 */
class Edge {
 public:
    std::uint32_t source;
    std::uint32_t dest;
    /**
     * constrcutor for emplace_back support
     */
    Edge(std::uint32_t source, std::uint32_t dest) : source(source), dest(dest) {}
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
 * class representing an undirected network, using edge lists as well suited to edge existan
 *
 * @todo - template the index type? Map indices to ID so a sparse matrix would make sense?
 */
class UndirectedGraph {
 public:
    /**
     * Default ctor
     */
    UndirectedGraph() { }

    /**
     * Constructor, taking a vector of vertex labels
     *
     * @param vertexLabels vertex labels, setting the number of vertices for the network
     */
    explicit UndirectedGraph(std::vector<flamegpu::id_t> vertexLabels) : _vertexLabels(vertexLabels) {
        this->_adjacencyList.resize(this->getNumVertices());
    }

    /**
     * copy constructor
     *
     * @param other instance of UndirectedGraph to copy
     */
    UndirectedGraph(const UndirectedGraph& other) :
    _vertexLabels(other._vertexLabels),
    _edges(other._edges),
    _adjacencyList(other._adjacencyList) {}

    /**
     * Get the number of nodes
     * @return number of vertices/nodes
     */
    std::uint32_t getNumVertices() const { return this->_vertexLabels.size(); }
    /**
     * Get the number of edges
     * @return number of edges
     */
    std::uint32_t getNumEdges() const { return this->_edges.size(); }
    /**
     * Check if an edge exists within the undirected graph
     * @param u vertex index for one end of the edge
     * @param v vertex index for the other end of the edge
     * @return bool indicating if edge exists or not
     */
    bool contains(std::uint32_t u, std::uint32_t v) const {
        return this->_adjacencyList[u].count(v) > 0;
    }
    /**
     * Check if an edge exists within the undirected graph
     * @param edge to check for existance of
     * @return bool indicating if edge exists or not
     */
    bool contains(Edge edge) const {
        return this->_adjacencyList[edge.source].count(edge.dest) > 0;
    }
    /**
     * Add an edge to the undirected graph
     * @param u vertex index for one end of the edge
     * @param v vertex index for the other end of the edge
     */
    void addEdge(std::uint32_t u, std::uint32_t v) {
        if (u > this->getNumVertices() || v > this->getNumVertices()) {
            throw std::runtime_error("Invalid u or v for UndirectedGraph::getNumVertices. @todo better error");
        }
        // Undirected, so add the edge in both directions
        this->_adjacencyList[u].insert(v);
        this->_adjacencyList[v].insert(u);
        // Only store the ascending ordered edge in the vector of edges
        if (u < v) {
            this->_edges.emplace_back(u, v);
        } else {
            this->_edges.emplace_back(v, u);
        }
    }
    /**
     * Add an edge to the undirected graph
     * @param edge edge to add
     */
    void addEdge(Edge edge) {
        return this->addEdge(edge.source, edge.dest);
    }
    /**
     * Remove an edge from the undirected graph
     * @param edge edge to remove
     */
    void removeEdge(Edge edge) {
        this->_adjacencyList[edge.source].erase(edge.dest);
        this->_adjacencyList[edge.dest].erase(edge.source);
        // Remove the edge from the edges_ vector
        auto it = std::remove_if(this->_edges.begin(), this->_edges.end(), [&edge](const Edge& e) { return e == edge; });
        this->_edges.erase(it, this->_edges.end());
    }
    /**
     * Remove an edge from the undirected graph
     * @param u vertex index for one end of the edge
     * @param v vertex index for the other end of the edge
     */
    void removeEdge(std::uint32_t u, std::uint32_t v) {
        return this->removeEdge({u, v});
    }

    /**
     * Get the degree for a vertex. this is in-out as undirected.
     *
     * @return vertex degree
     */
    std::uint32_t degree(std::uint32_t v) {
        return static_cast<std::uint32_t>(this->_adjacencyList[v].size());
    }

    /**
     * Get the vertex indices of neighbouring vertices
     * @param v vertex index
     * @return unordered set of neighbouring vertices to v
     */
    const std::unordered_set<std::uint32_t>& getNeighbours(std::uint32_t v) const {
        return this->_adjacencyList[v];
    }

    /**
     * Get the vector of edges
     *
     * @return vector of edges
     */
    const std::vector<Edge>& getEdges() const { return this->_edges; }

    /**
     * Get the vertex labels, which are probably 1-indexed FLAME GPU id_t (0 is unset value)
     *
     * @return vector of vertex labels
     */
    const std::vector<flamegpu::id_t>& getVertexLabels() const { return this->_vertexLabels; }

    /**
     * Get the a vertex label
     *
     * @param v vertex index
     * @return label for vertex
     */
    flamegpu::id_t getVertexLabel(std::uint32_t v) const { return this->_vertexLabels.at(v); }

 private:
    /**
     * Vector of nodes, which provides the mapping from the network-index based edges back to agent IDs
     */
    std::vector<flamegpu::id_t> _vertexLabels;


    /**
     * Vector of edge objects, each containing the directed source-dest pair. These are 0 indexed within this network.
     */
    std::vector<Edge> _edges;

    /**
     * Vector of unordered sets for fast lookup of edge existence.
     */
    std::vector<std::unordered_set<std::uint32_t>> _adjacencyList;
};

/**
 * Generate a fully connected undirected network given a vector of node labels
 * @param nodes vector of nodes (agent Id's) to include in this network. IDs may not be sequential and should be one indexed.
 * @return network struct for a fully connected network
 */
UndirectedGraph generateFullyConnectedUndirectedGraph(std::vector<flamegpu::id_t> nodes);

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
UndirectedGraph generateSmallWorldUndirectedGraph(std::vector<flamegpu::id_t> nodes, std::uint32_t K, double p, std::mt19937_64 rng);


}  // namespace network
}  // namespace exateppabm
