#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "fmt/core.h"
#include "flamegpu/flamegpu.h"
#include "exateppabm/network.h"

/**
 * Test fully connected undirected network generation
 *
 * @todo test more edge cases
 */
TEST(TestNetwork, generateFullyConnectedUndirectedGraph) {
    // Generate nodes from [1, 4]
    constexpr std::uint32_t N = 4;
    std::vector<flamegpu::id_t> nodes(N);
    std::iota(nodes.begin(), nodes.end(), 1);

    // Generate a fully connected network
    exateppabm::network::UndirectedGraph network = exateppabm::network::generateFullyConnectedUndirectedGraph(nodes);
    // Check the number of nodes and edges are correct
    EXPECT_EQ(network.getNumVertices(), N);
    EXPECT_EQ(network.getNumVertices(), nodes.size());
    EXPECT_EQ(network.getNumEdges(), (N * (N-1)) / 2);
    // Check the edges are the expected values.
}

/**
 * Test watts strogatz undirected network generation
 *
 * @todo - edge case testing.
 */
TEST(TestNetwork, generateSmallWorldUndirectedGraph) {
    // Generate nodes from [1, 16]
    constexpr std::uint32_t N = 16;
    std::vector<flamegpu::id_t> nodes(N);
    std::iota(nodes.begin(), nodes.end(), 1);

    // Generate a small world network
    constexpr std::uint32_t K = 4;  // degree 4
    constexpr double p_rewire = 0.1;  // probability of a rewire
    std::mt19937_64 rng(12);  // seeded mersenne twister rng engine
    exateppabm::network::UndirectedGraph network = exateppabm::network::generateSmallWorldUndirectedGraph(nodes, K, p_rewire, rng);

    // Ensure there are the correct number of nodes and edges still
    EXPECT_EQ(network.getNumVertices(), nodes.size());
    EXPECT_EQ(network.getNumEdges(), (N * K) / 2);
    // @todo - validate the in degree and out degree of each node.
    // If graph is directed, out degree will still be K, but indegree will vary per node
    // If graph is undirected, out degree and in degree would be the same, but mean degree will be K.
    // @todo - decide which is intended in this case.

    // @todo - actually test the network properties. Average degree etc.
    size_t edgeIdx = 0;
    for (const auto &edge : network.getEdges()) {
        // Every source and ever dest should be a valid node index
        EXPECT_LT(edge.source, network.getNumVertices());
        EXPECT_LT(edge.dest, network.getNumVertices());
        // No self-edges
        EXPECT_NE(edge.source, edge.dest);

        // fmt::print("edges[{}] = ID {} -> {} (idx {} -> {})\n", edgeIdx, network.nodes.at(edge.source), network.nodes.at(edge.dest), edge.source, edge.dest);
        ++edgeIdx;
    }
    // @todo - check for duplicates

    // Check invalid waltz strogatz parameters behave as intended
    // Odd degree - actually use K-1,
    auto n_k_odd = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4}}, 3, 0.1, rng);
    EXPECT_EQ(n_k_odd.getNumVertices(), 4);
    EXPECT_EQ(n_k_odd.getNumEdges(), ((3-1) * n_k_odd.getNumVertices())/2);
    // too high of a degree
    EXPECT_ANY_THROW(exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4}}, 5, 0.1, rng));
    // too low probability
    EXPECT_ANY_THROW(exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4}}, 2, -1.0, rng));
    // too high probability
    EXPECT_ANY_THROW(exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4}}, 2, 2.0, rng));

    // Check the networks contain the expected number of edges for edge case node counts
    // 0 nodes, no edges
    auto n0 = exateppabm::network::generateSmallWorldUndirectedGraph({}, 2, 0.1, rng);
    EXPECT_EQ(n0.getNumVertices(), 0u);
    EXPECT_EQ(n0.getNumEdges(), 0u);
    // 1 node, no edges
    auto n1 = exateppabm::network::generateSmallWorldUndirectedGraph({{1}}, 2, 0.1, rng);
    EXPECT_EQ(n1.getNumVertices(), 1u);
    EXPECT_EQ(n1.getNumEdges(), 0u);
    // 2 node, fully connected
    auto n2 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2}}, 2, 0.1, rng);
    EXPECT_EQ(n2.getNumVertices(), 2u);
    EXPECT_EQ(n2.getNumEdges(), 1u);
    // 3 nodes, mean degree 2, 3 edges
    auto n3 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3}}, 2, 0.1, rng);
    EXPECT_EQ(n3.getNumVertices(), 3u);
    EXPECT_EQ(n3.getNumEdges(), 3u);
    // 4 nodes, degree 2, 4 edges
    auto n4_2 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4}}, 2, 0.1, rng);
    EXPECT_EQ(n4_2.getNumVertices(), 4u);
    EXPECT_EQ(n4_2.getNumEdges(), 4u);
    // 4 nodes, degree 4, fully connected, so only 6 edges
    auto n4_4 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4}}, 4, 0.1, rng);
    EXPECT_EQ(n4_4.getNumVertices(), 4u);
    EXPECT_EQ(n4_4.getNumEdges(), 6u);

    // 12 nodes, degree 2
    auto n12_2 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}, 2, 0.1, rng);
    EXPECT_EQ(n12_2.getNumVertices(), 12u);
    EXPECT_EQ(n12_2.getNumEdges(), 12u);
    // 12 nodes, degree 2 (equiv to 2)
    auto n12_3 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}, 3, 0.1, rng);
    EXPECT_EQ(n12_3.getNumVertices(), 12u);
    EXPECT_EQ(n12_3.getNumEdges(), 12u);
    // 12 nodes, degree 4
    auto n12_4 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}, 4, 0.1, rng);
    EXPECT_EQ(n12_4.getNumVertices(), 12u);
    EXPECT_EQ(n12_4.getNumEdges(), 24u);
    // 12 nodes, degree 5 (equiv to 5)
    auto n12_5 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}, 5, 0.1, rng);
    EXPECT_EQ(n12_5.getNumVertices(), 12u);
    EXPECT_EQ(n12_5.getNumEdges(), 24u);
    // 12 nodes, degree 6
    auto n12_6 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}, 6, 0.1, rng);
    EXPECT_EQ(n12_6.getNumVertices(), 12u);
    EXPECT_EQ(n12_6.getNumEdges(), 36u);
    // 12 nodes, degree 8
    auto n12_8 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}, 8, 0.1, rng);
    EXPECT_EQ(n12_8.getNumVertices(), 12u);
    EXPECT_EQ(n12_8.getNumEdges(), 48u);
    // 12 nodes, degree 8
    auto n12_10 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}, 10, 0.1, rng);
    EXPECT_EQ(n12_10.getNumVertices(), 12u);
    EXPECT_EQ(n12_10.getNumEdges(), 60u);
    // 12 nodes, degree 8
    auto n12_12 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}, 12, 0.1, rng);
    EXPECT_EQ(n12_12.getNumVertices(), 12u);
    EXPECT_EQ(n12_12.getNumEdges(), 66u);  // fully connected

    // 8 nodes, degree 4, no rewiring
    auto n8_4_0 = exateppabm::network::generateSmallWorldUndirectedGraph({{1, 2, 3, 4, 5, 6, 7, 8}}, 4, 0.0, rng);
    EXPECT_EQ(n8_4_0.getNumVertices(), 8u);
    EXPECT_EQ(n8_4_0.getNumEdges(), 16u);
    // Check for valid edge, ascending internal order, ensure no dupes.
    std::set<exateppabm::network::Edge> uniqueEdges_n8_4_0 = {{}};
    for (const auto & edge : n8_4_0.getEdges()) {
        EXPECT_GE(edge.source, 0u);
        EXPECT_LT(edge.source, 8u);
        EXPECT_NE(edge.source, edge.dest);
        // Ensure the source node index is less than the dest index (i.e. undirected edges are always stored with the source having the lower index)
        EXPECT_LT(edge.source, edge.dest);
        // Edges should be unique, so this edge should not have been seen yet.
        EXPECT_EQ(uniqueEdges_n8_4_0.count(edge), 0);
        uniqueEdges_n8_4_0.insert(edge);
        // with rewiring 0, the dest should always be source +1 or source +2 or when wrapping the boundary source will be dest -1 or dest -2 wrapped + 2)
        EXPECT_TRUE((edge.dest == edge.source + 1) || (edge.dest == edge.source + 2) || (edge.source == (edge.dest + 1) % n8_4_0.getNumVertices()) || (edge.source == (edge.dest + 2) % n8_4_0.getNumVertices()));
    }
}
