#TODO reactivate test
# @testitem "NeighborLoader"  setup=[TestModule] begin
#     using .TestModule
#     # Helper function to create a simple graph with node features using GNNGraph
#     function create_test_graph()
#         source = [1, 2, 3, 4]  # Define source nodes of edges
#         target = [2, 3, 4, 5]  # Define target nodes of edges
#         node_features = rand(Float32, 5, 5)  # Create random node features (5 features for 5 nodes)

#         return GNNGraph(source, target, ndata = node_features)  # Create a GNNGraph with edges and features
#     end


#     # 1. Basic functionality: Check neighbor sampling and subgraph creation
#     @testset "Basic functionality" begin
#         g = create_test_graph()

#         # Define NeighborLoader with 2 neighbors per layer, 2 layers, batch size 2
#         loader = NeighborLoader(g; num_neighbors=[2, 2], input_nodes=[1, 2], num_layers=2, batch_size=2)

#         mini_batch_gnn, next_state = iterate(loader)

#         # Test if the mini-batch graph is not empty
#         @test !isempty(mini_batch_gnn.graph)

#         num_sampled_nodes = mini_batch_gnn.num_nodes
#         println("Number of nodes in mini-batch: ", num_sampled_nodes)

#         @test num_sampled_nodes == 2

#         # Test if there are edges in the subgraph
#         @test mini_batch_gnn.num_edges > 0
#     end

#     # 2. Edge case: Single node with no neighbors
#     @testset "Single node with no neighbors" begin
#         g = SimpleDiGraph(1)  # A graph with a single node and no edges
#         node_features = rand(Float32, 5, 1)
#         graph = GNNGraph(g, ndata = node_features)

#         loader = NeighborLoader(graph; num_neighbors=[2], input_nodes=[1], num_layers=1)

#         mini_batch_gnn, next_state = iterate(loader)

#         # Test if the mini-batch graph contains only one node
#         @test size(mini_batch_gnn.x, 2) == 1
#     end

#     # 3. Edge case: A node with no outgoing edges (isolated node)
#     @testset "Node with no outgoing edges" begin
#         g = SimpleDiGraph(2)  # Graph with 2 nodes, no edges
#         node_features = rand(Float32, 5, 2)
#         graph = GNNGraph(g, ndata = node_features)

#         loader = NeighborLoader(graph; num_neighbors=[1], input_nodes=[1, 2], num_layers=1)

#         mini_batch_gnn, next_state = iterate(loader)

#         # Test if the mini-batch graph contains the input nodes only (as no neighbors can be sampled)
#         @test size(mini_batch_gnn.x, 2) == 2  # Only two isolated nodes
#     end

#     # 4. Edge case: A fully connected graph
#     @testset "Fully connected graph" begin
#         g = SimpleDiGraph(3)
#         add_edge!(g, 1, 2)
#         add_edge!(g, 2, 3)
#         add_edge!(g, 3, 1)
#         node_features = rand(Float32, 5, 3)
#         graph = GNNGraph(g, ndata = node_features)

#         loader = NeighborLoader(graph; num_neighbors=[2, 2], input_nodes=[1], num_layers=2)

#         mini_batch_gnn, next_state = iterate(loader)

#         # Test if all nodes are included in the mini-batch since it's fully connected
#         @test size(mini_batch_gnn.x, 2) == 3  # All nodes should be included
#     end

#     # 5. Edge case: More layers than the number of neighbors
#     @testset "More layers than available neighbors" begin
#         g = SimpleDiGraph(3)
#         add_edge!(g, 1, 2)
#         add_edge!(g, 2, 3)
#         node_features = rand(Float32, 5, 3)
#         graph = GNNGraph(g, ndata = node_features)

#         # Test with 3 layers but only enough connections for 2 layers
#         loader = NeighborLoader(graph; num_neighbors=[1, 1, 1], input_nodes=[1], num_layers=3)

#         mini_batch_gnn, next_state = iterate(loader)

#         # Test if the mini-batch graph contains all available nodes
#         @test size(mini_batch_gnn.x, 2) == 1
#     end

#     # 6. Edge case: Large batch size greater than the number of input nodes
#     @testset "Large batch size" begin
#         g = create_test_graph()

#         # Define NeighborLoader with a larger batch size than input nodes
#         loader = NeighborLoader(g; num_neighbors=[2], input_nodes=[1, 2], num_layers=1, batch_size=10)

#         mini_batch_gnn, next_state = iterate(loader)

#         # Test if the mini-batch graph is not empty
#         @test !isempty(mini_batch_gnn.graph)

#         # Test if the correct number of nodes are sampled
#         @test size(mini_batch_gnn.x, 2) == length(unique([1, 2]))  # Nodes [1, 2] are expected
#     end

#     # 7. Edge case: No neighbors sampled (num_neighbors = [0]) and 1 layer
#     @testset "No neighbors sampled" begin
#         g = create_test_graph()

#         # Define NeighborLoader with 0 neighbors per layer, 1 layer, batch size 2
#         loader = NeighborLoader(g; num_neighbors=[0], input_nodes=[1, 2], num_layers=1, batch_size=2)

#         mini_batch_gnn, next_state = iterate(loader)

#         # Test if the mini-batch graph contains only the input nodes
#         @test size(mini_batch_gnn.x, 2) == 2  # No neighbors should be sampled, only nodes 1 and 2 should be in the graph
#     end

# end