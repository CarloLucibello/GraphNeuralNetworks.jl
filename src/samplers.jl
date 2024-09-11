using GraphNeuralNetworks

# Define a NeighborLoader structure for sampling neighbors
struct NeighborLoader
    graph::GNNGraph             # The input GNNGraph (graph + features from GraphNeuralNetworks.jl)
    num_neighbors::Vector{Int}  # Number of neighbors to sample per node, for each layer
    input_nodes::Vector{Int}    # Set of input nodes (starting nodes for sampling)
    num_layers::Int             # Number of layers for neighborhood expansion
    batch_size::Union{Int, Nothing}  # Optional batch size, defaults to the length of input_nodes if not given
    neighbors_cache::Dict{Int, Vector{Int}}  # Cache neighbors to avoid recomputation
end

# Constructor for NeighborLoader with optional batch size
function NeighborLoader(graph::GNNGraph, num_neighbors::Vector{Int}, input_nodes::Vector{Int}, num_layers::Int, batch_size::Union{Int, Nothing} = nothing)
    return NeighborLoader(graph, num_neighbors, input_nodes, num_layers, batch_size === nothing ? length(input_nodes) : batch_size, Dict{Int, Vector{Int}}())
end

# Function to get cached neighbors or compute them
function get_neighbors(loader::NeighborLoader, node::Int)
    if haskey(loader.neighbors_cache, node)
        return loader.neighbors_cache[node]
    else
        neighbors = neighbors(loader.graph, node)  # Get neighbors from graph
        loader.neighbors_cache[node] = neighbors
        return neighbors
    end
end

# Function to sample neighbors for a given node at a specific layer
function sample_neighbors(loader::NeighborLoader, node::Int, layer::Int)
    neighbors = get_neighbors(loader, node)
    num_samples = min(loader.num_neighbors[layer], length(neighbors))  # Limit to required samples for this layer
    return rand(neighbors, num_samples)  # Randomly sample neighbors
end

# Iterator protocol for NeighborLoader with lazy batch loading
function Base.iterate(loader::NeighborLoader, state=1)
    if state > length(loader.input_nodes)
        return nothing  # End of iteration
    end

    # Determine the size of the current batch
    batch_size = min(loader.batch_size, length(loader.input_nodes) - state + 1)
    batch_nodes = loader.input_nodes[state:state + batch_size - 1]

    # Set for tracking the subgraph nodes
    subgraph_nodes = Set(batch_nodes)
    
    # Preallocate memory for batch_features
    batch_features = loader.graph.x[:, batch_nodes]  # Initially just the features of the batch_nodes

    for node in batch_nodes
        # Initialize current layer of nodes (starting with the node itself)
        sampled_neighbors = Set([node])
        
        # For each GNN layer, sample the neighborhood
        for layer in 1:loader.num_layers
            new_neighbors = Set{Int}()
            for n in sampled_neighbors
                neighbors = sample_neighbors(loader, n, layer)  # Sample neighbors of the node for this layer
                new_neighbors = union(new_neighbors, neighbors)  # Avoid duplicates in the neighbor set
            end
            sampled_neighbors = new_neighbors
            subgraph_nodes = union(subgraph_nodes, sampled_neighbors)  # Expand the subgraph with the new neighbors
        end
    end

    # Collect subgraph nodes and their features
    subgraph_node_list = collect(subgraph_nodes)
    subgraph = induced_subgraph(loader.graph, subgraph_node_list)  # More efficient subgraph creation
    subgraph_features = loader.graph.x[:, subgraph_node_list]

    # Return a GNNGraph instance for this mini-batch
    mini_batch_gnn = GNNGraph(subgraph, subgraph_features)

    # Continue iteration for the next batch
    return mini_batch_gnn, state + batch_size
end

# Example of creating a GNN graph and training
# Sample Graph structure from GraphNeuralNetworks.jl
# Create a small graph with 5 nodes and example edges
graph = Graph(5)
add_edge!(graph, 1, 2)
add_edge!(graph, 1, 3)
add_edge!(graph, 2, 4)
add_edge!(graph, 3, 5)

# Create random features for the nodes (5 nodes, 3 features per node)
features = rand(3, 5)

# Create GNNGraph (use GNN's GNNGraph)
gnn_graph = GNNGraph(graph, features)

# Define input nodes (seed nodes) to start sampling
input_nodes = [1, 2, 3, 4, 5]

# Run the example
# Initialize the NeighborLoader with optional batch_size
loader = NeighborLoader(gnn_graph, [2, 3], input_nodes, 2, 3)  # Train without specifying batch_size (defaults to input_nodes size)
loader = NeighborLoader(gnn_graph, [2, 3], input_nodes, 2, 3, 2)  # Train with batch_size = 2

# Loop through the number of batches for training, using the iterator
batch_counter = 0
for mini_batch_gnn in loader
    batch_counter += 1
    println("Batch $batch_counter: Nodes in mini-batch graph: $(nv(mini_batch_gnn))")
    
    # Here, you would pass mini_batch_gnn to the GNN model for training
    # For example: model(mini_batch_gnn)

    # Stop if num_batches is reached
    if batch_counter >= num_batches
        break
    end
end