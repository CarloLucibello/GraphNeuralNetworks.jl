# Import necessary packages
using GraphNeuralNetworks

# Define a graph structure (using GraphNeuralNetworks.jl)
struct GNNGraph
    graph::Graph         # Graph structure from GraphNeuralNetworks.jl
    features::Matrix     # Feature matrix: rows represent nodes, columns are features
end

# Define a NeighborLoader structure for sampling neighbors
struct NeighborLoader
    graph::GNNGraph          # The input GNNGraph (graph + features)
    num_neighbors::Int       # Number of neighbors to sample per node
    batch_size::Int          # Number of nodes in each mini-batch
    num_layers::Int          # Number of layers for neighborhood expansion
end

# Function to sample neighbors for a given node
function sample_neighbors(loader::NeighborLoader, node::Int)
    neighbors = neighbors(loader.graph.graph, node)  # Get all neighbors of the node from the graph
    num_samples = min(loader.num_neighbors, length(neighbors))  # Choose min between neighbors and required sample size
    sampled_neighbors = rand(neighbors, num_samples)  # Randomly sample the neighbors
    return sampled_neighbors
end

# Function to create a mini-batch of nodes and their neighbors
function create_mini_batch(loader::NeighborLoader)
    # Randomly select batch_size nodes
    batch_nodes = rand(1:nv(loader.graph.graph), loader.batch_size)

    # Initialize storage for neighbors and features
    batch_neighbors = Dict{Int, Vector{Int}}()  # Store sampled neighbors
    batch_features = Dict{Int, Vector{Float64}}()  # Store node features

    for node in batch_nodes
        # Initialize current layer of nodes (starting with the node itself)
        sampled_neighbors = [node]
        
        # For each GNN layer, sample the neighborhood
        for layer in 1:loader.num_layers
            new_neighbors = []
            for n in sampled_neighbors
                neighbors = sample_neighbors(loader, n)  # Sample neighbors of current node
                append!(new_neighbors, neighbors)
            end
            sampled_neighbors = unique(new_neighbors)  # Update sampled neighbors for next layer
        end

        # Store neighbors and features of the node
        batch_neighbors[node] = sampled_neighbors
        batch_features[node] = loader.graph.features[:, node]  # Assuming column-wise features for each node
    end

    return batch_nodes, batch_neighbors, batch_features
end

# Function for training the model with the NeighborLoader
function train_model(graph::GNNGraph, num_neighbors::Int, batch_size::Int, num_layers::Int, num_batches::Int)
    # Initialize the NeighborLoader
    loader = NeighborLoader(graph, num_neighbors, batch_size, num_layers)

    # Loop through the number of batches for training
    for batch in 1:num_batches
        batch_nodes, batch_neighbors, batch_features = create_mini_batch(loader)
        println("Batch $batch: Nodes: $batch_nodes, Neighbors: $batch_neighbors")
        
        # Here, you would pass batch data to the GNN model for training
        # For example: model(batch_nodes, batch_neighbors, batch_features)
    end
end

# Example of creating a GNN graph and training
function main()
    # Sample Graph structure from GraphNeuralNetworks.jl
    # Create a small graph with 5 nodes and example edges
    graph = Graph(5)
    add_edge!(graph, 1, 2)
    add_edge!(graph, 1, 3)
    add_edge!(graph, 2, 4)
    add_edge!(graph, 3, 5)

    # Create random features for the nodes (5 nodes, 3 features per node)
    features = rand(3, 5)

    # Create GNNGraph
    gnn_graph = GNNGraph(graph, features)

    # Train model using NeighborLoader
    train_model(gnn_graph, num_neighbors=2, batch_size=2, num_layers=2, num_batches=3)
end

# Run the example
main()

## iterator