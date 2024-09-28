using GraphNeuralNetworks
using Graphs

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
function NeighborLoader(graph::GNNGraph; num_neighbors::Vector{Int}, input_nodes::Vector{Int}, num_layers::Int, batch_size::Union{Int, Nothing}=nothing)
    return NeighborLoader(graph, num_neighbors, input_nodes, num_layers, batch_size === nothing ? length(input_nodes) : batch_size, Dict{Int, Vector{Int}}())
end

# Function to get cached neighbors or compute them
function get_neighbors(loader::NeighborLoader, node::Int)
    if haskey(loader.neighbors_cache, node)
        return loader.neighbors_cache[node]
    else
        neighbors = Graphs.neighbors(loader.graph, node, dir = :in)  # Get neighbors from graph
        loader.neighbors_cache[node] = neighbors
        return neighbors
    end
end

# Function to sample neighbors for a given node at a specific layer
function sample_neighbors(loader::NeighborLoader, node::Int, layer::Int)
    neighbors = get_neighbors(loader, node)
    if isempty(neighbors)
        return Int[]
    else
        num_samples = min(loader.num_neighbors[layer], length(neighbors))  # Limit to required samples for this layer
        return rand(neighbors, num_samples)  # Randomly sample neighbors
    end
end

function induced_subgraph(graph::GNNGraph, nodes::Vector{Int})
    if isempty(nodes)
        return GNNGraph()  # Return empty graph if no nodes are provided
    end

    node_map = Dict(node => i for (i, node) in enumerate(nodes))

    # Collect edges to add
    source = Int[]
    target = Int[]
    backup_gnn = GNNGraph()
    for node in nodes
        neighbors = Graphs.neighbors(graph, node, dir = :in)
        if isempty(neighbors)
            backup_gnn = add_nodes(backup_gnn, 1)
        end
        for neighbor in neighbors
            if neighbor in keys(node_map)
                push!(source, node_map[node])
                push!(target, node_map[neighbor])
            end
        end
    end

    # Extract features for the new nodes
    new_features = graph.x[:, nodes]

    if isempty(source) && isempty(target)
        backup_gnn.ndata.x = new_features
        return backup_gnn  # Return empty graph if no nodes are provided
    end

    return GNNGraph(source, target, ndata = new_features)  # Return the new GNNGraph with subgraph and features
end

# Iterator protocol for NeighborLoader with lazy batch loading
function Base.iterate(loader::NeighborLoader, state=1)
    if state > length(loader.input_nodes) 
        return nothing  # End of iteration if batches are exhausted (state larger than amount of input nodes or current batch no >= batch number)
    end

    # Determine the size of the current batch
    batch_size = min(loader.batch_size, length(loader.input_nodes) - state + 1) # Conditional in case there is not enough nodes to fill the last batch
    batch_nodes = loader.input_nodes[state:state + batch_size - 1] # Each mini-batch uses different set of input nodes 

    # Set for tracking the subgraph nodes
    subgraph_nodes = Set(batch_nodes)

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

    if isempty(subgraph_node_list)
        return GNNGraph(), state + batch_size
    end

    mini_batch_gnn = induced_subgraph(loader.graph, subgraph_node_list)  # Create a subgraph of the nodes

    # Continue iteration for the next batch
    return mini_batch_gnn, state + batch_size
end