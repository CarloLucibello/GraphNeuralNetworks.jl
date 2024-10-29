"""
    struct NeighborLoader

A data structure for sampling neighbors from a graph for training Graph Neural Networks (GNNs). 
It supports multi-layer sampling of neighbors for a batch of input nodes, useful for mini-batch training
originally introduced in "Inductive Representation Learning on Large Graphs" paper.
[see https://arxiv.org/abs/1706.02216]

# Fields
- `graph::GNNGraph`: The input graph.
- `num_neighbors::Vector{Int}`: A vector specifying the number of neighbors to sample per node at each GNN layer.
- `input_nodes::Vector{Int}`: A vector containing the starting nodes for neighbor sampling.
- `num_layers::Int`: The number of layers for neighborhood expansion (how far to sample neighbors).
- `batch_size::Union{Int, Nothing}`: The size of the batch. If not specified, it defaults to the number of `input_nodes`.

# Usage
```julia
loader = NeighborLoader(graph; num_neighbors=[10, 5], input_nodes=[1, 2, 3], num_layers=2)
```
"""
struct NeighborLoader
    graph::GNNGraph             # The input GNNGraph (graph + features from GraphNeuralNetworks.jl)
    num_neighbors::Vector{Int}  # Number of neighbors to sample per node, for each layer
    input_nodes::Vector{Int}    # Set of input nodes (starting nodes for sampling)
    num_layers::Int             # Number of layers for neighborhood expansion
    batch_size::Union{Int, Nothing}  # Optional batch size, defaults to the length of input_nodes if not given
    neighbors_cache::Dict{Int, Vector{Int}}  # Cache neighbors to avoid recomputation
end

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

"""
    sample_nbrs(loader::NeighborLoader, node::Int, layer::Int)

Samples a specified number of neighbors for the given `node` at a particular `layer` of the GNN. 
    The number of neighbors sampled is defined in `loader.num_neighbors`.

# Arguments:
- `loader::NeighborLoader`: The `NeighborLoader` instance.
- `node::Int`: The node to sample neighbors for.
- `layer::Int`: The current GNN layer (used to determine how many neighbors to sample).

# Returns:
A vector of sampled neighbor node indices.
"""
# Function to sample neighbors for a given node at a specific layer
function sample_nbrs(loader::NeighborLoader, node::Int, layer::Int)
    neighbors = get_neighbors(loader, node)
    if isempty(neighbors)
        return Int[]
    else
        num_samples = min(loader.num_neighbors[layer], length(neighbors))  # Limit to required samples for this layer
        return rand(neighbors, num_samples)  # Randomly sample neighbors
    end
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
                neighbors = sample_nbrs(loader, n, layer)  # Sample neighbors of the node for this layer
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

    mini_batch_gnn = Graphs.induced_subgraph(loader.graph, subgraph_node_list)  # Create a subgraph of the nodes

    # Continue iteration for the next batch
    return mini_batch_gnn, state + batch_size
end
