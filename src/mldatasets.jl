# We load a Graph Dataset from MLDatasets without explicitely depending on it

"""
    mldataset2gnngraph(dataset)

Convert a graph dataset from the package MLDatasets.jl into one or many [`GNNGraph`](@ref)s.

# Examples

```julia-repl
julia> using MLDatasets, GraphNeuralNetworks

julia> mldataset2gnngraph(Cora())
GNNGraph:
    num_nodes = 2708
    num_edges = 10556
    ndata:
        features => 1433×2708 Matrix{Float32}
        targets => 2708-element Vector{Int64}
        train_mask => 2708-element BitVector
        val_mask => 2708-element BitVector
        test_mask => 2708-element BitVector
```
"""
function mldataset2gnngraph(dataset::D) where D
    @assert hasproperty(dataset, :graphs)
    graphs = mlgraph2gnngraph.(dataset.graphs)
    if length(graphs) == 1
        return graphs[1]
    else
        return graphs
    end
end

function mlgraph2gnngraph(g::G) where G
    @assert hasproperty(g, :num_nodes)
    @assert hasproperty(g, :edge_index)
    @assert hasproperty(g, :node_data)
    @assert hasproperty(g, :edge_data)
    return GNNGraph(g.edge_index; ndata=g.node_data, edata=g.edge_data, g.num_nodes)
end
