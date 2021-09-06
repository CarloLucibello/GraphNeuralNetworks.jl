using DataStructures: nlargest

@doc raw"""
    GlobalPool(aggr)

Global pooling layer for graph neural networks.
Takes a graph and feature nodes as inputs
and performs the operation

```math
\mathbf{u}_V = \box_{i \in V} \mathbf{x}_i
````
where ``V`` is the set of nodes of the input graph and 
the type of aggregation represented by `\box` is selected by the `aggr` argument. 
Commonly used aggregations are are `mean`, `max`, and `+`.

```julia
using GraphNeuralNetworks, LightGraphs

pool = GlobalPool(mean)

g = GNNGraph(random_regular_graph(10, 4))
X = rand(32, 10)
pool(g, X) # => 32x1 matrix
```
"""
struct GlobalPool{F}
    aggr::F
end

function (l::GlobalPool)(g::GNNGraph, X::AbstractArray)
    if isnothing(g.graph_indicator)
        # assume only one graph
        indexes = fill!(similar(X, Int, g.num_nodes), 1)     
    else 
        indexes = g.graph_indicator
    end
    return NNlib.scatter(l.aggr, X, indexes)
end

"""
    TopKPool(adj, k, in_channel)

Top-k pooling layer.

# Arguments

- `adj`: Adjacency matrix  of a graph.
- `k`: Top-k nodes are selected to pool together.
- `in_channel`: The dimension of input channel.
"""
struct TopKPool{T,S}
    A::AbstractMatrix{T}
    k::Int
    p::AbstractVector{S}
    Ã::AbstractMatrix{T}
end

function TopKPool(adj::AbstractMatrix, k::Int, in_channel::Int; init=glorot_uniform)
    TopKPool(adj, k, init(in_channel), similar(adj, k, k))
end

function (t::TopKPool)(X::AbstractArray)
    y = t.p' * X / norm(t.p)
    idx = topk_index(y, t.k)
    t.Ã .= view(t.A, idx, idx)
    X_ = view(X, :, idx) .* σ.(view(y, idx)')
    return X_
end

function topk_index(y::AbstractVector, k::Int)
    v = nlargest(k, y)
    return collect(1:length(y))[y .>= v[end]]
end

topk_index(y::Adjoint, k::Int) = topk_index(y', k)
