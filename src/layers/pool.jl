using DataStructures: nlargest

@doc raw"""
    GlobalPool(aggr)

Global pooling layer for graph neural networks.
Takes a graph and feature nodes as inputs
and performs the operation

```math
\mathbf{u}_V = \square_{i \in V} \mathbf{x}_i
```
where ``V`` is the set of nodes of the input graph and 
the type of aggregation represented by ``\square`` is selected by the `aggr` argument. 
Commonly used aggregations are `mean`, `max`, and `+`.

See also [`reduce_nodes`](@ref).

# Examples
```julia
using Flux, GraphNeuralNetworks, LightGraphs

pool = GlobalPool(mean)

g = GNNGraph(erdos_renyi(10, 4))
X = rand(32, 10)
pool(g, X) # => 32x1 matrix


g = Flux.batch([GNNGraph(erdos_renyi(10, 4)) for _ in 1:5])
X = rand(32, 50)
pool(g, X) # => 32x5 matrix
```
"""
struct GlobalPool{F} <: GNNLayer
    aggr::F
end

function (l::GlobalPool)(g::GNNGraph, x::AbstractArray)
    return reduce_nodes(l.aggr, g, x)
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
