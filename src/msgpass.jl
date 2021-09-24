"""
    propagate(f, g, aggr; xi, xj, e)  ->  m̄

Performs the message passing scheme on graph `g`.
Returns the aggregated node features `m̄` computed 

The computational steps are the following:

```julia
m = apply_edges(f, g, xi, xj, e)
m̄ = aggregate_neighbors(g, aggr, m)
```

GNN layers typically call propagate in their forward pass.

# Arguments

- `f`: A generic function that will be passed over to [`apply_edges`](@ref). 
      Takes as inputs `xi`, `xj`, and `e`
       (target nodes' features, source nodes' features, and edge features
       respetively) and returns new edge features `m`.

# Usage example

```julia
using GraphNeuralNetworks, Flux

struct GNNConv <: GNNLayer
    W
    b
    σ
end

Flux.@functor GNNConv

function GNNConv(ch::Pair{Int,Int}, σ=identity)
    in, out = ch
    W = Flux.glorot_uniform(out, in)
    b = zeros(Float32, out)
    GNNConv(W, b, σ)
end

function (l::GNNConv)(g::GNNGraph, x::AbstractMatrix)
    message(xi, xj, e) = l.W * xj
    m̄ = propagate(message, g, +, xj=x)
    return l.σ.(m̄ .+ l.bias)
end

l = GNNConv(10 => 20)
l(g, x)
```

See also [`apply_edges`](@ref).
"""
function propagate end 

propagate(l, g::GNNGraph, aggr; xi=nothing, xj=nothing, e=nothing) = 
    propagate(l, g, aggr, xi, xj, e)

function propagate(l, g::GNNGraph, aggr, xi, xj, e)
    m = apply_edges(l, g, xi, xj, e) 
    m̄ = aggregate_neighbors(g, aggr, m)
    return m̄
end

## APPLY EDGES

"""
    apply_edges(f, xi, xj, e)

Message function for the message-passing scheme
started by [`propagate`](@ref).
Returns the message from node `j` to node `i` .
In the message-passing scheme, the incoming messages 
from the neighborhood of `i` will later be aggregated
in order to update the features of node `i`.

The function operates on batches of edges, therefore
`xi`, `xj`, and `e` are tensors whose last dimension
is the batch size, or can be tuple/namedtuples of 
such tensors, according to the input to propagate.

By default, the function returns `xj`.
Custom layer should specialize this method with the desired behavior.

# Arguments

- `f`: A function that takes as inputs `xi`, `xj`, and `e`
    (target nodes' features, source nodes' features, and edge features
    respetively) and returns new edge features `m`.
- `xi`: Features of the central node `i`.
- `xj`: Features of the neighbor `j` of node `i`.
- `eij`: Features of edge `(i,j)`.

See also [`propagate`](@ref).
"""
function apply_edges end 

apply_edges(l, g::GNNGraph; xi=nothing, xj=nothing, e=nothing) = 
    apply_edges(l, g, xi, xj, e)

function apply_edges(f, g::GNNGraph, xi, xj, e)
    s, t = edge_index(g)
    xi = _gather(xi, t)   # size: (D, num_nodes) -> (D, num_edges)
    xj = _gather(xj, s)
    m = f(xi, xj, e)
    return m
end

_gather(x::NamedTuple, i) = map(x -> _gather(x, i), x)
_gather(x::Tuple, i) = map(x -> _gather(x, i), x)
_gather(x::AbstractArray, i) = NNlib.gather(x, i)
_gather(x::Nothing, i) = nothing


##  AGGREGATE NEIGHBORS

function aggregate_neighbors(g::GNNGraph, aggr, m)
    s, t = edge_index(g)
    return _scatter(aggr, m, t)
end

_scatter(aggr, m::NamedTuple, t) = map(m -> _scatter(aggr, m, t), m)
_scatter(aggr, m::Tuple, t) = map(m -> _scatter(aggr, m, t), m)
_scatter(aggr, m::AbstractArray, t) = NNlib.scatter(aggr, m, t)



### SPECIALIZATIONS OF PROPAGATE ###
copyxi(xi, xj, e) = xi
copyxj(xi, xj, e) = xj
ximulxj(xi, xj, e) = xi .* xj
xiaddxj(xi, xj, e) = xi .+ xj

function propagate(::typeof(copyxj), g::GNNGraph, ::typeof(+), xi, xj, e)
    A = adjacency_matrix(g)
    return xj * A
end

# TODO divide  by degree
# propagate(::typeof(copyxj), g::GNNGraph, ::typeof(mean), xi, xj, e)

