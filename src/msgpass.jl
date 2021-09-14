"""
    propagate(l, g, aggr, [x, e]) -> x′, e′
    propagate(l, g, aggr) -> g′

Performs the message-passing for GNN layer `l` on graph `g` . 
Returns updated node and edge features `x` and `e`.

In case no input and edge features are given as input, 
extracts them from `g` and returns the same graph
with updated feautres.

The computational steps are the following:

```julia
m = compute_batch_message(l, g, x, e)  # calls `compute_message`
m̄ = aggregate_neighbors(g, aggr, m)
x′ = update_node(l, m̄, x)
e′ = update_edge(l, m, e)
```

Custom layers typically define their own [`update_node`](@ref)
and [`compute_message`](@ref) functions, then call
this method in the forward pass:

# Usage example

```julia
using GraphNeuralNetworks, Flux

struct GNNConv <: GNNLayer
    W
    b
    σ
end

Flux.@functor GNNConv

function GNNConv(ch::Pair{Int,Int}, σ=identity;
                 init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, in)
    b = Flux.create_bias(W, bias, out)
    GNNConv(W, b, σ, aggr)
end

compute_message(l::GNNConv, x_i, x_j, e_ij) = l.W * x_j
update_node(l::GNNConv, m̄, x) = l.σ.(m̄ .+ l.bias)

function (l::GNNConv)(g::GNNGraph, x::AbstractMatrix)
    x, _ = propagate(l, g, +, x)
    return x
end
```

See also [`compute_message`](@ref) and [`update_node`](@ref).
"""
function propagate end 

function propagate(l, g::GNNGraph, aggr)
    x, e = propagate(l, g, aggr, node_features(g), edge_features(g))
    return GNNGraph(g, ndata=x, edata=e)
end

function propagate(l, g::GNNGraph, aggr, x, e=nothing)
    m = compute_batch_message(l, g, x, e) 
    m̄ = aggregate_neighbors(g, aggr, m)
    x′ = update_node(l, m̄, x)
    e′ = update_edge(l, m, e)
    return x′, e′
end

## Step 1.

"""
    compute_message(l, x_i, x_j, [e_ij])

Message function for the message-passing scheme
started by [`propagate`](@ref).
Returns the message from node `j` to node `i` .
In the message-passing scheme, the incoming messages 
from the neighborhood of `i` will later be aggregated
in order to update (see [`update_node`](@ref)) the features of node `i`.

The function operates on batches of edges, therefore
`x_i`, `x_j`, and `e_ij` are tensors whose last dimension
is the batch size, or can be tuple/namedtuples of 
such tensors, according to the input to propagate.

By default, the function returns `x_j`.
Custom layer should specialize this method with the desired behavior.

# Arguments

- `l`: A gnn layer.
- `x_i`: Features of the central node `i`.
- `x_j`: Features of the neighbor `j` of node `i`.
- `e_ij`: Features of edge `(i,j)`.

See also [`update_node`](@ref) and [`propagate`](@ref).
"""
function compute_message end 

@inline compute_message(l, x_i, x_j, e_ij) = compute_message(l, x_i, x_j)
@inline compute_message(l, x_i, x_j) = x_j

_gather(x::NamedTuple, i) = map(x -> _gather(x, i), x)
_gather(x::Tuple, i) = map(x -> _gather(x, i), x)
_gather(x::AbstractArray, i) = NNlib.gather(x, i)
_gather(x::Nothing, i) = nothing

function compute_batch_message(l, g::GNNGraph, x, e)
    s, t = edge_index(g)
    xi = _gather(x, t)
    xj = _gather(x, s)
    m = compute_message(l, xi, xj, e)
    return m
end

##  Step 2

_scatter(aggr, e::NamedTuple, t) = map(e -> _scatter(aggr, e, t), e)
_scatter(aggr, e::Tuple, t) = map(e -> _scatter(aggr, e, t), e)
_scatter(aggr, e::AbstractArray, t) = NNlib.scatter(aggr, e, t)
_scatter(aggr, e::Nothing, t) = nothing

function aggregate_neighbors(g::GNNGraph, aggr, e)
    s, t = edge_index(g)
    _scatter(aggr, e, t)
end

aggregate_neighbors(g::GNNGraph, aggr::Nothing, e) = nothing

## Step 3

"""
    update_node(l, m̄, x)

Node update function for the GNN layer `l`,
returning a new set of node features `x′` based on old 
features `x` and the aggregated message `m̄` from the neighborhood.

The input `m̄` is an array, a tuple or a named tuple, 
reflecting the output of [`compute_message`](@ref).

By default, the function returns `m̄`.
Custom layers should  specialize this method with the desired behavior.

See also [`compute_message`](@ref), [`update_edge`](@ref), and [`propagate`](@ref).
"""
function update_node end

@inline update_node(l, m̄, x) = m̄

## Step 4


"""
    update_edge(l, m, e)

Edge update function for the GNN layer `l`,
returning a new set of edge features `e′` based on old 
features `e` and the newly computed messages `m`
from the [`compute_message`](@ref) function.

By default, the function returns `e`.
Custom layers should specialize this method with the desired behavior.

See also [`compute_message`](@ref), [`update_node`](@ref), and [`propagate`](@ref).
"""
function update_edge end

@inline update_edge(l, m, e) = e

### end steps ###
