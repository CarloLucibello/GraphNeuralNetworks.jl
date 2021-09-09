# Adapted message passing from paper 
# "Relational inductive biases, deep learning, and graph networks"

"""
    propagate(l, g, aggr, [X, E, U]) -> X′, E′, U′
    propagate(l, g, aggr) -> g′

Perform the sequence of operations implementing the message-passing scheme
of gnn layer `l` on graph `g` . 
Updates the node, edge, and global features `X`, `E`, and `U` respectively.

The computation involved is the following:

```julia
M = compute_batch_message(l, g, X, E, U) 
M̄ = aggregate_neighbors(l, aggr, g, M)
X′ = update(l, X, M̄, U)
E′ = update_edge(l, E, M, U)
U′ = update_global(l, U, X′, E′)
```

Custom layers typically define their own [`update`](@ref)
and [`message`](@ref) functions, then call
this method in the forward pass:

```julia
function (l::MyLayer)(g, X)
    ... some prepocessing if needed ...
    propagate(l, g, +, X, E, U)
end
```

See also [`message`](@ref) and [`update`](@ref).
"""
function propagate end 

function propagate(l, g::GNNGraph, aggr)
    X, E, U = propagate(l, g, aggr, 
                node_features(g), edge_features(g), global_features(g))
    
    return GNNGraph(g, ndata=X, edata=E, gdata=U)
end

function propagate(l, g::GNNGraph, aggr, X, E=nothing, U=nothing)
    # TODO consider g.graph_indicator in propagating U
    M = compute_batch_message(l, g, X, E, U) 
    M̄ = aggregate_neighbors(l, g, aggr, M)
    X′ = update(l, X, M̄, U)
    E′ = update_edge(l, E, M, U)
    U′ = update_global(l, U, X′, E′)
    return X′, E′, U′
end

"""
    message(l, x_i, x_j, [e_ij, u])

Message function for the message-passing scheme,
returning the message from node `j` to node `i` .
In the message-passing scheme, the incoming messages 
from the neighborhood of `i` will later be aggregated
in order to [`update`](@ref) the features of node `i`.

By default, the function returns `x_j`.
Custom layer should specialize this method with the desired behavior.

# Arguments

- `l`: A gnn layer.
- `x_i`: Features of the central node `i`.
- `x_j`: Features of the neighbor `j` of node `i`.
- `e_ij`: Features of edge (`i`, `j`).
- `u`: Global features.

See also [`update`](@ref) and [`propagate`](@ref).
"""
function message end 

"""
    update(l, x, m̄, [u])

Update function for the message-passing scheme,
returning a new set of node features `x′` based on old 
features `x` and the incoming message from the neighborhood
aggregation `m̄`.

By default, the function returns `m̄`.
Custom layers should  specialize this method with the desired behavior.

# Arguments

- `l`: A gnn layer.
- `m̄`: Aggregated edge messages from the [`message`](@ref) function.
- `x`: Node features to be updated.
- `u`: Global features.

See also [`message`](@ref) and [`propagate`](@ref).
"""
function update end

_gather(x, i) = NNlib.gather(x, i)
_gather(x::Nothing, i) = nothing

## Step 1.

function compute_batch_message(l, g, X, E, U)
    s, t = edge_index(g)
    Xi = _gather(X, t)
    Xj = _gather(X, s)
    M = message(l, Xi, Xj, E, U)
    return M
end

@inline message(l, x_i, x_j, e_ij, u) = message(l, x_i, x_j, e_ij)
@inline message(l, x_i, x_j, e_ij) = message(l, x_i, x_j)
@inline message(l, x_i, x_j) = x_j

##  Step 2

function aggregate_neighbors(l, g, aggr, E)
    s, t = edge_index(g)
    NNlib.scatter(aggr, E, t)
end

aggregate_neighbors(l, g, aggr::Nothing, E) = nothing

## Step 3

@inline update(l, x, m̄, u) = update(l, x, m̄)
@inline update(l, x, m̄) = m̄

## Step 4

@inline update_edge(l, E, M, U) = update_edge(l, E, M)
@inline update_edge(l, E, M) = E

## Step 5

@inline update_global(l, U, X, E) = update_global(l, U, X)
@inline update_global(l, U, X) = U

### end steps ###
