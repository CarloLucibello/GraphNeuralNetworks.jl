# Adapted message passing from paper 
# "Relational inductive biases, deep learning, and graph networks"

"""
    propagate(mp, g::GNNGraph, aggr)
    propagate(mp, g::GNNGraph, E, X, u, aggr)

Perform the sequence of operation implementing the message-passing scheme
and updating node, edge, and global features `X`, `E`, and `u` respectively.

The computation involved is the following:

```julia
M = compute_batch_message(mp, g, E, X, u) 
E = update_edge(mp, M, E, u)
M̄ = aggregate_neighbors(mp, aggr, g, M)
X = update(mp, M̄, X, u)
u = update_global(mp, E, X, u)
```

Custom layers typically define their own [`update`](@ref)
and [`message`](@ref) function, then call
this method in the forward pass:

```julia
function (l::GNNLayer)(g, X)
    ... some prepocessing if needed ...
    E = nothing
    u = nothing
    propagate(l, g, E, X, u, +)
end
```

See also [`message`](@ref) and [`update`](@ref).
"""
function propagate end 

function propagate(mp, g::GNNGraph, aggr)
    E, X, u = propagate(mp, g,
                        edge_feature(g), node_feature(g), global_feature(g), 
                        aggr)
    GNNGraph(g, nf=X, ef=E, gf=u)
end

function propagate(mp, g::GNNGraph, E, X, u, aggr)
    M = compute_batch_message(mp, g, E, X, u) 
    E = update_edge(mp, M, E, u)
    M̄ = aggregate_neighbors(mp, aggr, g, M)
    X = update(mp, M̄, X, u)
    u = update_global(mp, E, X, u)
    return E, X, u
end

"""
    message(mp, x_i, x_j, [e_ij, u])

Message function for the message-passing scheme,
returning the message from node `j` to node `i` .
In the message-passing scheme. the incoming messages 
from the neighborhood of `i` will later be aggregated
in order to [`update`](@ref) the features of node `i`.

By default, the function returns `x_j`.
Custom layer should specialize this method with the desired behavior.

## Arguments

- `mp`: A gnn layer.
- `x_i`: Features of the central node `i`.
- `x_j`: Features of the neighbor `j` of node `i`.
- `e_ij`: Features of edge (`i`, `j`).
- `u`: Global features.

See also [`update`](@ref) and [`propagate`](@ref).
"""
function message end 

"""
    update(mp, m̄, x, [u])

Update function for the message-passing scheme,
returning a new set of node features `x′` based on old 
features `x` and the incoming message from the neighborhood
aggregation `m̄`.

By default, the function returns `m̄`.
Custom layers should  specialize this method with the desired behavior.

## Arguments

- `mp`: A gnn layer.
- `m̄`: Aggregated edge messages from the [`message`](@ref) function.
- `x`: Node features to be updated.
- `u`: Global features.

See also [`message`](@ref) and [`propagate`](@ref).
"""
function update end


_gather(x, i) = NNlib.gather(x, i)
_gather(x::Nothing, i) = nothing

## Step 1.

function compute_batch_message(mp, g, E, X, u)
    s, t = edge_index(g)
    Xi = _gather(X, t)
    Xj = _gather(X, s)
    M = message(mp, Xi, Xj, E, u)
    return M
end

# @inline message(mp, i, j, x_i, x_j, e_ij, u) = message(mp, x_i, x_j, e_ij, u) # TODO add in the future
@inline message(mp, x_i, x_j, e_ij, u) = message(mp, x_i, x_j, e_ij)
@inline message(mp, x_i, x_j, e_ij) = message(mp, x_i, x_j)
@inline message(mp, x_i, x_j) = x_j

## Step 2

@inline update_edge(mp, M, E, u) = update_edge(mp, M, E)
@inline update_edge(mp, M, E) = E

##  Step 3

function aggregate_neighbors(mp, aggr, g, E)
    s, t = edge_index(g)
    NNlib.scatter(aggr, E, t)
end

aggregate_neighbors(mp, aggr::Nothing, g, E) = nothing

## Step 4

# @inline update(mp, i, m̄, x, u) = update(mp, m, x, u)
@inline update(mp, m̄, x, u) = update(mp, m̄, x)
@inline update(mp, m̄, x) = m̄

## Step 5

@inline update_global(mp, E, X, u) = u

### end steps ###
