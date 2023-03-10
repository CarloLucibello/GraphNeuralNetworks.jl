ofeltype(x, y) = convert(float(eltype(x)), y)

"""
    reduce_nodes(aggr, g, x)

For a batched graph `g`, return the graph-wise aggregation of the node
features `x`. The aggregation operator `aggr` can be `+`, `mean`, `max`, or `min`.
The returned array will have last dimension `g.num_graphs`.
"""
function reduce_nodes(aggr, g::GNNGraph, x)
    @assert size(x)[end] == g.num_nodes
    indexes = graph_indicator(g)
    return NNlib.scatter(aggr, x, indexes)
end

"""
    reduce_edges(aggr, g, e)

For a batched graph `g`, return the graph-wise aggregation of the edge
features `e`. The aggregation operator `aggr` can be `+`, `mean`, `max`, or `min`.
The returned array will have last dimension `g.num_graphs`.
"""
function reduce_edges(aggr, g::GNNGraph, e)
    @assert size(e)[end] == g.num_edges
    s, t = edge_index(g)
    indexes = graph_indicator(g)[s]
    return NNlib.scatter(aggr, e, indexes)
end

"""
    softmax_nodes(g, x)

Graph-wise softmax of the node features `x`.
"""
function softmax_nodes(g::GNNGraph, x)
    @assert size(x)[end] == g.num_nodes
    gi = graph_indicator(g)
    max_ = gather(scatter(max, x, gi), gi)
    num = exp.(x .- max_)
    den = reduce_nodes(+, g, num)
    den = gather(den, gi)
    return num ./ den
end

"""
    softmax_edges(g, e)

Graph-wise softmax of the edge features `e`.
"""
function softmax_edges(g::GNNGraph, e)
    @assert size(e)[end] == g.num_edges
    gi = graph_indicator(g, edges = true)
    max_ = gather(scatter(max, e, gi), gi)
    num = exp.(e .- max_)
    den = reduce_edges(+, g, num)
    den = gather(den, gi)
    return num ./ (den .+ eps(eltype(e)))
end

@doc raw"""
    softmax_edge_neighbors(g, e)

Softmax over each node's neighborhood of the edge features `e`.

```math
\mathbf{e}'_{j\to i} = \frac{e^{\mathbf{e}_{j\to i}}}
                    {\sum_{j'\in N(i)} e^{\mathbf{e}_{j'\to i}}}.
```
"""
function softmax_edge_neighbors(g::GNNGraph, e)
    @assert size(e)[end] == g.num_edges
    s, t = edge_index(g)
    max_ = gather(scatter(max, e, t), t)
    num = exp.(e .- max_)
    den = gather(scatter(+, num, t), t)
    return num ./ den
end

"""
    broadcast_nodes(g, x)

Graph-wise broadcast array `x` of size `(*, g.num_graphs)` 
to size `(*, g.num_nodes)`.
"""
function broadcast_nodes(g::GNNGraph, x)
    @assert size(x)[end] == g.num_graphs
    gi = graph_indicator(g)
    return gather(x, gi)
end

"""
    broadcast_edges(g, x)

Graph-wise broadcast array `x` of size `(*, g.num_graphs)` 
to size `(*, g.num_edges)`.
"""
function broadcast_edges(g::GNNGraph, x)
    @assert size(x)[end] == g.num_graphs
    gi = graph_indicator(g, edges = true)
    return gather(x, gi)
end

function _sort_col(matrix::AbstractArray; rev::Bool = true, sortby::Int = 1)
    index = sortperm(view(matrix, sortby, :); rev)
    return matrix[:, index]
end

function _topk_matrix(matrix::AbstractArray, k::Int; rev::Bool = true, sortby = nothing)
    if sortby === nothing
        return sort(matrix, dims = 2; rev)[:, 1:k]
    else
        return _sort_col(matrix; rev, sortby)[:, 1:k]
    end
end

function _topk_batch(matrices::AbstractArray, k::Int; rev::Bool = true,
                     sortby = nothing)
    sorted_matrix = map(x -> _topk_matrix(x, k; rev, sortby), matrices)
    return reduce(hcat, sorted_matrix)
end

"""
    topk_nodes(g, feat, k; rev = true, sortby = nothing)

Graph-wise top-k on node features `feat` according to the `sortby` feature index.
"""
function topk_nodes(g::GNNGraph, feat::Symbol, k::Int; rev = true, sortby = nothing)
    if g.num_graphs == 1
        matrix = getproperty(g.ndata, feat)
        return _topk_matrix(matrix, k; rev, sortby)
    else
        graphs = [getgraph(g, i) for i in 1:(g.num_graphs)]
        matrices = map(graph -> getproperty(graph.ndata, feat), graphs)
        return _topk_batch(matrices, k; rev, sortby)
    end
end

"""
    topk_edges(g, feat, k; rev = true, sortby = nothing)

Graph-wise top-k on edge features `feat` according to the `sortby` feature index.
"""
function topk_edges(g::GNNGraph, feat::Symbol, k::Int; rev = true, sortby = nothing)
    if g.num_graphs == 1
        matrix = getproperty(g.edata, feat)
        return _topk_matrix(matrix, k; rev, sortby)
    else
        graphs = [getgraph(g, i) for i in 1:(g.num_graphs)]
        matrices = map(graph -> getproperty(graph.edata, feat), graphs)
        return _topk_batch(matrices, k; rev, sortby)
    end
end
