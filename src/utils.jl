ofeltype(x, y) = convert(float(eltype(x)), y)

"""
    reduce_nodes(aggr, g, x)

For a batched graph `g`, return the graph-wise aggregation of the node
features `x`. The aggregation operator `aggr` can be `+`, `mean`, `max`, or `min`.
The returned array will have last dimension `g.num_graphs`.

See also: [`reduce_edges`](@ref).
"""
function reduce_nodes(aggr, g::GNNGraph, x)
    @assert size(x)[end] == g.num_nodes
    indexes = graph_indicator(g)
    return NNlib.scatter(aggr, x, indexes)
end

"""
    reduce_nodes(aggr, indicator::AbstractVector, x)

Return the graph-wise aggregation of the node features `x` given the
graph indicator `indicator`. The aggregation operator `aggr` can be `+`, `mean`, `max`, or `min`.

See also [`graph_indicator`](@ref).
"""
function reduce_nodes(aggr, indicator::AbstractVector, x)
    return NNlib.scatter(aggr, x, indicator)
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
function softmax_edge_neighbors(g::AbstractGNNGraph, e)
    if g isa GNNHeteroGraph
        for (key, value) in g.num_edges
            @assert size(e)[end] == value
        end
    else
        @assert size(e)[end] == g.num_edges
    end
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
    return matrix[:, index], index
end

function _topk_matrix(matrix::AbstractArray, k::Int; rev::Bool = true, sortby::Union{Nothing, Int} = nothing)
    if sortby === nothing
        sorted_matrix = sort(matrix, dims = 2; rev)[:, 1:k]
        vector_indices = map(x -> sortperm(x; rev), eachrow(matrix))
        indices = reduce(vcat, vector_indices')[:, 1:k]
        return sorted_matrix, indices
    else
        sorted_matrix, indices = _sort_col(matrix; rev, sortby)
        return sorted_matrix[:, 1:k], indices[1:k]
    end
end

function _topk_batch(matrices::AbstractArray, k::Int; rev::Bool = true,
                     sortby::Union{Nothing, Int} = nothing)
    num_graphs = length(matrices)
    num_feat = size(matrices[1], 1)
    sorted_matrix = map(x -> _topk_matrix(x, k; rev, sortby)[1], matrices)
    output_matrix = reshape(reduce(hcat, sorted_matrix), num_feat, k, num_graphs)
    indices = map(x -> _topk_matrix(x, k; rev, sortby)[2], matrices)
    if sortby === nothing
        output_indices = reshape(reduce(hcat, indices), num_feat, k, num_graphs)
    else
        output_indices = reshape(reduce(hcat, indices), k, 1, num_graphs)
    end
    return output_matrix, output_indices
end

"""
    topk_feature(g, feat, k; rev = true, sortby = nothing)

Graph-wise top-`k` on feature array `x` according to the `sortby` index.
Returns a tuple of the top-`k` features and their indices.

# Arguments

- `g`: a `GNNGraph``.
- `feat`: a feature array of size `(number_features, g.num_nodes)` or `(number_features, g.num_edges)` of the graph `g`.
- `k`: the number of top features to return.
- `rev`: if `true`, sort in descending order otherwise returns the `k` smallest elements.
- `sortby`: the index of the feature to sort by. If `nothing`, every row independently.

# Examples
    
```julia
julia> g = rand_graph(5, 4, ndata = rand(3,5));

julia> g.ndata.x
3×5 Matrix{Float64}:
 0.333661  0.683551  0.315145  0.794089   0.840085
 0.263023  0.726028  0.626617  0.412247   0.0914052
 0.296433  0.186584  0.960758  0.0999844  0.813808

julia> topk_feature(g, g.ndata.x, 2)
([0.8400845757074524 0.7940891040468462; 0.7260276789396128 0.6266174187625888; 0.9607582005024967 0.8138081223752274], [5 4; 2 3; 3 5])

julia> topk_feature(g, g.ndata.x, 2; sortby=3)
([0.3151452763177829 0.8400845757074524; 0.6266174187625888 0.09140519108918477; 0.9607582005024967 0.8138081223752274], [3, 5])

```

"""
function topk_feature(g::GNNGraph, feat::AbstractArray, k::Int; rev::Bool = true,
                      sortby::Union{Nothing, Int} = nothing)
    if g.num_graphs == 1
        return _topk_matrix(feat, k; rev, sortby)
    else
        matrices = [feat[:, g.graph_indicator .== i] for i in 1:(g.num_graphs)]
        return _topk_batch(matrices, k; rev, sortby)
    end
end

expand_srcdst(g::AbstractGNNGraph, x) = throw(ArgumentError("Invalid input type, expected matrix or tuple of matrices."))
expand_srcdst(g::AbstractGNNGraph, x::AbstractMatrix) = (x, x)
expand_srcdst(g::AbstractGNNGraph, x::Tuple{<:AbstractMatrix, <:AbstractMatrix}) = x
