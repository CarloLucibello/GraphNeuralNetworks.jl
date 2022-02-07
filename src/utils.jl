ofeltype(x, y) = convert(float(eltype(x)), y)

# function NNlib.scatter(op,
#                 src::Number,
#                 idx::AbstractArray{Tidx,Nidx};
#                 init = nothing, dstsize = nothing) where {Tsrc,Tidx,Nsrc,Nidx}
    
#     dims = Nsrc - Nidx
#     dstsz = isnothing(dstsize) ? (size(src)[1:dims]..., NNlib.maximum_dims(idx)...) : dstsize 
#     dst = similar(src, Tsrc, dstsz)
#     xinit = isnothing(init) ? NNlib.scatter_empty(op, Tsrc) : init 
#     fill!(dst, xinit)
#     NNlib.scatter!(op, dst, src, idx)
# end

# Considers the src a zero dimensional object.
# Useful for implementing `StatsBase.counts`, `degree`, etc...
# function NNlib.scatter!(op, dst::AbstractArray, src::Number, idx::AbstractArray)
#     for k in CartesianIndices(idx)
#         # dst_v = NNlib._view(dst, idx[k])
#         # dst_v .= (op).(dst_v, src)
#         dst[idx[k]] .= (op).(dst[idx[k]], src)
#     end
#     dst
# end

# 10 time faster than the generic version above. 
# All the speedup comes from not broadcasting `op`, i dunno why.
function NNlib.scatter!(op, dst::AbstractVector, src::Number, idx::AbstractVector{<:Integer})
    for i in idx
        dst[i] = op(dst[i], src)
    end
end

# NNlib._view(X, k) = view(X, k...)
# NNlib._view(X, k::Union{Integer, CartesianIndex}) = view(X,  k)

# Considers src as a zero dimensional object to be scattered
# function NNlib.scatter(op,
#                 src::Tsrc,
#                 idx::AbstractArray{Tidx,Nidx};
#                 init = nothing, dstsize = nothing) where {Tsrc<:Number,Tidx,Nidx}
    
#     dstsz = isnothing(dstsize) ? maximum_dims(idx) : dstsize 
#     dst = similar(src, Tsrc, dstsz)
#     xinit = isnothing(init) ? scatter_empty(op, Tsrc) : init 
#     fill!(dst, xinit)
#     scatter!(op, dst, src, idx)
# end


function scatter_scalar_kernel!(op, dst, src, idx)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= length(idx)
        CUDA.@atomic dst[idx[index]...] = op(dst[idx[index]...], src)
    end
    return nothing
end

function NNlib.scatter!(op, dst::AnyCuArray, src::Number, idx::AnyCuArray)
    max_idx = length(idx)
    args = op, dst, src, idx
    
    kernel = @cuda launch=false scatter_scalar_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(max_idx, config.threads)
    blocks = cld(max_idx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return dst
end

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
    gi = graph_indicator(g, edges=true)
    max_ = gather(scatter(max, e, gi), gi)
    num = exp.(e .- max_)
    den = reduce_edges(+, g, num)
    den = gather(den, gi)
    return num ./ den
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
    gi = graph_indicator(g, edges=true)
    return gather(x, gi)
end

