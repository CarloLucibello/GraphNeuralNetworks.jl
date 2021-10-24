function check_num_nodes(g::GNNGraph, x::AbstractArray)
    @assert g.num_nodes == size(x, ndims(x))    
end
function check_num_edges(g::GNNGraph, e::AbstractArray)
    @assert g.num_edges == size(e, ndims(e))    
end

sort_edge_index(eindex::Tuple) = sort_edge_index(eindex...)

function sort_edge_index(u, v)
    uv = collect(zip(u, v))
    p = sortperm(uv) # isless lexicographically defined for tuples
    return u[p], v[p]
end

cat_features(x1::Nothing, x2::Nothing) = nothing 
cat_features(x1::AbstractArray, x2::AbstractArray) = cat(x1, x2, dims=ndims(x1))
cat_features(x1::Union{Number, AbstractVector}, x2::Union{Number, AbstractVector}) = 
    cat(x1, x2, dims=1)


function cat_features(x1::NamedTuple, x2::NamedTuple)
    sort(collect(keys(x1))) == sort(collect(keys(x2))) ||
        @error "cannot concatenate feature data with different keys"
    
    NamedTuple(k => cat_features(getfield(x1,k), getfield(x2,k)) for k in keys(x1))
end

# Turns generic type into named tuple
normalize_graphdata(data::Nothing; kws...) = NamedTuple()

normalize_graphdata(data; default_name::Symbol, kws...) = 
    normalize_graphdata(NamedTuple{(default_name,)}((data,)); default_name, kws...) 

function normalize_graphdata(data::NamedTuple; default_name, n, duplicate_if_needed=false)
    # This had to workaround two Zygote bugs with NamedTuples
    # https://github.com/FluxML/Zygote.jl/issues/1071
    # https://github.com/FluxML/Zygote.jl/issues/1072
    
    if n == 1
        # If last array dimension is not 1, add a new dimension. 
        # This is mostly usefule to reshape globale feature vectors
        # of size D to Dx1 matrices.
        function unsqz(v)
            if v isa AbstractArray && size(v)[end] != 1
                v = reshape(v, size(v)..., 1)
            end
            v
        end

        data = NamedTuple{keys(data)}(unsqz.(values(data)))
    end
    
    sz = map(x -> x isa AbstractArray ? size(x)[end] : 0, data)
    if duplicate_if_needed 
        # Used to copy edge features on reverse edges    
        @assert all(s -> s == 0 ||  s == n || s == n÷2, sz)

        function duplicate(v)
            if v isa AbstractArray && size(v)[end] == n÷2
                v = cat(v, v, dims=ndims(v))
            end
            v
        end

        data = NamedTuple{keys(data)}(duplicate.(values(data)))
    else
        @assert all(s -> s == 0 ||  s == n, sz)
    end
    return data
end

zeros_like(x::AbstractArray, T=eltype(x), sz=size(x)) = fill!(similar(x, T, sz), 0)
zeros_like(x::SparseMatrixCSC, T=eltype(x), sz=size(x)) = zeros(T, sz)
zeros_like(x::CUMAT_T, T=eltype(x), sz=size(x)) = CUDA.zeros(T, sz)

ones_like(x::AbstractArray, T=eltype(x), sz=size(x)) = fill!(similar(x, T, sz), 1)
ones_like(x::SparseMatrixCSC, T=eltype(x), sz=size(x)) = ones(T, sz)
ones_like(x::CUMAT_T, T=eltype(x), sz=size(x)) = CUDA.ones(T, sz)

ofeltype(x, y) = convert(float(eltype(x)), y)

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
                    {\sum_{j'\in N(i)} e^{\mathbf{e}_{j\to i}}}.
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


function graph_indicator(g; edges=false)
    if isnothing(g.graph_indicator)
        gi = ones_like(edge_index(g)[1], Int, g.num_nodes)
    else 
        gi = g.graph_indicator
    end
    if edges
        s, t = edge_index(g)
        return gi[s]
    else
        return gi
    end
end
