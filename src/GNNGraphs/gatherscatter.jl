_gather(x::NamedTuple, i) = map(x -> _gather(x, i), x)
_gather(x::Dict, i) = Dict(k => _gather(v, i) for (k, v) in x)
_gather(x::Tuple, i) = map(x -> _gather(x, i), x)
_gather(x::AbstractArray, i) = NNlib.gather(x, i)
_gather(x::Nothing, i) = nothing

_scatter(aggr::A, src::Nothing, idx, n) where A = nothing
_scatter(aggr::A, src::NamedTuple, idx, n) where A = map(s -> _scatter(aggr, s, idx, n), src)
_scatter(aggr::A, src::Tuple, idx, n) where A = map(s -> _scatter(aggr, s, idx, n), src)
_scatter(aggr::A, src::Dict, idx, n) where A = Dict(k => _scatter(aggr, v, idx, n) for (k, v) in src)

function _scatter(aggr,
                  src::AbstractArray,
                  idx::AbstractVector{<:Integer},
                  n::Integer)
    dstsize = (size(src)[1:(end - 1)]..., n)
    return NNlib.scatter(aggr, src, idx; dstsize)
end

## TO MOVE TO NNlib ######################################################

### Considers the src a zero dimensional object.
### Useful for implementing `StatsBase.counts`, `degree`, etc...
### function NNlib.scatter!(op, dst::AbstractArray, src::Number, idx::AbstractArray)
###     for k in CartesianIndices(idx)
###         # dst_v = NNlib._view(dst, idx[k])
###         # dst_v .= (op).(dst_v, src)
###         dst[idx[k]] .= (op).(dst[idx[k]], src)
###     end
###     dst
### end

# 10 times faster than the generic version above. 
# All the speedup comes from not broadcasting `op`, i dunno why.
# function NNlib.scatter!(op, dst::AbstractVector, src::Number, idx::AbstractVector{<:Integer})
#     for i in idx
#         dst[i] = op(dst[i], src)
#     end
# end

## NNlib._view(X, k) = view(X, k...)
## NNlib._view(X, k::Union{Integer, CartesianIndex}) = view(X,  k)
#
## Considers src as a zero dimensional object to be scattered
## function NNlib.scatter(op,
##                 src::Tsrc,
##                 idx::AbstractArray{Tidx,Nidx};
##                 init = nothing, dstsize = nothing) where {Tsrc<:Number,Tidx,Nidx}   
##     dstsz = isnothing(dstsize) ? maximum_dims(idx) : dstsize 
##     dst = similar(src, Tsrc, dstsz)
##     xinit = isnothing(init) ? scatter_empty(op, Tsrc) : init 
##     fill!(dst, xinit)
##     scatter!(op, dst, src, idx)
## end

# function scatter_scalar_kernel!(op, dst, src, idx)
#     index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

#     @inbounds if index <= length(idx)
#         CUDA.@atomic dst[idx[index]...] = op(dst[idx[index]...], src)
#     end
#     return nothing
# end

# function NNlib.scatter!(op, dst::AnyCuArray, src::Number, idx::AnyCuArray)
#     max_idx = length(idx)
#     args = op, dst, src, idx

#     kernel = @cuda launch=false scatter_scalar_kernel!(args...)
#     config = launch_configuration(kernel.fun; max_threads=256)
#     threads = min(max_idx, config.threads)
#     blocks = cld(max_idx, threads)
#     kernel(args...; threads=threads, blocks=blocks)
#     return dst
# end
