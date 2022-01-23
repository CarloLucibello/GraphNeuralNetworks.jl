_gather(x::NamedTuple, i) = map(x -> _gather(x, i), x)
_gather(x::Tuple, i) = map(x -> _gather(x, i), x)
_gather(x::AbstractArray, i) = NNlib.gather(x, i)
_gather(x::Nothing, i) = nothing

_scatter(aggr, m::NamedTuple, t; dstsize=nothing) = map(m -> _scatter(aggr, m, t; dstsize), m)
_scatter(aggr, m::Tuple, t; dstsize=nothing) = map(m -> _scatter(aggr, m, t; dstsize), m)
_scatter(aggr, m::AbstractArray, t; dstsize=nothing) = NNlib.scatter(aggr, m, t; dstsize)
_scatter(aggr, m::Nothing, t; dstsize=nothing) = nothing
