"""
    abstract type GNNLayer <: AbstractLuxLayer end

An abstract type from which graph neural network layers are derived.
It is Derived from Lux's `AbstractLuxLayer` type.

See also `GNNChain`.
"""
abstract type GNNLayer <: AbstractLuxLayer end

abstract type GNNContainerLayer{T} <: AbstractLuxContainerLayer{T} end

@concrete struct GNNChain <: GNNContainerLayer{(:layers,)}
    layers <: NamedTuple
end

GNNChain(xs...) = GNNChain(; (Symbol("layer_", i) => x for (i, x) in enumerate(xs))...)

function GNNChain(; kw...)
    :layers in Base.keys(kw) &&
        throw(ArgumentError("a GNNChain cannot have a named layer called `layers`"))
    nt = NamedTuple{keys(kw)}(values(kw))
    nt = map(_wrapforchain, nt)    
    return GNNChain(nt)
end

_wrapforchain(l::AbstractLuxLayer) = l
_wrapforchain(l) = Lux.WrappedFunction(l)

Base.keys(c::GNNChain) = Base.keys(getfield(c, :layers))
Base.getindex(c::GNNChain, i::Int) = c.layers[i]
Base.getindex(c::GNNChain, i::AbstractVector) = GNNChain(NamedTuple{keys(c)[i]}(Tuple(c.layers)[i]))

function Base.getproperty(c::GNNChain, name::Symbol)
    hasfield(typeof(c), name) && return getfield(c, name)
    layers = getfield(c, :layers)
    hasfield(typeof(layers), name) && return getfield(layers, name)
    throw(ArgumentError("$(typeof(c)) has no field or layer $name"))
end

Base.length(c::GNNChain) = length(c.layers)
Base.lastindex(c::GNNChain) = lastindex(c.layers)
Base.firstindex(c::GNNChain) = firstindex(c.layers)

LuxCore.outputsize(c::GNNChain) = LuxCore.outputsize(c.layers[end])

(c::GNNChain)(g::GNNGraph, x, ps, st) = _applychain(c.layers, g, x, ps.layers, st.layers)

function _applychain(layers, g::GNNGraph, x, ps, st)  # type-unstable path, helps compile times
    newst = (;)
    for (name, l) in pairs(layers)
        x, s′ = _applylayer(l, g, x, getproperty(ps, name), getproperty(st, name))
        newst = merge(newst, (; name => s′))
    end
    return x, newst
end

_applylayer(l, g::GNNGraph, x, ps, st) = l(x), (;)
_applylayer(l::AbstractLuxLayer, g::GNNGraph, x, ps, st) = l(x, ps, st)
_applylayer(l::GNNLayer, g::GNNGraph, x, ps, st) = l(g, x, ps, st)
_applylayer(l::GNNContainerLayer, g::GNNGraph, x, ps, st) = l(g, x, ps, st)
