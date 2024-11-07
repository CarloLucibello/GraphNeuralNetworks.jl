"""
    abstract type GNNLayer <: AbstractLuxLayer end

An abstract type from which graph neural network layers are derived.
It is derived from Lux's `AbstractLuxLayer` type.

See also [`GNNLux.GNNChain`](@ref).
"""
abstract type GNNLayer <: AbstractLuxLayer end

abstract type GNNContainerLayer{T} <: AbstractLuxContainerLayer{T} end

"""
    GNNChain(layers...)
    GNNChain(name = layer, ...)

Collects multiple layers / functions to be called in sequence
on given input graph and input node features. 

It allows to compose layers in a sequential fashion as `Lux.Chain`
does, propagating the output of each layer to the next one.
In addition, `GNNChain` handles the input graph as well, providing it 
as a first argument only to layers subtyping the [`GNNLayer`](@ref) abstract type. 

`GNNChain` supports indexing and slicing, `m[2]` or `m[1:end-1]`,
and if names are given, `m[:name] == m[1]` etc.

# Examples
```jldoctest
julia> using Lux, GNNLux, Random

julia> rng = Random.default_rng();

julia> m = GNNChain(GCNConv(2=>5), 
                    x -> relu.(x), 
                    Dense(5=>4))

julia> x = randn(rng, Float32, 2, 3);

julia> g = rand_graph(3, 6)
GNNGraph:
  num_nodes: 3
  num_edges: 6

julia> ps, st = LuxCore.setup(rng,m);

julia> m(g,x,ps,st)     # First entry is the output, second entry is the state of the model
(Float32[-0.15594329 -0.15594329 -0.15594329; 0.93431795 0.93431795 0.93431795; 0.27568763 0.27568763 0.27568763; 0.12568939 0.12568939 0.12568939], (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()))
```
"""
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
