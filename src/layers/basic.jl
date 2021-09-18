"""
    abstract type GNNLayer end

An abstract type from which graph neural network layers are derived.

See also [`GNNChain`](@ref).
"""
abstract type GNNLayer end

# Forward pass with graph-only input.
# To be specialized by layers also needing edge features as input (e.g. NNConv). 
(l::GNNLayer)(g::GNNGraph) = GNNGraph(g, ndata=l(g, node_features(g)))

"""
    GNNChain(layers...)
    GNNChain(name = layer, ...)

Collects multiple layers / functions to be called in sequence
on given input graph and input node features. 

It allows to compose layers in a sequential fashion as `Flux.Chain`
does, propagating the output of each layer to the next one.
In addition, `GNNChain` handles the input graph as well, providing it 
as a first argument only to layers subtyping the [`GNNLayer`](@ref) abstract type. 

`GNNChain` supports indexing and slicing, `m[2]` or `m[1:end-1]`,
and if names are given, `m[:name] == m[1]` etc.

# Examples

```juliarepl
julia> m = GNNChain(GCNConv(2=>5), BatchNorm(5), x -> relu.(x), Dense(5, 4));

julia> x = randn(Float32, 2, 3);

julia> g = GNNGraph([1,1,2,3], [2,3,1,1]);

julia> m(g, x)
4×3 Matrix{Float32}:
  0.157941    0.15443     0.193471
  0.0819516   0.0503105   0.122523
  0.225933    0.267901    0.241878
 -0.0134364  -0.0120716  -0.0172505
```
"""
struct GNNChain{T} <: GNNLayer
  layers::T

  GNNChain(xs...) = new{typeof(xs)}(xs)
  
  function GNNChain(; kw...)
    :layers in Base.keys(kw) && throw(ArgumentError("a GNNChain cannot have a named layer called `layers`"))
    isempty(kw) && return new{Tuple{}}(())
    new{typeof(values(kw))}(values(kw))
  end
end

@forward GNNChain.layers Base.getindex, Base.length, Base.first, Base.last,
    Base.iterate, Base.lastindex, Base.keys

Flux.functor(::Type{<:GNNChain}, c) = c.layers, ls -> GNNChain(ls...)

applylayer(l, g::GNNGraph, x) = l(x)
applylayer(l::GNNLayer, g::GNNGraph, x) = l(g, x)

applychain(::Tuple{}, g::GNNGraph, x) = x
applychain(fs::Tuple, g::GNNGraph, x) = applychain(tail(fs), g, applylayer(first(fs), g, x))

(c::GNNChain)(g::GNNGraph, x) = applychain(Tuple(c.layers), g, x)

Base.getindex(c::GNNChain, i::AbstractArray) = GNNChain(c.layers[i]...)
Base.getindex(c::GNNChain{<:NamedTuple}, i::AbstractArray) = 
    GNNChain(; NamedTuple{Base.keys(c)[i]}(Tuple(c.layers)[i])...)

function Base.show(io::IO, c::GNNChain)
    print(io, "GNNChain(")
    _show_layers(io, c.layers)
    print(io, ")")
end
_show_layers(io, layers::Tuple) = join(io, layers, ", ")
_show_layers(io, layers::NamedTuple) = join(io, ["$k = $v" for (k, v) in pairs(layers)], ", ")