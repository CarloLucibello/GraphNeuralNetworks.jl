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
    WithGraph(model, g::GNNGraph; traingraph=false) 

A type wrapping the `model` and tying it to the graph `g`.
In the forward pass, can only take feature arrays as inputs,
returning `model(g, x...; kws...)`.

If `traingraph=false`, the graph's parameters, won't be collected
when calling `Flux.params` on a `WithGraph` object.

# Examples

```julia
g = GNNGraph([1,2,3], [2,3,1])
x = rand(Float32, 2, 3)
model = SAGEConv(2 => 3)
wg = WithGraph(model, g)
# No need to feed the graph to `wg`
@assert wg(x) == model(g, x)

g2 = GNNGraph([1,1,2,3], [2,4,1,1])
x2 = rand(Float32, 2, 4)
# WithGraph will ignore the internal graph if fed with a new one. 
@assert wg(g2, x2) == model(g2, x2)
```
"""
struct WithGraph{M, G<:GNNGraph}
    model::M
    g::G
    traingraph::Bool
end

WithGraph(model, g::GNNGraph; traingraph=false) = WithGraph(model, g, traingraph)

@functor WithGraph
Flux.trainable(l::WithGraph) = l.traingraph ? (l.model, l.g) : (l.model,)

# Work around 
# https://github.com/FluxML/Flux.jl/issues/1733
# Revisit after 
# https://github.com/FluxML/Flux.jl/pull/1742
function Flux.destructure(m::WithGraph)
    @assert m.traingraph == false # TODO
    p, re = Flux.destructure(m.model)
    function  re_withgraph(x)
        WithGraph(re(x), m.g, m.traingraph)        
    end

    return p, re_withgraph
end

(l::WithGraph)(g::GNNGraph, x...; kws...) = l.model(g, x...; kws...)
(l::WithGraph)(x...; kws...) = l.model(l.g, x...; kws...)


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

# input from graph
applylayer(l, g::GNNGraph) = GNNGraph(g, ndata=l(node_features(g)))
applylayer(l::GNNLayer, g::GNNGraph) = l(g)

# explicit input
applylayer(l, g::GNNGraph, x) = l(x)
applylayer(l::GNNLayer, g::GNNGraph, x) = l(g, x)

# Handle Flux.Parallel
applylayer(l::Parallel, g::GNNGraph) = GNNGraph(g, ndata=applylayer(l, g, node_features(g)))
applylayer(l::Parallel, g::GNNGraph, x::AbstractArray) = mapreduce(f -> applylayer(f, g, x), l.connection, l.layers)

# input from graph
applychain(::Tuple{}, g::GNNGraph) = g
applychain(fs::Tuple, g::GNNGraph) = applychain(tail(fs), applylayer(first(fs), g))

# explicit input
applychain(::Tuple{}, g::GNNGraph, x) = x
applychain(fs::Tuple, g::GNNGraph, x) = applychain(tail(fs), g, applylayer(first(fs), g, x))

(c::GNNChain)(g::GNNGraph, x) = applychain(Tuple(c.layers), g, x)
(c::GNNChain)(g::GNNGraph) = applychain(Tuple(c.layers), g)


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


"""
    DotDecoder()

A graph neural network layer that 
for given input graph `g` and node features `x`,
returns the dot product `x_i ⋅ xj` on each edge. 

# Usage 

```juliarepl

```
"""
struct DotDecoder <: GNNLayer end 

function (::DotDecoder)(g, x)
    apply_edges(xi_dot_xj, g, xi=x, xj=x)
end
