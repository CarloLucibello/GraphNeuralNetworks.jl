"""
    abstract type GNNLayer end

An abstract type from which graph neural network layers are derived.

See also [`GNNChain`](@ref).
"""
abstract type GNNLayer end

"""
    GNNChain(layers...)
    GNNChain(name = layer, ...)

Collects multiple layers / functions to be called in sequence
on a given input. Supports indexing and slicing, `m[2]` or `m[1:end-1]`,
and if names are given, `m[:name] == m[1]` etc.

## Examples

```jldoctest
julia> m = GNNChain(x -> x^2, x -> x+1);

julia> m(5) == 26
true

julia> m = GNNChain(Dense(10, 5, tanh), Dense(5, 2));

julia> x = rand(10, 32);

julia> m(x) == m[2](m[1](x))
true

julia> m2 = GNNChain(enc = GNNChain(Flux.flatten, Dense(10, 5, tanh)), 
                  dec = Dense(5, 2));

                  julia> m2(x) == (m2[:dec] ∘ m2[:enc])(x)
true
```
"""
struct GNNChain{T}
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

functor(::Type{<:GNNChain}, c) = c.layers, ls -> GNNChain(ls...)

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))

(c::GNNChain)(x) = applychain(Tuple(c.layers), x)

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