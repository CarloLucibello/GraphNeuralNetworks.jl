"""
    DataStore([n, data])
    DataStore([n], k1 = x1, k2 = x2, ...)

A container for feature arrays. The optional argument `n` enforces that
`numobs(x) == n` for each array contained in the `Datastore``.

At construction time, the `data` can be provided as any iterables of pairs
of symbols and arrays or as keyword arguments:

```julia-repl
julia> ds = DataStore(3, x = rand(2, 3), y = rand(3))
DataStore(3) with 2 elements:
  y = 3-element Vector{Float64}
  x = 2×3 Matrix{Float64}

julia> ds = DataStore(3, Dict(:x => rand(2, 3), :y => rand(3))); # equivalent to above

julia> ds = DataStore(3, (x = rand(2, 3), y = rand(30)))
ERROR: AssertionError: DataStore: data[y] has 30 observations, but n = 3
Stacktrace:
 [1] DataStore(n::Int64, data::Dict{Symbol, Any})
   @ GraphNeuralNetworks.GNNGraphs ~/.julia/dev/GraphNeuralNetworks/src/GNNGraphs/datastore.jl:54
 [2] DataStore(n::Int64, data::NamedTuple{(:x, :y), Tuple{Matrix{Float64}, Vector{Float64}}})
   @ GraphNeuralNetworks.GNNGraphs ~/.julia/dev/GraphNeuralNetworks/src/GNNGraphs/datastore.jl:73
 [3] top-level scope
   @ REPL[13]:1

julia> ds = DataStore(x = rand(2, 3), y = rand(30)) # no checks
DataStore() with 2 elements:
  y = 30-element Vector{Float64}
  x = 2×3 Matrix{Float64}
```

The `DataStore` has an interface similar to both dictionaries and named tuples.
Arrays can be accessed and added using either the indexing or the property syntax:

```julia-repl
julia> ds = DataStore(x = ones(2, 3), y = zeros(3))
DataStore() with 2 elements:
  y = 3-element Vector{Float64}
  x = 2×3 Matrix{Float64}

julia> ds.x   # same as `ds[:x]`
2×3 Matrix{Float64}:
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> ds.z = zeros(3)  # Add new feature array `z`. Same as `ds[:z] = rand(3)`
3-element Vector{Float64}:
0.0
0.0
0.0
```

The `DataStore` can be iterated over, and the keys and values can be accessed
using `keys(ds)` and `values(ds)`. `map(f, ds)` applies the function `f`
to each feature array:

```julia-repl
julia> ds = DataStore(a = zeros(2), b = zeros(2));

julia> ds2 = map(x -> x .+ 1, ds)

julia> ds2.a
2-element Vector{Float64}:
 1.0
 1.0
```
"""
struct DataStore
    _n::Int # either -1 or numobs(data)
    _data::Dict{Symbol, Any}

    function DataStore(n::Int, data::Dict{Symbol, Any})
        if n >= 0
            for (k, v) in data
                @assert numobs(v)==n "DataStore: data[$k] has $(numobs(v)) observations, but n = $n"
            end
        end
        return new(n, data)
    end
end

@functor DataStore

DataStore(data) = DataStore(-1, data)
DataStore(n::Int, data::NamedTuple) = DataStore(n, Dict{Symbol, Any}(pairs(data)))
DataStore(n::Int, data) = DataStore(n, Dict{Symbol, Any}(data))

DataStore(; kws...) = DataStore(-1; kws...)
DataStore(n::Int; kws...) = DataStore(n, Dict{Symbol, Any}(kws...))

getdata(ds::DataStore) = getfield(ds, :_data)
getn(ds::DataStore) = getfield(ds, :_n)
# setn!(ds::DataStore, n::Int) = setfield!(ds, :n, n)

function Base.getproperty(ds::DataStore, s::Symbol)
    if s === :_n
        return getn(ds)
    elseif s === :_data
        return getdata(ds)
    else
        return getdata(ds)[s]
    end
end

function Base.setproperty!(ds::DataStore, s::Symbol, x)
    @assert s != :_n "cannot set _n directly"
    @assert s != :_data "cannot set _data directly"
    if getn(ds) >= 0
        numobs(x) == getn(ds) || throw(DimensionMismatch("expected $(getn(ds)) object features but got $(numobs(x))."))
    end
    return getdata(ds)[s] = x
end

Base.getindex(ds::DataStore, s::Symbol) = getproperty(ds, s)
Base.setindex!(ds::DataStore, s::Symbol, x) = setproperty!(ds, s, x)

function Base.show(io::IO, ds::DataStore)
    len = length(ds)
    n = getn(ds)
    if n < 0
        print(io, "DataStore()")
    else
        print(io, "DataStore($(getn(ds)))")
    end
    if len > 0
        print(io, " with $(length(getdata(ds))) element")
        len > 1 && print(io, "s")
        print(io, ":")
        for (k, v) in getdata(ds)
            print(io, "\n  $(k) = $(summary(v))")
        end
    end
end

Base.iterate(ds::DataStore) = iterate(getdata(ds))
Base.iterate(ds::DataStore, state) = iterate(getdata(ds), state)
Base.keys(ds::DataStore) = keys(getdata(ds))
Base.values(ds::DataStore) = values(getdata(ds))
Base.length(ds::DataStore) = length(getdata(ds))
Base.haskey(ds::DataStore, k) = haskey(getdata(ds), k)
Base.get(ds::DataStore, k, default) = get(getdata(ds), k, default)
Base.pairs(ds::DataStore) = pairs(getdata(ds))
Base.:(==)(ds1::DataStore, ds2::DataStore) = getdata(ds1) == getdata(ds2)
Base.isempty(ds::DataStore) = isempty(getdata(ds))
Base.delete!(ds::DataStore, k) = delete!(getdata(ds), k)

function Base.map(f, ds::DataStore)
    d = getdata(ds)
    newd = Dict{Symbol, Any}(k => f(v) for (k, v) in d)
    return DataStore(getn(ds), newd)
end

MLUtils.numobs(ds::DataStore) = numobs(getdata(ds))

function MLUtils.getobs(ds::DataStore, i::Int)
    newdata = getobs(getdata(ds), i)
    return DataStore(-1, newdata)
end

function MLUtils.getobs(ds::DataStore,
                        i::AbstractVector{T}) where {T <: Union{Integer, Bool}}
    newdata = getobs(getdata(ds), i)
    n = getn(ds)
    if n >= 0
        if length(ds) > 0
            n = numobs(newdata)
        else
            # if newdata is empty, then we can't get the number of observations from it
            n = T == Bool ? sum(i) : length(i)
        end
    end
    if !(newdata isa Dict{Symbol, Any})
        newdata = Dict{Symbol, Any}(newdata)
    end
    return DataStore(n, newdata)
end

function cat_features(ds1::DataStore, ds2::DataStore)
    n1, n2 = getn(ds1), getn(ds2)
    n1 = n1 >= 0 ? n1 : 1
    n2 = n2 >= 0 ? n2 : 1
    return DataStore(n1 + n2, cat_features(getdata(ds1), getdata(ds2)))
end

function cat_features(dss::AbstractVector{DataStore}; kws...)
    ns = getn.(dss)
    ns = map(n -> n >= 0 ? n : 1, ns)
    return DataStore(sum(ns), cat_features(getdata.(dss); kws...))
end

# DataStore is always already normalized
normalize_graphdata(ds::DataStore; kws...) = ds

_gather(x::DataStore, i) = map(x -> _gather(x, i), x)

function _scatter(aggr, src::DataStore, idx, n)
    newdata = _scatter(aggr, getdata(src), idx, n)
    if !(newdata isa Dict{Symbol, Any})
        newdata = Dict{Symbol, Any}(newdata)
    end
    return DataStore(n, newdata)
end

function Base.hash(ds::D, h::UInt) where {D <: DataStore}
    fs = (getfield(ds, k) for k in fieldnames(D))
    return foldl((h, f) -> hash(f, h), fs, init = hash(D, h))
end
