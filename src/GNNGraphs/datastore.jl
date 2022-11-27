""""
    DataStore(n = -1)
    DataStore(data, n = -1)


"""
struct DataStore
    _data::Dict{Symbol, Any}
    _n::Int # either -1 or numobs(data)

    function DataStore(data::Dict{Symbol,Any},  n::Int = -1)
        if n >= 0
            for (k, v) in data
                @assert numobs(v) == n "DataStore: data[$k] has $(numobs(v)) observations, but n = $n"
            end
        end
        return new(data, n)
    end
end

@functor DataStore

# function Functors.functor(::Type{DataStore}, ds::DataStore) 
#     children = (data = getdata(ds), n = getn(ds))
#     reconstruct((data, n)) = DataStore(data, n)
#     return children, reconstruct
# end

DataStore(data::NamedTuple, n::Int = -1) = DataStore(Dict{Symbol,Any}(pairs(data)), n)
DataStore(data, n::Int = -1) = DataStore(Dict{Symbol,Any}(data), n)


function DataStore(n::Int = -1)
    return DataStore(Dict{Symbol, Any}(), n)
end

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
    if getn(ds) > 0
        @assert numobs(x) == getn(ds) "expected (numobs(x) == getn(ds)) but got $(numobs(x)) != $(getn(ds))"
    end
   return getdata(ds)[s] = x
end

Base.getindex(ds::DataStore, s::Symbol) = getproperty(ds, s)
Base.setindex!(ds::DataStore, s::Symbol, x) = setproperty!(ds, s, x)

function Base.show(io::IO, ds::DataStore)
    print(io, "DataStore with $(length(getdata(ds))) element")
    length(ds) != 1 && print(io, "s")
    println(io, ":")
    for (k, v) in getdata(ds)
        println(io, "  $(k) = $(summary(v))")
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
    return DataStore(newd, getn(ds))
end

MLUtils.numobs(ds::DataStore) = numobs(getdata(ds))

function MLUtils.getobs(ds::DataStore, i::Int)
    newdata = getobs(getdata(ds), i)
    return DataStore(newdata, -1)
end

function MLUtils.getobs(ds::DataStore, i::AbstractVector{T}) where T <: Union{Integer,Bool}
    newdata = getobs(getdata(ds), i)
    n = getn(ds)
    if n > -1
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
    return DataStore(newdata, n)
end

function cat_features(ds1::DataStore, ds2::DataStore)
    n1, n2 = getn(ds1), getn(ds2)
    n1 = n1 > 0 ? n1 : 1 
    n2 = n2 > 0 ? n2 : 1
    return DataStore(cat_features(getdata(ds1), getdata(ds2)), n1 + n2)
end

function cat_features(dss::AbstractVector{DataStore}; kws...)
    ns = getn.(dss)
    ns = map(n -> n > 0 ? n : 1, ns)
    return DataStore(cat_features(getdata.(dss); kws...), sum(ns))
end

# DataStore is always already normalized
normalize_graphdata(ds::DataStore; kws...) = ds

_gather(x::DataStore, i) = map(x -> _gather(x, i), x)

function _scatter(aggr, src::DataStore, idx, n)
    newdata = _scatter(aggr, getdata(src), idx, n)
    if !(newdata isa Dict{Symbol, Any})
        newdata = Dict{Symbol, Any}(newdata)
    end
    return DataStore(newdata, n)
end

function Base.hash(ds::D, h::UInt) where {D <: DataStore}
    fs = (getfield(ds, k) for k in fieldnames(D))
    return foldl((h, f) -> hash(f, h),  fs, init=hash(D, h))
end
