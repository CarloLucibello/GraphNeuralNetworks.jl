#missig docs!
struct TemporalDataStore 
    _n::Int # either -1 or numobs(data)
    _t::Int # number of snapshots
    _data::Dict{Symbol, Any}

    function TemporalDataStore(n::Int, t::Int, data::Dict{Symbol, Any})
        if n >= 0 && t != 1
            for (k, v) in data
                @assert size(v)[end-1]==n "TemporalDataStore: data[$k] has $(size(v)[end-1]) observations, but n = $n"  #should be DimensionMismatch?
                @assert size(v)[end]==t "TemporalDataStore: data[$k] has $(size(v)[end]) snapshots, but t = $t"
            end
        end
        if t == 1
            return DataStore(n,data)
        end 
        return new(n, t, data)
    end
end

@functor TemporalDataStore

TemporalDataStore(data) = TemporalDataStore(-1, 0, data)
TemporalDataStore(n::Int,t::Int, data::NamedTuple) = TemporalDataStore(n, t, Dict{Symbol, Any}(pairs(data)))
TemporalDataStore(n::Int, t::Int, data) = TemporalDataStore(n, t, Dict{Symbol, Any}(data))

TemporalDataStore(; kws...) = TemporalDataStore(-1, 0; kws...)
TemporalDataStore(n::Int, t::Int; kws...) = TemporalDataStore(n, t, Dict{Symbol, Any}(kws...))


getdata(tds::TemporalDataStore) = getfield(tds, :_data)
getn(tds::TemporalDataStore) = getfield(tds, :_n)
gett(tds::TemporalDataStore) = getfield(tds, :_t)

function Base.getproperty(tds::TemporalDataStore, s::Symbol)
    if s === :_n
        return getn(tds)
    elseif s === :_data
        return getdata(tds)
    elseif s === :_t
        return gett(tds)
    else
        return getdata(tds)[s]
    end
end

function Base.setproperty!(tds::TemporalDataStore, s::Symbol, x)
    @assert s != :_n "cannot set _n directly"
    @assert s != :_data "cannot set _data directly"
    @assert s != :_t "cannot set _t directly"
    if getn(tds) >= 0 && gett(tds) > 1
        size(x)[end-1] == getn(tds) || throw(DimensionMismatch("expected $(getn(tds)) object features but got $(size(x)[end-1])."))
        size(x)[end] == gett(tds) || throw(DimensionMismatch("expected $(gett(tds)) snapshots but got $(size(x)[end])."))
    end
    return getdata(tds)[s] = x
end

Base.getindex(tds::TemporalDataStore, s::Symbol) = getproperty(tds, s)
Base.setindex!(tds::TemporalDataStore, s::Symbol, x) = setproperty!(tds, s, x)

Base.length(tds::TemporalDataStore) = length(getdata(tds))

function Base.show(io::IO, tds::TemporalDataStore)
    len = length(tds)
    n = getn(tds)
    t = gett(tds)
    if n < 0 || t == 0
        print(io, "TemporalDataStore()")
    else
        print(io, "TemporalDataStore($(getn(tds)), $(gett(tds)))")
    end
    if len > 0
        print(io, " with $(length(getdata(tds))) element")
        len > 1 && print(io, "s")
        print(io, ":")
        for (k, v) in getdata(tds)
            print(io, "\n  $(k) = $(summary(v))")
        end
    end
end

Base.iterate(tds::TemporalDataStore) = iterate(getdata(tds))
Base.iterate(tds::TemporalDataStore, state) = iterate(getdata(tds), state)
Base.keys(tds::TemporalDataStore) = keys(getdata(tds))
Base.values(tds::TemporalDataStore) = values(getdata(tds))
Base.haskey(tds::TemporalDataStore, k) = haskey(getdata(tds), k)
Base.get(tds::TemporalDataStore, k, default) = get(getdata(tds), k, default)
Base.pairs(tds::TemporalDataStore) = pairs(getdata(tds))
Base.:(==)(tds1::TemporalDataStore, tds2::TemporalDataStore) = getdata(tds1) == getdata(tds2)
Base.isempty(tds::TemporalDataStore) = isempty(getdata(tds))
Base.delete!(tds::TemporalDataStore, k) = delete!(getdata(tds), k)

function Base.map(f, tds::TemporalDataStore)
    d = getdata(tds)
    newd = Dict{Symbol, Any}(k => f(v) for (k, v) in d)
    return TemporalDataStore(getn(tds), gett(tds), newd)
end

function gettobs(A::AbstractArray{<:Any, N}, idx) where N
    I = ntuple(_ -> :, N-2)
    return A[I..., idx, :]
end

gettobs(data::Dict, i) = Dict(k => gettobs(v, i) for (k, v) in pairs(data))

function MLUtils.getobs(tds::TemporalDataStore, i)
    newdata = gettobs(getdata(tds), i)
    return TemporalDataStore(-1, gett(tds),newdata)
end

function getsnaps(A::AbstractArray{<:Any, N}, idx) where N
    I = ntuple(_ -> :, N-1)
    return A[I..., idx]
end

getsnaps(data::Dict, i) = Dict(k => getsnaps(v, i) for (k, v) in pairs(data))

function getsnaps(tds::TemporalDataStore, i)
    newdata = getsnaps(getdata(tds), i)
    return TemporalDataStore(getn(tds), length(i), newdata)
end

function cat_features(tds1::TemporalDataStore, tds2::TemporalDataStore)
    n1, n2 = getn(tds1), getn(tds2)
    n1 = n1 >= 0 ? n1 : 1
    n2 = n2 >= 0 ? n2 : 1
    return TemporalDataStore(n1 + n2, gett(tds1), cat_tempfeatures(getdata(tds1), getdata(tds2)))
end

function cat_features(tdss::AbstractVector{TemporalDataStore}; kws...)
    ns = getn.(tdss)
    ns = map(n -> n >= 0 ? n : 1, ns)
    return TemporalDataStore(sum(ns),gett(tdss[1]), cat_features(getdata.(tdss); kws...))
end
