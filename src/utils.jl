function check_num_nodes(g::GNNGraph, x::AbstractArray)
    @assert g.num_nodes == size(x, ndims(x))    
end

sort_edge_index(eindex::Tuple) = sort_edge_index(eindex...)

function sort_edge_index(u, v)
    uv = collect(zip(u, v))
    p = sortperm(uv) # isless lexicographically defined for tuples
    return u[p], v[p]
end

cat_features(x1::Nothing, x2::Nothing) = nothing 
cat_features(x1::AbstractArray, x2::AbstractArray) = cat(x1, x2, dims=ndims(x1))
cat_features(x1::Union{Number, AbstractVector}, x2::Union{Number, AbstractVector}) = 
    cat(x1, x2, dims=1)


function cat_features(x1::NamedTuple, x2::NamedTuple)
    sort(collect(keys(x1))) == sort(collect(keys(x2))) ||
        @error "cannot concatenate feature data with different keys"
    
    NamedTuple(k => cat_features(getfield(x1,k), getfield(x2,k)) for k in keys(x1))
end

# Turns generic type into named tuple
normalize_graphdata(data::Nothing; kws...) = NamedTuple()

normalize_graphdata(data; default_name::Symbol, kws...) = 
    normalize_graphdata(NamedTuple{(default_name,)}((data,)); default_name, kws...) 

function normalize_graphdata(data::NamedTuple; default_name=:z, n, duplicate_if_needed=false)
    sz = map(x -> x isa AbstractArray ? size(x)[end] : 0, data)
    
    if duplicate_if_needed # used to copy edge features on reverse edges    
        @assert all(s -> s == 0 ||  s == n || s == n÷2, sz)

        function replace(k, v)
            if v isa AbstractArray && size(v)[end] == n÷2
                v = cat(v, v, dims=ndims(v))
            end
            k => v
        end

        data = NamedTuple(replace(k,v) for (k,v) in pairs(data))
    else
        @assert all(s -> s == 0 ||  s == n, sz)
    end
    return data
end

