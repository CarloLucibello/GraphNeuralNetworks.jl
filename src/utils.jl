function check_num_nodes(g::GNNGraph, x::AbstractArray)
    @assert nv(g) == size(x, ndims(x))    
end

sort_edge_index(eindex::Tuple) = sort_edge_index(eindex...)

function sort_edge_index(u, v)
    uv = collect(zip(u, v))
    p = sortperm(uv) # isless lexicographically defined for tuples
    return u[p], v[p]
end
