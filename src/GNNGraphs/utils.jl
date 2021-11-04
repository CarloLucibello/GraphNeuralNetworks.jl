function check_num_nodes(g::GNNGraph, x::AbstractArray)
    @assert g.num_nodes == size(x, ndims(x))    
end
function check_num_edges(g::GNNGraph, e::AbstractArray)
    @assert g.num_edges == size(e, ndims(e))    
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
    sort(collect(keys(x1))) == sort(collect(keys(x2))) || @error "cannot concatenate feature data with different keys"
    
    NamedTuple(k => cat_features(getfield(x1,k), getfield(x2,k)) for k in keys(x1))
end

# Turns generic type into named tuple
normalize_graphdata(data::Nothing; kws...) = NamedTuple()

normalize_graphdata(data; default_name::Symbol, kws...) = 
normalize_graphdata(NamedTuple{(default_name,)}((data,)); default_name, kws...) 

function normalize_graphdata(data::NamedTuple; default_name, n, duplicate_if_needed=false)
    # This had to workaround two Zygote bugs with NamedTuples
    # https://github.com/FluxML/Zygote.jl/issues/1071
    # https://github.com/FluxML/Zygote.jl/issues/1072
    
    if n == 1
        # If last array dimension is not 1, add a new dimension. 
        # This is mostly usefule to reshape globale feature vectors
        # of size D to Dx1 matrices.
        function unsqz(v)
            if v isa AbstractArray && size(v)[end] != 1
                v = reshape(v, size(v)..., 1)
            end
            v
        end
    
        data = NamedTuple{keys(data)}(unsqz.(values(data)))
    end

    sz = map(x -> x isa AbstractArray ? size(x)[end] : 0, data)
    if duplicate_if_needed 
        # Used to copy edge features on reverse edges    
        @assert all(s -> s == 0 ||  s == n || s == n÷2, sz)
    
        function duplicate(v)
            if v isa AbstractArray && size(v)[end] == n÷2
                v = cat(v, v, dims=ndims(v))
            end
            v
        end
        data = NamedTuple{keys(data)}(duplicate.(values(data)))
    else
        @assert all(s -> s == 0 ||  s == n, sz)
    end
    return data
end

ones_like(x::AbstractArray, T=eltype(x), sz=size(x)) = fill!(similar(x, T, sz), 1)
ones_like(x::SparseMatrixCSC, T=eltype(x), sz=size(x)) = ones(T, sz)
ones_like(x::CUMAT_T, T=eltype(x), sz=size(x)) = CUDA.ones(T, sz)


# each edge is represented by a number in
# 1:N^2
function edge_encoding(s, t, n; directed=true)
    if directed
        # directed edges and self-loops allowed
        idx = (s .- 1) .* n .+ t
        maxid = n^2
    else 
        # Undirected edges and self-loops allowed
        # In this encoding, each edge has 2 possible encodings (also the self-loops).
        # We return the canonical one given by the upper triangular adj matrix
        maxid = n * (n + 1) ÷ 2
        mask = s .> t
        # s1, t1 = s[mask], t[mask]
        # t2, s2 = s[.!mask], t[.!mask]
        snew = copy(s)
        tnew = copy(t)
        snew[mask] .= t[mask]
        tnew[mask] .= s[mask]
        s, t = snew, tnew
        
        # idx = ∑_{i',i'<i} ∑_{j',j'>=i'}^n 1 + ∑_{j',i<=j'<=j} 1 
        #     = ∑_{i',i'<i} ∑_{j',j'>=i'}^n 1 + j - i + 1
        #     = ∑_{i',i'<i} (n - i' + 1) + (j - i + 1)
        #     = (i - 1)*(2*(n+1)-i)÷2 + (j - i + 1)
        idx = @. (s-1)*(2*(n+1)-s)÷2 + (t-s+1)
    end
    return idx, maxid
end

# each edge is represented by a number in
# 1:N^2
function edge_decoding(idx, n; directed=true)
    # g = remove_self_loops(g)
    s =  (idx .- 1) .÷ n .+ 1
    t =  (idx .- 1) .% n .+ 1
    return s, t
end

@non_differentiable edge_encoding(x...)
@non_differentiable edge_decoding(x...)

