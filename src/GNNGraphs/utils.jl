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

# workaround for issue #98 #104
cat_features(x1::NamedTuple{(), Tuple{}}, x2::NamedTuple{(), Tuple{}}) = (;)
cat_features(xs::AbstractVector{NamedTuple{(), Tuple{}}}) = (;)

function cat_features(x1::NamedTuple, x2::NamedTuple)
    sort(collect(keys(x1))) == sort(collect(keys(x2))) || @error "cannot concatenate feature data with different keys"
    
    NamedTuple(k => cat_features(getfield(x1,k), getfield(x2,k)) for k in keys(x1))
end

function cat_features(xs::AbstractVector{<:AbstractArray{T,N}}) where {T<:Number, N}
   cat(xs...; dims=N) 
end

cat_features(xs::AbstractVector{Nothing}) = nothing
cat_features(xs::AbstractVector{<:Number}) = xs

function cat_features(xs::AbstractVector{<:NamedTuple})
    symbols = [sort(collect(keys(x))) for x in xs]
    all(y -> y==symbols[1], symbols) || @error "cannot concatenate feature data with different keys"
    length(xs) == 1 && return xs[1] 

    # concatenate 
    syms = symbols[1]
    NamedTuple(
        k => cat_features([x[k] for x in xs]) for (ii,k) in enumerate(syms)
    )
end

# Turns generic type into named tuple
normalize_graphdata(data::Nothing; kws...) = NamedTuple()

normalize_graphdata(data; default_name::Symbol, kws...) = 
normalize_graphdata(NamedTuple{(default_name,)}((data,)); default_name, kws...) 

function normalize_graphdata(data::NamedTuple; default_name, n, duplicate_if_needed=false)
    # This had to workaround two Zygote bugs with NamedTuples
    # https://github.com/FluxML/Zygote.jl/issues/1071
    # https://github.com/FluxML/Zygote.jl/issues/1072

    if n != 1
        @assert all(x -> x isa AbstractArray, data) "Non-array features provided."
    end
    
    if n == 1
        # If last array dimension is not 1, add a new dimension. 
        # This is mostly useful to reshape global feature vectors
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
        @assert all(s -> s == 0 ||  s == n || s == n÷2, sz)  "Wrong size in last dimension for feature array."
    
        function duplicate(v)
            if size(v)[end] == n÷2
                v = cat(v, v, dims=ndims(v))
            end
            v
        end
        data = NamedTuple{keys(data)}(duplicate.(values(data)))
    else
        @assert all(x -> x == 0 || x == n, sz) "Wrong size in last dimension for feature array."
    end
    return data
end

ones_like(x::AbstractArray, T::Type, sz=size(x)) = fill!(similar(x, T, sz), 1)
ones_like(x::SparseMatrixCSC, T::Type, sz=size(x)) = ones(T, sz)
ones_like(x::CUMAT_T, T::Type, sz=size(x)) = CUDA.ones(T, sz)
ones_like(x, sz=size(x)) = ones_like(x, eltype(x), sz)

numnonzeros(a::AbstractSparseMatrix) = nnz(a)
numnonzeros(a::AbstractMatrix) = count(!=(0), a)

# each edge is represented by a number in
# 1:N^2
function edge_encoding(s, t, n; directed=true)
    if directed
        # directed edges and self-loops allowed
        idx = (s .- 1) .* n .+ t
        maxid = n^2
    else 
        # Undirected edges and self-loops allowed
        maxid = n * (n + 1) ÷ 2
        
        mask = s .> t
        snew = copy(s)
        tnew = copy(t)
        snew[mask] .= t[mask]
        tnew[mask] .= s[mask]
        s, t = snew, tnew

        # idx = ∑_{i',i'<i} ∑_{j',j'>=i'}^n 1 + ∑_{j',i<=j'<=j} 1 
        #     = ∑_{i',i'<i} ∑_{j',j'>=i'}^n 1 + (j - i + 1)
        #     = ∑_{i',i'<i} (n - i' + 1) + (j - i + 1)
        #     = (i - 1)*(2*(n+1)-i)÷2 + (j - i + 1)
        idx = @. (s-1)*(2*(n+1)-s)÷2 + (t-s+1)
    end
    return idx, maxid
end

# each edge is represented by a number in
# 1:N^2
function edge_decoding(idx, n; directed=true)
    if directed
        # g = remove_self_loops(g)
        s =  (idx .- 1) .÷ n .+ 1
        t =  (idx .- 1) .% n .+ 1
    else
        # We replace j=n in 
        # idx = (i - 1)*(2*(n+1)-i)÷2 + (j - i + 1) 
        # and obtain
        # idx = (i - 1)*(2*(n+1)-i)÷2 + (n - i + 1) 
        
        # OR We replace j=i  and obtain??
        # idx = (i - 1)*(2*(n+1)-i)÷2 + 1 
        
        # inverting we have
        s = @. ceil(Int, -sqrt((n + 1/2)^2 - 2*idx) + n + 1/2)
        t = @. idx - (s-1)*(2*(n+1)-s)÷2 - 1 + s
        # t =  (idx .- 1) .% n .+ 1
    end
    return s, t
end

binarize(x) = map(>(0), x)

@non_differentiable binarize(x...)
@non_differentiable edge_encoding(x...)
@non_differentiable edge_decoding(x...)



####################################
# FROM MLBASE.jl
# https://github.com/JuliaML/MLBase.jl/pull/1/files
# remove when package is registered
##############################################

numobs(A::AbstractArray{<:Any, N}) where {N} = size(A, N)

# 0-dim arrays
numobs(A::AbstractArray{<:Any, 0}) = 1

function getobs(A::AbstractArray{<:Any, N}, idx) where N
    I = ntuple(_ -> :, N-1)
    return A[I..., idx]
end

getobs(A::AbstractArray{<:Any, 0}, idx) = A[idx]

function getobs!(buffer::AbstractArray, A::AbstractArray{<:Any, N}, idx) where N
    I = ntuple(_ -> :, N-1)
    buffer .= A[I..., idx]
    return buffer
end

# --------------------------------------------------------------------
# Tuples and NamedTuples

_check_numobs_error() =
    throw(DimensionMismatch("All data containers must have the same number of observations."))

function _check_numobs(tup::Union{Tuple, NamedTuple})
    length(tup) == 0 && return
    n1 = numobs(tup[1])
    for i=2:length(tup)
        numobs(tup[i]) != n1 && _check_numobs_error()
    end
end

function numobs(tup::Union{Tuple, NamedTuple})::Int
    _check_numobs(tup)
    return length(tup) == 0 ? 0 : numobs(tup[1])
end

function getobs(tup::Union{Tuple, NamedTuple}, indices)
    _check_numobs(tup)
    return map(x -> getobs(x, indices), tup)
end

function getobs!(buffers::Union{Tuple, NamedTuple},
                  tup::Union{Tuple, NamedTuple},
                  indices)
    _check_numobs(tup)

    return map(buffers, tup) do buffer, x
                getobs!(buffer, x, indices)
            end
end
#######################################################