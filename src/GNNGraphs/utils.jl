function check_num_nodes(g::GNNGraph, x::AbstractArray)
    @assert g.num_nodes == size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension size instead of num_nodes=$(g.num_nodes)"
    return true
end
function check_num_nodes(g::GNNGraph, x::Union{Tuple,NamedTuple})
    map(x -> check_num_nodes(g, x), x)
    return true
end

check_num_nodes(::GNNGraph, ::Nothing) = true

function check_num_edges(g::GNNGraph, e::AbstractArray)
    @assert g.num_edges == size(e, ndims(e)) "Got $(size(e, ndims(e))) as last dimension size instead of num_edges=$(g.num_edges)"
    return true    
end
function check_num_edges(g::GNNGraph, x::Union{Tuple,NamedTuple})
    map(x -> check_num_edges(g, x), x)
    return true
end

check_num_edges(::GNNGraph, ::Nothing) = true


sort_edge_index(eindex::Tuple) = sort_edge_index(eindex...)

function sort_edge_index(u, v)
    uv = collect(zip(u, v))
    p = sortperm(uv) # isless lexicographically defined for tuples
    return u[p], v[p]
end

function sort_edge_index(u::AnyCuArray, v::AnyCuArray)
    #TODO proper cuda friendly implementation
    sort_edge_index(u |> Flux.cpu, v |> Flux.cpu) |> Flux.gpu
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
        unsqz_last(v::AbstractArray) = size(v)[end] != 1 ? reshape(v, size(v)..., 1) : v
        unsqz_last(v) = v
    
        data = map(unsqz_last, data)
    end
    
    ## Turn vectors in 1 x n matrices. 
    # unsqz_first(v::AbstractVector) = reshape(v, 1, length(v))
    # unsqz_first(v) = v
    # data = map(unsqz_first, data)
    
    if duplicate_if_needed 
        function duplicate(v)
            if v isa AbstractArray && size(v)[end] == n÷2
                v = cat(v, v, dims=ndims(v))
            end
            v
        end
        data = map(duplicate, data)
    end
    
    for x in data
        if x isa AbstractArray
            @assert size(x)[end] == n "Wrong size in last dimension for feature array, expected $n but got $(size(x)[end])."
        end
    end    
    return data
end

# For heterogeneous graphs
normalize_heterographdata(data; kws...) =
    normalize_heterographdata(Dict(data); kws...)

function normalize_heterographdata(data::Dict; default_name::Symbol, n::Dict, kws...)
    isempty(data) && return data
    Dict(k => normalize_graphdata(v; default_name=default_name, n=n[k], kws...)
            for (k,v) in data)
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

# each edge is represented by a number in
# 1:n1*n2
function edge_decoding(idx, n1, n2)
    @assert all(1 .<= idx .<= n1*n2)
    s =  (idx .- 1) .÷ n2 .+ 1
    t =  (idx .- 1) .% n2 .+ 1
    return s, t
end

binarize(x) = map(>(0), x)

@non_differentiable binarize(x...)
@non_differentiable edge_encoding(x...)
@non_differentiable edge_decoding(x...)


### PRINTING #####


function shortsummary(io::IO, x)
    s = shortsummary(x)
    s === nothing && return
    print(io, s)
end

shortsummary(x) = summary(x)
shortsummary(x::Number) = "$x"

function shortsummary(x::NamedTuple) 
    if length(x) == 0
        return nothing
    elseif length(x) === 1
        return "$(keys(x)[1]) = $(shortsummary(x[1]))"
    else
        "(" * join(("$k = $(shortsummary(x[k]))" for k in keys(x)), ", ") * ")"
    end
end

