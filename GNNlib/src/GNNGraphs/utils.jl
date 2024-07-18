function check_num_nodes(g::GNNGraph, x::AbstractArray)
    @assert g.num_nodes==size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension size instead of num_nodes=$(g.num_nodes)"
    return true
end
function check_num_nodes(g::GNNGraph, x::Union{Tuple, NamedTuple})
    map(x -> check_num_nodes(g, x), x)
    return true
end

check_num_nodes(::GNNGraph, ::Nothing) = true

function check_num_nodes(g::GNNGraph, x::Tuple)
    @assert length(x) == 2
    check_num_nodes(g, x[1])
    check_num_nodes(g, x[2])
    return true
end

# x = (Xsrc, Xdst) = (Xj, Xi)
function check_num_nodes(g::GNNHeteroGraph, x::Tuple)
    @assert length(x) == 2
    @assert length(g.etypes) == 1
    nt1, _, nt2 = only(g.etypes)
    if x[1] isa AbstractArray
        @assert size(x[1], ndims(x[1])) == g.num_nodes[nt1]
    end
    if x[2] isa AbstractArray
        @assert size(x[2], ndims(x[2])) == g.num_nodes[nt2] 
    end
    return true
end

function check_num_edges(g::GNNGraph, e::AbstractArray)
    @assert g.num_edges==size(e, ndims(e)) "Got $(size(e, ndims(e))) as last dimension size instead of num_edges=$(g.num_edges)"
    return true
end
function check_num_edges(g::AbstractGNNGraph, x::Union{Tuple, NamedTuple})
    map(x -> check_num_edges(g, x), x)
    return true
end

check_num_edges(::AbstractGNNGraph, ::Nothing) = true

function check_num_edges(g::GNNHeteroGraph, e::AbstractArray)
    num_edgs = only(g.num_edges)[2]
    @assert only(num_edgs)==size(e, ndims(e)) "Got $(size(e, ndims(e))) as last dimension size instead of num_edges=$(num_edgs)"
    return true
end

sort_edge_index(eindex::Tuple) = sort_edge_index(eindex...)

"""
    sort_edge_index(ei::Tuple) -> u', v'
    sort_edge_index(u, v) -> u', v'

Return a sorted version of the tuple of vectors `ei = (u, v)`,
applying a common permutation to `u` and `v`.
The sorting is lexycographic, that is the pairs `(ui, vi)` 
are sorted first according to the `ui` and then according to `vi`. 
"""
function sort_edge_index(u, v)
    uv = collect(zip(u, v))
    p = sortperm(uv) # isless lexicographically defined for tuples
    return u[p], v[p]
end



cat_features(x1::Nothing, x2::Nothing) = nothing
cat_features(x1::AbstractArray, x2::AbstractArray) = cat(x1, x2, dims = ndims(x1))
function cat_features(x1::Union{Number, AbstractVector}, x2::Union{Number, AbstractVector})
    cat(x1, x2, dims = 1)
end

# workaround for issue #98 #104
# See https://github.com/JuliaStrings/InlineStrings.jl/issues/21
# Remove when minimum supported version is julia v1.8
cat_features(x1::NamedTuple{(), Tuple{}}, x2::NamedTuple{(), Tuple{}}) = (;)
cat_features(xs::AbstractVector{NamedTuple{(), Tuple{}}}) = (;)

function cat_features(x1::NamedTuple, x2::NamedTuple)
    sort(collect(keys(x1))) == sort(collect(keys(x2))) ||
        @error "cannot concatenate feature data with different keys"

    return NamedTuple(k => cat_features(x1[k], x2[k]) for k in keys(x1))
end

function cat_features(x1::Dict{Symbol, T}, x2::Dict{Symbol, T}) where {T}
    sort(collect(keys(x1))) == sort(collect(keys(x2))) ||
        @error "cannot concatenate feature data with different keys"

    return Dict{Symbol, T}([k => cat_features(x1[k], x2[k]) for k in keys(x1)]...)
end

function cat_features(x::Dict)
    return Dict([k => cat_features(v) for (k, v) in pairs(x)]...)
end


function cat_features(xs::AbstractVector{<:AbstractArray{T, N}}) where {T <: Number, N}
    cat(xs...; dims = N)
end

cat_features(xs::AbstractVector{Nothing}) = nothing
cat_features(xs::AbstractVector{<:Number}) = xs

function cat_features(xs::AbstractVector{<:NamedTuple})
    symbols = [sort(collect(keys(x))) for x in xs]
    all(y -> y == symbols[1], symbols) ||
        @error "cannot concatenate feature data with different keys"
    length(xs) == 1 && return xs[1]

    # concatenate
    syms = symbols[1]
    NamedTuple(k => cat_features([x[k] for x in xs]) for k in syms)
end

# function cat_features(xs::AbstractVector{Dict{Symbol, T}}) where {T}
#     symbols = [sort(collect(keys(x))) for x in xs]
#     all(y -> y == symbols[1], symbols) ||
#         @error "cannot concatenate feature data with different keys"
#     length(xs) == 1 && return xs[1]

#     # concatenate 
#     syms = symbols[1]
#     return Dict{Symbol, T}([k => cat_features([x[k] for x in xs]) for k in syms]...)
# end

function cat_features(xs::AbstractVector{<:Dict})
    _allkeys = [sort(collect(keys(x))) for x in xs]
    _keys = union(_allkeys...)
    length(xs) == 1 && return xs[1]

    # concatenate 
    return Dict([k => cat_features([x[k] for x in xs if haskey(x, k)]) for k in _keys]...)
end


# Used to concatenate edge weights
cat_features(w1::Nothing, w2::Nothing, n1::Int, n2::Int) = nothing
cat_features(w1::AbstractVector, w2::Nothing, n1::Int, n2::Int) = cat_features(w1, ones_like(w1, n2))
cat_features(w1::Nothing, w2::AbstractVector, n1::Int, n2::Int) = cat_features(ones_like(w2, n1), w2)
cat_features(w1::AbstractVector, w2::AbstractVector, n1::Int, n2::Int) = cat_features(w1, w2)


# Turns generic type into named tuple
normalize_graphdata(data::Nothing; n, kws...) = DataStore(n)

function normalize_graphdata(data; default_name::Symbol, kws...)
    normalize_graphdata(NamedTuple{(default_name,)}((data,)); default_name, kws...)
end

function normalize_graphdata(data::NamedTuple; default_name, n, duplicate_if_needed = false)
    # This had to workaround two Zygote bugs with NamedTuples
    # https://github.com/FluxML/Zygote.jl/issues/1071
    # https://github.com/FluxML/Zygote.jl/issues/1072

    if n > 1
        @assert all(x -> x isa AbstractArray, data) "Non-array features provided."
    end

    if n <= 1
        # If last array dimension is not 1, add a new dimension.
        # This is mostly useful to reshape global feature vectors
        # of size D to Dx1 matrices.
        unsqz_last(v::AbstractArray) = size(v)[end] != 1 ? reshape(v, size(v)..., 1) : v
        unsqz_last(v) = v

        data = map(unsqz_last, data)
    end

    if n > 0
        if duplicate_if_needed
            function duplicate(v)
                if v isa AbstractArray && size(v)[end] == n ÷ 2
                    v = cat(v, v, dims = ndims(v))
                end
                return v
            end
            data = map(duplicate, data)
        end

        for x in data
            if x isa AbstractArray
                @assert size(x)[end]==n "Wrong size in last dimension for feature array, expected $n but got $(size(x)[end])."
            end
        end
    end

    return DataStore(n, data)
end

# For heterogeneous graphs
function normalize_heterographdata(data::Nothing; default_name::Symbol, ns::Dict, kws...)
    Dict([k => normalize_graphdata(nothing; default_name = default_name, n, kws...)
         for (k, n) in ns]...)
end

normalize_heterographdata(data; kws...) = normalize_heterographdata(Dict(data); kws...)

function normalize_heterographdata(data::Dict; default_name::Symbol, ns::Dict, kws...)
    Dict([k => normalize_graphdata(get(data, k, nothing); default_name = default_name, n, kws...)
         for (k, n) in ns]...)
end

numnonzeros(a::AbstractSparseMatrix) = nnz(a)
numnonzeros(a::AbstractMatrix) = count(!=(0), a)

# each edge is represented by a number in
# 1:N^2
function edge_encoding(s, t, n; directed = true)
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
        idx = @. (s - 1) * (2 * (n + 1) - s) ÷ 2 + (t - s + 1)
    end
    return idx, maxid
end

# each edge is represented by a number in
# 1:N^2
function edge_decoding(idx, n; directed = true)
    if directed
        # g = remove_self_loops(g)
        s = (idx .- 1) .÷ n .+ 1
        t = (idx .- 1) .% n .+ 1
    else
        # We replace j=n in
        # idx = (i - 1)*(2*(n+1)-i)÷2 + (j - i + 1)
        # and obtain
        # idx = (i - 1)*(2*(n+1)-i)÷2 + (n - i + 1)

        # OR We replace j=i  and obtain??
        # idx = (i - 1)*(2*(n+1)-i)÷2 + 1

        # inverting we have
        s = @. ceil(Int, -sqrt((n + 1 / 2)^2 - 2 * idx) + n + 1 / 2)
        t = @. idx - (s - 1) * (2 * (n + 1) - s) ÷ 2 - 1 + s
        # t =  (idx .- 1) .% n .+ 1
    end
    return s, t
end

# each edge is represented by a number in
# 1:n1*n2
function edge_decoding(idx, n1, n2)
    @assert all(1 .<= idx .<= n1 * n2)
    s = (idx .- 1) .÷ n2 .+ 1
    t = (idx .- 1) .% n2 .+ 1
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

function shortsummary(x::DataStore)
    length(x) == 0 && return nothing
    return "DataStore(" * join(("$k = [$(shortsummary(x[k]))]" for k in keys(x)), ", ") *
           ")"
end

# from (2,2,3) output of size function to a string "2×2×3"
function dims2string(d)
    isempty(d) ? "0-dimensional" :
    length(d) == 1 ? "$(d[1])-element" :
    join(map(string, d), '×')
end

@non_differentiable normalize_graphdata(::NamedTuple{(), Tuple{}})
@non_differentiable normalize_graphdata(::Nothing)

iscuarray(x::AbstractArray) = false 
@non_differentiable iscuarray(::Any)


@doc raw"""
    color_refinement(g::GNNGraph, [x0]) -> x, num_colors, niters

The color refinement algorithm for graph coloring. 
Given a graph `g` and an initial coloring `x0`, the algorithm 
iteratively refines the coloring until a fixed point is reached.

At each iteration the algorithm computes a hash of the coloring and the sorted list of colors
of the neighbors of each node. This hash is used to determine if the coloring has changed.

```math
x_i' = hashmap((x_i, sort([x_j for j \in N(i)]))).
````

This algorithm is related to the 1-Weisfeiler-Lehman algorithm for graph isomorphism testing.

# Arguments
- `g::GNNGraph`: The graph to color.
- `x0::AbstractVector{<:Integer}`: The initial coloring. If not provided, all nodes are colored with 1.

# Returns
- `x::AbstractVector{<:Integer}`: The final coloring.
- `num_colors::Int`: The number of colors used.
- `niters::Int`: The number of iterations until convergence.
"""
color_refinement(g::GNNGraph) = color_refinement(g, ones(Int, g.num_nodes))

function color_refinement(g::GNNGraph, x0::AbstractVector{<:Integer})
    @assert length(x0) == g.num_nodes
    s, t = edge_index(g)
    t, s = sort_edge_index(t, s) # sort by target
    degs = degree(g, dir=:in)
    x = x0 

    hashmap = Dict{UInt64, Int}()
    x′ = zeros(Int, length(x0))
    niters = 0    
    while true
        xneigs = chunk(x[s], size=degs)
        for (i, (xi, xineigs)) in enumerate(zip(x, xneigs))
            idx = hash((xi, sort(xineigs)))
            x′[i] = get!(hashmap, idx, length(hashmap) + 1)
        end
        niters += 1
        x == x′ && break
        x = x′
    end
    num_colors = length(union(x))
    return x, num_colors, niters
end