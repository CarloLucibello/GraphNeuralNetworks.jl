
"""
    edge_index(g::GNNGraph)

Return a tuple containing two vectors, respectively storing 
the source and target nodes for each edges in `g`.

```julia
s, t = edge_index(g)
```
"""
edge_index(g::GNNGraph{<:COO_T}) = g.graph[1:2]

edge_index(g::GNNGraph{<:ADJMAT_T}) = to_coo(g.graph, num_nodes=g.num_nodes)[1][1:2]

get_edge_weight(g::GNNGraph{<:COO_T}) = g.graph[3]

get_edge_weight(g::GNNGraph{<:ADJMAT_T}) = to_coo(g.graph, num_nodes=g.num_nodes)[1][3]

Graphs.edges(g::GNNGraph) = zip(edge_index(g)...)

Graphs.edgetype(g::GNNGraph) = Tuple{Int, Int}

# """
#     eltype(g::GNNGraph)
#
# Type of nodes in `g`,
# an integer type like `Int`, `Int32`, `Uint16`, ....
# """
function Base.eltype(g::GNNGraph{<:COO_T})
    s, t = edge_index(g)
    w = get_edge_weight(g)
    return w !== nothing ? eltype(w) : eltype(s)
end

Base.eltype(g::GNNGraph{<:ADJMAT_T}) = eltype(g.graph)

function Graphs.has_edge(g::GNNGraph{<:COO_T}, i::Integer, j::Integer)
    s, t = edge_index(g)
    return any((s .== i) .& (t .== j))
end

Graphs.has_edge(g::GNNGraph{<:ADJMAT_T}, i::Integer, j::Integer) = g.graph[i,j] != 0

graph_type_symbol(g::GNNGraph{<:COO_T}) = :coo 
graph_type_symbol(g::GNNGraph{<:SPARSE_T}) = :sparse
graph_type_symbol(g::GNNGraph{<:ADJMAT_T}) = :dense

Graphs.nv(g::GNNGraph) = g.num_nodes
Graphs.ne(g::GNNGraph) = g.num_edges
Graphs.has_vertex(g::GNNGraph, i::Int) = 1 <= i <= g.num_nodes
Graphs.vertices(g::GNNGraph) = 1:g.num_nodes

function Graphs.neighbors(g::GNNGraph, i; dir=:out)
    @assert dir ∈ (:in, :out)
    if dir == :out
        outneighbors(g, i)
    else
        inneighbors(g, i)
    end
end

function Graphs.outneighbors(g::GNNGraph{<:COO_T}, i::Integer)
    s, t = edge_index(g)
    return t[s .== i]
end

function Graphs.outneighbors(g::GNNGraph{<:ADJMAT_T}, i::Integer)
    A = g.graph
    return findall(!=(0), A[i,:])
end

function Graphs.inneighbors(g::GNNGraph{<:COO_T}, i::Integer)
    s, t = edge_index(g)
    return s[t .== i]
end

function Graphs.inneighbors(g::GNNGraph{<:ADJMAT_T}, i::Integer)
    A = g.graph
    return findall(!=(0), A[:,i])
end

Graphs.is_directed(::GNNGraph) = true
Graphs.is_directed(::Type{<:GNNGraph}) = true

"""
    adjacency_list(g; dir=:out)
    adjacency_list(g, nodes; dir=:out)

Return the adjacency list representation (a vector of vectors)
of the graph `g`.

Calling `a` the adjacency list, if `dir=:out` than
`a[i]` will contain the neighbors of node `i` through
outgoing edges. If `dir=:in`, it will contain neighbors from
incoming edges instead.

If `nodes` is given, return the neighborhood of the nodes in `nodes` only.
"""
function adjacency_list(g::GNNGraph, nodes; dir=:out, with_eid=false)
    @assert dir ∈ [:out, :in]
    s, t = edge_index(g)
    if dir == :in
        s, t = t, s
    end
    T = eltype(s)
    idict = 0
    dmap = Dict(n => (idict += 1) for n in nodes)
    adjlist = [T[] for _=1:length(dmap)]
    eidlist = [T[] for _=1:length(dmap)]
    for (eid, (i, j)) in enumerate(zip(s, t))
        inew = get(dmap, i, 0)
        inew == 0 && continue
        push!(adjlist[inew], j)
        push!(eidlist[inew], eid)
    end
    if with_eid
        return adjlist, eidlist
    else
        return adjlist
    end
end

# function adjacency_list(g::GNNGraph, nodes; dir=:out)
#     @assert dir ∈ [:out, :in]
#     fneighs = dir == :out ? outneighbors : inneighbors
#     return [fneighs(g, i) for i in nodes]
# end

adjacency_list(g::GNNGraph; dir=:out) = adjacency_list(g, 1:g.num_nodes; dir)


"""
    adjacency_matrix(g::GNNGraph, T=eltype(g); dir=:out, weighted=true)

Return the adjacency matrix `A` for the graph `g`. 

If `dir=:out`, `A[i,j] > 0` denotes the presence of an edge from node `i` to node `j`.
If `dir=:in` instead, `A[i,j] > 0` denotes the presence of an edge from node `j` to node `i`.

User may specify the eltype `T` of the returned matrix. 

If `weighted=true`, the `A` will contain the edge weigths if any, otherwise the elements of `A` will be either 0 or 1.
"""
function Graphs.adjacency_matrix(g::GNNGraph{<:COO_T}, T::DataType=eltype(g); dir=:out, weighted=true)
    if g.graph[1] isa CuVector
        # TODO revisit after https://github.com/JuliaGPU/CUDA.jl/pull/1152
        A, n, m = to_dense(g.graph, T; num_nodes=g.num_nodes, weighted)
    else
        A, n, m = to_sparse(g.graph, T; num_nodes=g.num_nodes, weighted)
    end
    @assert size(A) == (n, n)
    return dir == :out ? A : A'
end

function Graphs.adjacency_matrix(g::GNNGraph{<:ADJMAT_T}, T::DataType=eltype(g); dir=:out, weighted=true)
    @assert dir ∈ [:in, :out]
    A = g.graph
    if !weighted
        A = binarize(A)
    end
    A = T != eltype(A) ? T.(A) : A
    return dir == :out ? A : A'
end

function _get_edge_weight(g, edge_weight)
    if edge_weight === true || edge_weight === nothing 
        ew = get_edge_weight(g)
    elseif edge_weight === false
        ew = nothing 
    elseif edge_weight isa AbstractVector
        ew = edge_weight 
    else
        error("Invalid edge_weight argument.")
    end
    return ew
end

"""
    degree(g::GNNGraph, T=nothing; dir=:out, edge_weight=true)

Return a vector containing the degrees of the nodes in `g`.

# Arguments
- `g`: A graph.
- `T`: Element type of the returned vector. If `nothing`, is
       chosen based on the graph type and will be an integer
       if `edge_weight=false`.
- `dir`: For `dir=:out` the degree of a node is counted based on the outgoing edges.
         For `dir=:in`, the ingoing edges are used. If `dir=:both` we have the sum of the two.
- `edge_weight`: If `true` and the graph contains weighted edges, the degree will 
                be weighted. Set to `false` instead to just count the number of
                outgoing/ingoing edges.
                In alternative, you can also pass a vector of weights to be used
                instead of the graph's own weights.
"""
function Graphs.degree(g::GNNGraph{<:COO_T}, T::TT=nothing; dir=:out, edge_weight=true) where 
        TT<:Union{Nothing, Type{<:Number}}
    s, t = edge_index(g)

    edge_weight = _get_edge_weight(g, edge_weight)
    edge_weight = edge_weight === nothing ? ones_like(s) : edge_weight

    T = isnothing(T) ? eltype(edge_weight) : T
    degs = fill!(similar(s, T, g.num_nodes), 0)

    if dir ∈ [:out, :both]
        degs = degs .+ NNlib.scatter(+, edge_weight, s, dstsize=(g.num_nodes,))
    end
    if dir ∈ [:in, :both]
        degs = degs .+ NNlib.scatter(+, edge_weight, t, dstsize=(g.num_nodes,))
    end
    return degs 
end

# TODO:: Make efficient
Graphs.degree(g::GNNGraph, i::Union{Int, AbstractVector}; dir=:out) = degree(g; dir)[i]

function Graphs.degree(g::GNNGraph{<:ADJMAT_T}, T::TT=nothing; dir=:out, edge_weight=true) where TT
        TT<:Union{Nothing, Type{<:Number}}
    # edge_weight=true or edge_weight=nothing act the same here
    @assert !(edge_weight isa AbstractArray) "passing the edge weights is not support by adjacency matrix representations" 
    @assert dir ∈ (:in, :out, :both)
    if T === nothing
        Nt = eltype(g)
        if edge_weight === false && !(Nt <: Integer) 
            T = Nt == Float32 ? Int32 : 
                Nt == Float16 ? Int16 : Int
        else
            T = Nt
        end
    end
    A = adjacency_matrix(g)
    if edge_weight === false
        A = binarize(A)
    end
    A = eltype(A) != T ? T.(A) : A
    return dir == :out ? vec(sum(A, dims=2)) : 
           dir == :in  ? vec(sum(A, dims=1)) :
                  vec(sum(A, dims=1)) .+ vec(sum(A, dims=2)) 
end

function Graphs.laplacian_matrix(g::GNNGraph, T::DataType=eltype(g); dir::Symbol=:out)
    A = adjacency_matrix(g, T; dir=dir)
    D = Diagonal(vec(sum(A; dims=2)))
    return D - A
end


"""
    normalized_laplacian(g, T=Float32; add_self_loops=false, dir=:out)

Normalized Laplacian matrix of graph `g`.

# Arguments

- `g`: A `GNNGraph`.
- `T`: result element type.
- `add_self_loops`: add self-loops while calculating the matrix.
- `dir`: the edge directionality considered (:out, :in, :both).
"""
function normalized_laplacian(g::GNNGraph, T::DataType=Float32; 
                        add_self_loops::Bool=false, dir::Symbol=:out)
    Ã = normalized_adjacency(g, T; dir, add_self_loops)
    return I - Ã
end

function normalized_adjacency(g::GNNGraph, T::DataType=Float32; 
                        add_self_loops::Bool=false, dir::Symbol=:out)
    A = adjacency_matrix(g, T; dir=dir)
    if add_self_loops
        A = A + I
    end
    degs = vec(sum(A; dims=2))
    inv_sqrtD = Diagonal(inv.(sqrt.(degs)))
    return inv_sqrtD * A * inv_sqrtD
end

@doc raw"""
    scaled_laplacian(g, T=Float32; dir=:out)

Scaled Laplacian matrix of graph `g`,
defined as ``\hat{L} = \frac{2}{\lambda_{max}} L - I`` where ``L`` is the normalized Laplacian matrix.

# Arguments

- `g`: A `GNNGraph`.
- `T`: result element type.
- `dir`: the edge directionality considered (:out, :in, :both).
"""
function scaled_laplacian(g::GNNGraph, T::DataType=Float32; dir=:out)
    L = normalized_laplacian(g, T)
    # @assert issymmetric(L) "scaled_laplacian only works with symmetric matrices"
    λmax = _eigmax(L)
    return  2 / λmax * L - I
end

# _eigmax(A) = eigmax(Symmetric(A)) # Doesn't work on sparse arrays
function _eigmax(A)
    x0 = _rand_dense_vector(A)
    KrylovKit.eigsolve(Symmetric(A), x0, 1, :LR)[1][1] # also eigs(A, x0, nev, mode) available 
end

_rand_dense_vector(A::AbstractMatrix{T}) where T = randn(float(T), size(A, 1))
_rand_dense_vector(A::CUMAT_T)= CUDA.randn(size(A, 1))

# Eigenvalues for cuarray don't seem to be well supported. 
# https://github.com/JuliaGPU/CUDA.jl/issues/154
# https://discourse.julialang.org/t/cuda-eigenvalues-of-a-sparse-matrix/46851/5

"""
    graph_indicator(g)

Return a vector containing the graph membership
(an integer from `1` to `g.num_graphs`) of each node in the graph.
"""
function graph_indicator(g; edges=false)
    if isnothing(g.graph_indicator)
        gi = ones_like(edge_index(g)[1], Int, g.num_nodes)
    else 
        gi = g.graph_indicator
    end
    if edges
        s, t = edge_index(g)
        return gi[s]
    else
        return gi
    end
end

function node_features(g::GNNGraph)
    if isempty(g.ndata)
        return nothing
    elseif length(g.ndata) > 1
        @error "Multiple feature arrays, access directly through `g.ndata`"
    else
        return g.ndata[1]
    end
end

function edge_features(g::GNNGraph)
    if isempty(g.edata)
        return nothing
    elseif length(g.edata) > 1
        @error "Multiple feature arrays, access directly through `g.edata`"
    else
        return g.edata[1]
    end
end

function graph_features(g::GNNGraph)
    if isempty(g.gdata)
        return nothing
    elseif length(g.gdata) > 1
        @error "Multiple feature arrays, access directly through `g.gdata`"
    else
        return g.gdata[1]
    end
end

"""
    is_bidirected(g::GNNGraph)

Check if the directed graph `g` essentially corresponds
to an undirected graph, i.e. if for each edge it also contains the 
reverse edge. 
"""
function is_bidirected(g::GNNGraph)
    s, t = edge_index(g)
    s1, t1 = sort_edge_index(s, t)
    s2, t2 = sort_edge_index(t, s)
    all((s1 .== s2) .& (t1 .== t2))
end

"""
    has_self_loops(g::GNNGraph)

Return `true` if `g` has any self loops.
"""
function Graphs.has_self_loops(g::GNNGraph)
    s, t = edge_index(g)
    any(s .== t)
end

"""
    has_multi_edges(g::GNNGraph)

Return `true` if `g` has any multiple edges.
"""
function has_multi_edges(g::GNNGraph)
    s, t = edge_index(g)
    idxs, _ = edge_encoding(s, t, g.num_nodes)
    length(union(idxs)) < length(idxs)
end

@non_differentiable edge_index(x...)
@non_differentiable adjacency_list(x...)
@non_differentiable graph_indicator(x...)
@non_differentiable has_multi_edges(x...)
@non_differentiable Graphs.has_self_loops(x...) 
@non_differentiable is_bidirected(x...)
@non_differentiable normalized_adjacency(x...) # TODO remove this in the future
@non_differentiable normalized_laplacian(x...) # TODO remove this in the future
@non_differentiable scaled_laplacian(x...) # TODO remove this in the future
