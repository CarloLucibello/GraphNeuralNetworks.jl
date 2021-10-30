
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

edge_weight(g::GNNGraph{<:COO_T}) = g.graph[3]

edge_weight(g::GNNGraph{<:ADJMAT_T}) = to_coo(g.graph, num_nodes=g.num_nodes)[1][3]

Graphs.edges(g::GNNGraph) = zip(edge_index(g)...)

Graphs.edgetype(g::GNNGraph) = Tuple{Int, Int}

function Graphs.has_edge(g::GNNGraph{<:COO_T}, i::Integer, j::Integer)
    s, t = edge_index(g)
    return any((s .== i) .& (t .== j))
end

Graphs.has_edge(g::GNNGraph{<:ADJMAT_T}, i::Integer, j::Integer) = g.graph[i,j] != 0

Graphs.nv(g::GNNGraph) = g.num_nodes
Graphs.ne(g::GNNGraph) = g.num_edges
Graphs.has_vertex(g::GNNGraph, i::Int) = 1 <= i <= g.num_nodes
Graphs.vertices(g::GNNGraph) = 1:g.num_nodes

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

Return the adjacency list representation (a vector of vectors)
of the graph `g`.

Calling `a` the adjacency list, if `dir=:out`
`a[i]` will contain the neighbors of node `i` through
outgoing edges. If `dir=:in`, it will contain neighbors from
incoming edges instead.
"""
function adjacency_list(g::GNNGraph; dir=:out)
    @assert dir ∈ [:out, :in]
    fneighs = dir == :out ? outneighbors : inneighbors
    return [fneighs(g, i) for i in 1:g.num_nodes]
end

function Graphs.adjacency_matrix(g::GNNGraph{<:COO_T}, T::DataType=Int; dir=:out)
    if g.graph[1] isa CuVector
        # TODO revisi after https://github.com/JuliaGPU/CUDA.jl/pull/1152
        A, n, m = to_dense(g.graph, T, num_nodes=g.num_nodes)
    else
        A, n, m = to_sparse(g.graph, T, num_nodes=g.num_nodes)
    end
    @assert size(A) == (n, n)
    return dir == :out ? A : A'
end

function Graphs.adjacency_matrix(g::GNNGraph{<:ADJMAT_T}, T::DataType=eltype(g.graph); dir=:out)
    @assert dir ∈ [:in, :out]
    A = g.graph
    A = T != eltype(A) ? T.(A) : A
    return dir == :out ? A : A'
end

function Graphs.degree(g::GNNGraph{<:COO_T}, T=nothing; dir=:out)
    s, t = edge_index(g)
    T = isnothing(T) ? eltype(s) : T
    degs = fill!(similar(s, T, g.num_nodes), 0)
    src = 1
    if dir ∈ [:out, :both]
        NNlib.scatter!(+, degs, src, s)
    end
    if dir ∈ [:in, :both]
        NNlib.scatter!(+, degs, src, t)
    end
    return degs 
end

function Graphs.degree(g::GNNGraph{<:ADJMAT_T}, T=Int; dir=:out)
    @assert dir ∈ (:in, :out)
    A = adjacency_matrix(g, T)
    return dir == :out ? vec(sum(A, dims=2)) : vec(sum(A, dims=1))
end

function Graphs.laplacian_matrix(g::GNNGraph, T::DataType=Int; dir::Symbol=:out)
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
    @assert issymmetric(L) "scaled_laplacian only works with symmetric matrices"
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

@non_differentiable normalized_laplacian(x...)
@non_differentiable normalized_adjacency(x...)
@non_differentiable scaled_laplacian(x...)
@non_differentiable adjacency_matrix(x...)
@non_differentiable adjacency_list(x...)
@non_differentiable degree(x...)
@non_differentiable graph_indicator(x...)
