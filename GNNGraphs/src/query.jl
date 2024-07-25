
"""
    edge_index(g::GNNGraph)

Return a tuple containing two vectors, respectively storing 
the source and target nodes for each edges in `g`.

```julia
s, t = edge_index(g)
```
"""
edge_index(g::GNNGraph{<:COO_T}) = g.graph[1:2]

edge_index(g::GNNGraph{<:ADJMAT_T}) = to_coo(g.graph, num_nodes = g.num_nodes)[1][1:2]

"""
    edge_index(g::GNNHeteroGraph, [edge_t])

Return a tuple containing two vectors, respectively storing the source and target nodes
for each edges in `g` of type `edge_t = (src_t, rel_t, trg_t)`.

If `edge_t` is not provided, it will error if `g` has more than one edge type.
"""
edge_index(g::GNNHeteroGraph{<:COO_T}, edge_t::EType) = g.graph[edge_t][1:2]
edge_index(g::GNNHeteroGraph{<:COO_T}) = only(g.graph)[2][1:2]

get_edge_weight(g::GNNGraph{<:COO_T}) = g.graph[3]

get_edge_weight(g::GNNGraph{<:ADJMAT_T}) = to_coo(g.graph, num_nodes = g.num_nodes)[1][3]

get_edge_weight(g::GNNHeteroGraph{<:COO_T}, edge_t::EType) = g.graph[edge_t][3]

Graphs.edges(g::GNNGraph) = Graphs.Edge.(edge_index(g)...)

Graphs.edgetype(g::GNNGraph) = Graphs.Edge{eltype(g)}

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

Graphs.has_edge(g::GNNGraph{<:ADJMAT_T}, i::Integer, j::Integer) = g.graph[i, j] != 0

"""
    has_edge(g::GNNHeteroGraph, edge_t, i, j)

Return `true` if there is an edge of type `edge_t` from node `i` to node `j` in `g`.

# Examples

```jldoctest
julia> g = rand_bipartite_heterograph((2, 2), (4, 0), bidirected=false)
GNNHeteroGraph:
  num_nodes: (:A => 2, :B => 2)
  num_edges: ((:A, :to, :B) => 4, (:B, :to, :A) => 0)

julia> has_edge(g, (:A,:to,:B), 1, 1)
true

julia> has_edge(g, (:B,:to,:A), 1, 1)
false
```
"""
function Graphs.has_edge(g::GNNHeteroGraph, edge_t::EType, i::Integer, j::Integer)
    s, t = edge_index(g, edge_t)
    return any((s .== i) .& (t .== j))
end

graph_type_symbol(::GNNGraph{<:COO_T}) = :coo
graph_type_symbol(::GNNGraph{<:SPARSE_T}) = :sparse
graph_type_symbol(::GNNGraph{<:ADJMAT_T}) = :dense

Graphs.nv(g::GNNGraph) = g.num_nodes
Graphs.ne(g::GNNGraph) = g.num_edges
Graphs.has_vertex(g::GNNGraph, i::Int) = 1 <= i <= g.num_nodes
Graphs.vertices(g::GNNGraph) = 1:(g.num_nodes)


"""
    neighbors(g::GNNGraph, i::Integer; dir=:out)

Return the neighbors of node `i` in the graph `g`.
If `dir=:out`, return the neighbors through outgoing edges.
If `dir=:in`, return the neighbors through incoming edges.

See also [`outneighbors`](@ref Graphs.outneighbors), [`inneighbors`](@ref Graphs.inneighbors).
"""
function Graphs.neighbors(g::GNNGraph, i::Integer; dir::Symbol = :out)
    @assert dir ∈ (:in, :out)
    if dir == :out
        outneighbors(g, i)
    else
        inneighbors(g, i)
    end
end

"""
    outneighbors(g::GNNGraph, i::Integer)

Return the neighbors of node `i` in the graph `g` through outgoing edges.

See also [`neighbors`](@ref Graphs.neighbors) and [`inneighbors`](@ref Graphs.inneighbors).
"""
function Graphs.outneighbors(g::GNNGraph{<:COO_T}, i::Integer)
    s, t = edge_index(g)
    return t[s .== i]
end

function Graphs.outneighbors(g::GNNGraph{<:ADJMAT_T}, i::Integer)
    A = g.graph
    return findall(!=(0), A[i, :])
end

"""
    inneighbors(g::GNNGraph, i::Integer)

Return the neighbors of node `i` in the graph `g` through incoming edges.

See also [`neighbors`](@ref Graphs.neighbors) and [`outneighbors`](@ref Graphs.outneighbors).
"""
function Graphs.inneighbors(g::GNNGraph{<:COO_T}, i::Integer)
    s, t = edge_index(g)
    return s[t .== i]
end

function Graphs.inneighbors(g::GNNGraph{<:ADJMAT_T}, i::Integer)
    A = g.graph
    return findall(!=(0), A[:, i])
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
function adjacency_list(g::GNNGraph, nodes; dir = :out, with_eid = false)
    @assert dir ∈ [:out, :in]
    s, t = edge_index(g)
    if dir == :in
        s, t = t, s
    end
    T = eltype(s)
    idict = 0
    dmap = Dict(n => (idict += 1) for n in nodes)
    adjlist = [T[] for _ in 1:length(dmap)]
    eidlist = [T[] for _ in 1:length(dmap)]
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

adjacency_list(g::GNNGraph; dir = :out) = adjacency_list(g, 1:(g.num_nodes); dir)

"""
    adjacency_matrix(g::GNNGraph, T=eltype(g); dir=:out, weighted=true)

Return the adjacency matrix `A` for the graph `g`. 

If `dir=:out`, `A[i,j] > 0` denotes the presence of an edge from node `i` to node `j`.
If `dir=:in` instead, `A[i,j] > 0` denotes the presence of an edge from node `j` to node `i`.

User may specify the eltype `T` of the returned matrix. 

If `weighted=true`, the `A` will contain the edge weights if any, otherwise the elements of `A` will be either 0 or 1.
"""
function Graphs.adjacency_matrix(g::GNNGraph{<:COO_T}, T::DataType = eltype(g); dir = :out,
                                 weighted = true)
    if iscuarray(g.graph[1])
        # Revisit after 
        # https://github.com/JuliaGPU/CUDA.jl/issues/1113
        A, n, m = to_dense(g.graph, T; num_nodes = g.num_nodes, weighted)
    else
        A, n, m = to_sparse(g.graph, T; num_nodes = g.num_nodes, weighted)
    end
    @assert size(A) == (n, n)
    return dir == :out ? A : A'
end

function Graphs.adjacency_matrix(g::GNNGraph{<:ADJMAT_T}, T::DataType = eltype(g);
                                 dir = :out, weighted = true)
    @assert dir ∈ [:in, :out]
    A = g.graph
    if !weighted
        A = binarize(A)
    end
    A = T != eltype(A) ? T.(A) : A
    return dir == :out ? A : A'
end

function ChainRulesCore.rrule(::typeof(adjacency_matrix), g::G, T::DataType; 
            dir = :out, weighted = true) where {G <: GNNGraph{<:ADJMAT_T}}
    A = adjacency_matrix(g, T; dir, weighted)
    if !weighted
        function adjacency_matrix_pullback_noweight(Δ)
            return (NoTangent(), ZeroTangent(), NoTangent())  
        end
        return A, adjacency_matrix_pullback_noweight
    else
        function adjacency_matrix_pullback_weighted(Δ)
            dg = Tangent{G}(; graph = Δ .* binarize(A))
            return (NoTangent(), dg, NoTangent())  
        end
        return A, adjacency_matrix_pullback_weighted
    end
end

function ChainRulesCore.rrule(::typeof(adjacency_matrix), g::G, T::DataType; 
            dir = :out, weighted = true) where {G <: GNNGraph{<:COO_T}}
    A = adjacency_matrix(g, T; dir, weighted)
    w = get_edge_weight(g)
    if !weighted || w === nothing
        function adjacency_matrix_pullback_noweight(Δ)
            return (NoTangent(), ZeroTangent(), NoTangent())  
        end
        return A, adjacency_matrix_pullback_noweight
    else
        function adjacency_matrix_pullback_weighted(Δ)
            s, t = edge_index(g)
            dg = Tangent{G}(; graph = (NoTangent(), NoTangent(), NNlib.gather(Δ, s, t)))
            return (NoTangent(), dg, NoTangent())  
        end
        return A, adjacency_matrix_pullback_weighted
    end
end

function _get_edge_weight(g, edge_weight::Bool)
    if edge_weight === true
        return get_edge_weight(g)
    elseif edge_weight === false
        return nothing
    end
end

_get_edge_weight(g, edge_weight::AbstractVector) = edge_weight

"""
    degree(g::GNNGraph, T=nothing; dir=:out, edge_weight=true)

Return a vector containing the degrees of the nodes in `g`.

The gradient is propagated through this function only if `edge_weight` is `true`
or a vector.

# Arguments

- `g`: A graph.
- `T`: Element type of the returned vector. If `nothing`, is
       chosen based on the graph type and will be an integer
       if `edge_weight = false`. Default `nothing`.
- `dir`: For `dir = :out` the degree of a node is counted based on the outgoing edges.
         For `dir = :in`, the ingoing edges are used. If `dir = :both` we have the sum of the two.
- `edge_weight`: If `true` and the graph contains weighted edges, the degree will 
                be weighted. Set to `false` instead to just count the number of
                outgoing/ingoing edges. 
                Finally, you can also pass a vector of weights to be used
                instead of the graph's own weights.
                Default `true`.

"""
function Graphs.degree(g::GNNGraph{<:COO_T}, T::TT = nothing; dir = :out,
                       edge_weight = true) where {
                                                  TT <: Union{Nothing, Type{<:Number}}}
    s, t = edge_index(g)

    ew = _get_edge_weight(g, edge_weight)
    
    T = if isnothing(T)
            if !isnothing(ew)
                eltype(ew)
            else
                eltype(s)
            end
        else 
            T
        end
    return _degree((s, t), T, dir, ew, g.num_nodes)
end

# TODO:: Make efficient
Graphs.degree(g::GNNGraph, i::Union{Int, AbstractVector}; dir = :out) = degree(g; dir)[i]

function Graphs.degree(g::GNNGraph{<:ADJMAT_T}, T::TT = nothing; dir = :out,
                       edge_weight = true) where {TT<:Union{Nothing, Type{<:Number}}}
    
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
    return _degree(A, T, dir, edge_weight, g.num_nodes)
end

"""
    degree(g::GNNHeteroGraph, edge_type::EType; dir = :in) 

Return a vector containing the degrees of the nodes in `g` GNNHeteroGraph
given `edge_type`.

# Arguments

- `g`: A graph.
- `edge_type`: A tuple of symbols `(source_t, edge_t, target_t)` representing the edge type.
- `T`: Element type of the returned vector. If `nothing`, is
       chosen based on the graph type. Default `nothing`.
- `dir`: For `dir = :out` the degree of a node is counted based on the outgoing edges.
         For `dir = :in`, the ingoing edges are used. If `dir = :both` we have the sum of the two.
         Default `dir = :out`.

"""
function Graphs.degree(g::GNNHeteroGraph, edge::EType, 
                       T::TT = nothing; dir = :out) where {
                                                         TT <: Union{Nothing, Type{<:Number}}}  

    s, t = edge_index(g, edge)

    T = isnothing(T) ? eltype(s) : T

    n_type = dir == :in ? g.ntypes[2] : g.ntypes[1]

    return _degree((s, t), T, dir, nothing, g.num_nodes[n_type])
end

function _degree((s, t)::Tuple, T::Type, dir::Symbol, edge_weight::Nothing, num_nodes::Int)
    _degree((s, t), T, dir, ones_like(s, T), num_nodes)
end

function _degree((s, t)::Tuple, T::Type, dir::Symbol, edge_weight::AbstractVector, num_nodes::Int)
    degs = zeros_like(s, T, num_nodes)

    if dir ∈ [:out, :both]
        degs = degs .+ NNlib.scatter(+, edge_weight, s, dstsize = (num_nodes,))
    end
    if dir ∈ [:in, :both]
        degs = degs .+ NNlib.scatter(+, edge_weight, t, dstsize = (num_nodes,))
    end
    return degs
end

function _degree(A::AbstractMatrix, T::Type, dir::Symbol, edge_weight::Bool, num_nodes::Int)
    if edge_weight === false
        A = binarize(A)
    end
    A = eltype(A) != T ? T.(A) : A
    return dir == :out ? vec(sum(A, dims = 2)) :
           dir == :in ? vec(sum(A, dims = 1)) :
           vec(sum(A, dims = 1)) .+ vec(sum(A, dims = 2))
end

function ChainRulesCore.rrule(::typeof(_degree), graph, T, dir, edge_weight::Nothing, num_nodes)
    degs = _degree(graph, T, dir, edge_weight, num_nodes)
    function _degree_pullback(Δ)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return degs, _degree_pullback
end

function ChainRulesCore.rrule(::typeof(_degree), A::ADJMAT_T, T, dir, edge_weight::Bool, num_nodes)
    degs = _degree(A, T, dir, edge_weight, num_nodes)
    if edge_weight === false
        function _degree_pullback_noweights(Δ)
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end
        return degs, _degree_pullback_noweights
    else
        function _degree_pullback_weights(Δ)
            # We propagate the gradient only to the non-zero elements
            # of the adjacency matrix.
            bA = binarize(A)
            if dir == :in
                dA = bA .* Δ'
            elseif dir == :out
                dA = Δ .* bA
            else # dir == :both
                dA = Δ .* bA + Δ' .* bA
            end
            return (NoTangent(), dA, NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end
        return degs, _degree_pullback_weights
    end
end

"""
    has_isolated_nodes(g::GNNGraph; dir=:out)

Return true if the graph `g` contains nodes with out-degree (if `dir=:out`)
or in-degree (if `dir = :in`) equal to zero.
"""
function has_isolated_nodes(g::GNNGraph; dir = :out)
    return any(iszero, degree(g; dir))
end

function Graphs.laplacian_matrix(g::GNNGraph, T::DataType = eltype(g); dir::Symbol = :out)
    A = adjacency_matrix(g, T; dir = dir)
    D = Diagonal(vec(sum(A; dims = 2)))
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
function normalized_laplacian(g::GNNGraph, T::DataType = Float32;
                              add_self_loops::Bool = false, dir::Symbol = :out)
    Ã = normalized_adjacency(g, T; dir, add_self_loops)
    return I - Ã
end

function normalized_adjacency(g::GNNGraph, T::DataType = Float32;
                              add_self_loops::Bool = false, dir::Symbol = :out)
    A = adjacency_matrix(g, T; dir = dir)
    if add_self_loops
        A = A + I
    end
    degs = vec(sum(A; dims = 2))
    ChainRulesCore.ignore_derivatives() do
        @assert all(!iszero, degs) "Graph contains isolated nodes, cannot compute `normalized_adjacency`."
    end
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
function scaled_laplacian(g::GNNGraph, T::DataType = Float32; dir = :out)
    L = normalized_laplacian(g, T)
    # @assert issymmetric(L) "scaled_laplacian only works with symmetric matrices"
    λmax = _eigmax(L)
    return 2 / λmax * L - I
end

# _eigmax(A) = eigmax(Symmetric(A)) # Doesn't work on sparse arrays
function _eigmax(A)
    x0 = _rand_dense_vector(A)
    KrylovKit.eigsolve(Symmetric(A), x0, 1, :LR)[1][1] # also eigs(A, x0, nev, mode) available 
end

_rand_dense_vector(A::AbstractMatrix{T}) where {T} = randn(float(T), size(A, 1))

# Eigenvalues for cuarray don't seem to be well supported. 
# https://github.com/JuliaGPU/CUDA.jl/issues/154
# https://discourse.julialang.org/t/cuda-eigenvalues-of-a-sparse-matrix/46851/5

"""
    graph_indicator(g::GNNGraph; edges=false)

Return a vector containing the graph membership
(an integer from `1` to `g.num_graphs`) of each node in the graph.
If `edges=true`, return the graph membership of each edge instead.
"""
function graph_indicator(g::GNNGraph; edges = false)
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

"""
    graph_indicator(g::GNNHeteroGraph, [node_t])

Return a Dict of vectors containing the graph membership
(an integer from `1` to `g.num_graphs`) of each node in the graph for each node type.
If `node_t` is provided, return the graph membership of each node of type `node_t` instead.

See also [`batch`](@ref).
"""
function graph_indicator(g::GNNHeteroGraph)
    return g.graph_indicator
end

function graph_indicator(g::GNNHeteroGraph, node_t::Symbol)
    @assert node_t ∈ g.ntypes
    if isnothing(g.graph_indicator)
        gi = ones_like(edge_index(g, first(g.etypes))[1], Int, g.num_nodes[node_t])
    else
        gi = g.graph_indicator[node_t]
    end
    return gi
end

function node_features(g::GNNGraph)
    if isempty(g.ndata)
        return nothing
    elseif length(g.ndata) > 1
        @error "Multiple feature arrays, access directly through `g.ndata`"
    else
        return first(values(g.ndata))
    end
end

function edge_features(g::GNNGraph)
    if isempty(g.edata)
        return nothing
    elseif length(g.edata) > 1
        @error "Multiple feature arrays, access directly through `g.edata`"
    else
        return first(values(g.edata))
    end
end

function graph_features(g::GNNGraph)
    if isempty(g.gdata)
        return nothing
    elseif length(g.gdata) > 1
        @error "Multiple feature arrays, access directly through `g.gdata`"
    else
        return first(values(g.gdata))
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

"""
    khop_adj(g::GNNGraph,k::Int,T::DataType=eltype(g); dir=:out, weighted=true)

Return ``A^k`` where ``A`` is the adjacency matrix of the graph 'g'.

"""
function khop_adj(g::GNNGraph, k::Int, T::DataType = eltype(g); dir = :out, weighted = true)
    return (adjacency_matrix(g, T; dir, weighted))^k
end

"""
    laplacian_lambda_max(g::GNNGraph, T=Float32; add_self_loops=false, dir=:out)

Return the largest eigenvalue of the normalized symmetric Laplacian of the graph `g`.

If the graph is batched from multiple graphs, return the list of the largest eigenvalue for each graph.
"""
function laplacian_lambda_max(g::GNNGraph, T::DataType = Float32;
                              add_self_loops::Bool = false, dir::Symbol = :out)
    if g.num_graphs == 1
        return _eigmax(normalized_laplacian(g, T; add_self_loops, dir))
    else
        eigenvalues = zeros(g.num_graphs)
        for i in 1:(g.num_graphs)
            eigenvalues[i] = _eigmax(normalized_laplacian(getgraph(g, i), T; add_self_loops,
                                                          dir))
        end
        return eigenvalues
    end
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
