#===================================
Define GNNGraph type as a subtype of LightGraphs' AbstractGraph.
For the core methods to be implemented by any AbstractGraph, see
https://juliagraphs.org/LightGraphs.jl/latest/types/#AbstractGraph-Type
https://juliagraphs.org/LightGraphs.jl/latest/developing/#Developing-Alternate-Graph-Types
=============================================#

const COO_T = Tuple{T, T, V} where {T <: AbstractVector, V}
const ADJLIST_T = AbstractVector{T} where T <: AbstractVector
const ADJMAT_T = AbstractMatrix
const SPARSE_T = AbstractSparseMatrix # subset of ADJMAT_T

"""
    GNNGraph(data; [graph_type, nf, ef, gf, num_nodes, num_graphs, graph_indicator, dir])
    GNNGraph(g::GNNGraph; [nf, ef, gf])

A type representing a graph structure and storing also arrays 
that contain features associated to nodes, edges, and the whole graph. 
    
A `GNNGraph` can be constructed out of different objects `data` representing
the connections inside the graph, while the internal representation type
is governed by `graph_type`. 
When constructed from another graph `g`, the internal graph representation
is preserved and shared. 

A `GNNGraph` can also represent multiple graphs batched togheter 
(see [`Flux.batch`](@ref) or [`SparseArrays.blockdiag`](@ref)).
The field `g.graph_indicator` contains the graph membership
of each node.

A `GNNGraph` is a LightGraphs' `AbstractGraph`, therefore any functionality
from the LightGraphs' graph library can be used on it.

# Arguments 

- `data`: Some data representing the graph topology. Possible type are 
    - An adjacency matrix
    - An adjacency list.
    - A tuple containing the source and target vectors (COO representation)
    - A LightGraphs' graph.
- `graph_type`: A keyword argument that specifies 
                the underlying representation used by the GNNGraph. 
                Currently supported values are 
    - `:coo`. Graph represented as a tuple `(source, target)`, such that the `k`-th edge 
              connects the node `source[k]` to node `target[k]`.
              Optionally, also edge weights can be given: `(source, target, weights)`.
    - `:sparse`. A sparse adjacency matrix representation.
    - `:dense`. A dense adjacency matrix representation.  
    Default `:coo`.
- `dir`. The assumed edge direction when given adjacency matrix or adjacency list input data `g`. 
        Possible values are `:out` and `:in`. Default `:out`.
- `num_nodes`. The number of nodes. If not specified, inferred from `g`. Default `nothing`.
- `num_graphs`. The number of graphs. Larger than 1 in case of batched graphs. Default `1`.
- `graph_indicator`. For batched graphs, a vector containeing the graph assigment of each node. Default `nothing`.  
- `nf`: Node features. Either nothing, or an array whose last dimension has size num_nodes. Default `nothing`.
- `ef`: Edge features. Either nothing, or an array whose last dimension has size num_edges. Default `nothing`.
- `gf`: Global features. Default `nothing`. 

# Usage. 

```
using Flux, GraphNeuralNetworks

# Construct from adjacency list representation
data = [[2,3], [1,4,5], [1], [2,5], [2,4]]
g = GNNGraph(data)

# Number of nodes and edges
g.num_nodes  # 5
g.num_edges  # 10 

# Same graph in COO representation
s = [1,1,2,2,2,3,4,4,5,5]
t = [2,3,1,4,5,3,2,5,2,4]
g = GNNGraph(s, t)

# From a LightGraphs' graph
g = GNNGraph(erdos_renyi(100, 20))

# Copy graph while also adding node features
g = GNNGraph(g, nf=rand(100, 5))

# Send to gpu
g = g |> gpu

# Collect edges' source and target nodes.
# Both source and target are vectors of length num_edges
source, target = edge_index(g)
```

See also [`graph`](@ref), [`edge_index`](@ref), [`node_feature`](@ref), [`edge_feature`](@ref), and [`global_feature`](@ref) 
"""
struct GNNGraph{T<:Union{COO_T,ADJMAT_T}}
    graph::T
    num_nodes::Int
    num_edges::Int
    num_graphs::Int
    graph_indicator
    nf
    ef
    gf
    ## possible future property stores
    # ndata::Dict{String, Any} # https://github.com/FluxML/Zygote.jl/issues/717        
    # edata::Dict{String, Any}
    # gdata::Dict{String, Any}
end

@functor GNNGraph

function GNNGraph(data; 
                        num_nodes = nothing,
                        num_graphs = 1,
                        graph_indicator = nothing, 
                        graph_type = :coo,
                        dir = :out,
                        nf = nothing, 
                        ef = nothing, 
                        gf = nothing,
                        # ndata = Dict{String, Any}(), 
                        # edata = Dict{String, Any}(),
                        # gdata = Dict{String, Any}()
                        )

    @assert graph_type ∈ [:coo, :dense, :sparse] "Invalid graph_type $graph_type requested"
    @assert dir ∈ [:in, :out]
    if graph_type == :coo
        g, num_nodes, num_edges = to_coo(data; num_nodes, dir)
    elseif graph_type == :dense
        g, num_nodes, num_edges = to_dense(data; dir)
    elseif graph_type == :sparse
        g, num_nodes, num_edges = to_sparse(data; dir)
    end
    if num_graphs > 1
        @assert len(graph_indicator) = num_nodes "When batching multiple graphs `graph_indicator` should be filled with the nodes' memberships."
    end 

    ## Possible future implementation of feature maps. 
    ## Currently this doesn't play well with zygote due to 
    ## https://github.com/FluxML/Zygote.jl/issues/717    
    # ndata["x"] = nf
    # edata["e"] = ef
    # gdata["g"] = gf
    
    GNNGraph(g, num_nodes, num_edges, 
            num_graphs, graph_indicator,
            nf, ef, gf)
end

# COO convenience constructors
GNNGraph(s::AbstractVector, t::AbstractVector, v = nothing; kws...) = GNNGraph((s, t, v); kws...)
GNNGraph((s, t)::NTuple{2}; kws...) = GNNGraph((s, t, nothing); kws...)

# GNNGraph(g::AbstractGraph; kws...) = GNNGraph(adjacency_matrix(g, dir=:out); kws...)

function GNNGraph(g::AbstractGraph; kws...)
    s = LightGraphs.src.(LightGraphs.edges(g))
    t = LightGraphs.dst.(LightGraphs.edges(g)) 
    GNNGraph((s, t); kws...)
end

function GNNGraph(g::GNNGraph; 
                nf=node_feature(g), ef=edge_feature(g), gf=global_feature(g))
                # ndata=copy(g.ndata), edata=copy(g.edata), gdata=copy(g.gdata), # copy keeps the refs to old data 
    
    GNNGraph(g.graph, g.num_nodes, g.num_edges, g.num_graphs, g.graph_indicator, nf, ef, gf) #   ndata, edata, gdata, 
end


"""
    edge_index(g::GNNGraph)

Return a tuple containing two vectors, respectively storing 
the source and target nodes for each edges in `g`.

```julia
s, t = edge_index(g)
```
"""
edge_index(g::GNNGraph{<:COO_T}) = graph(g)[1:2]

edge_index(g::GNNGraph{<:ADJMAT_T}) = to_coo(graph(g))[1][1:2]

edge_weight(g::GNNGraph{<:COO_T}) = graph(g)[3]

"""
    graph(g::GNNGraph)

Return the underlying implementation of the graph structure of `g`,
either an adjacency matrix or an edge list in the COO format.
"""
graph(g::GNNGraph) = g.graph

LightGraphs.edges(g::GNNGraph) = zip(edge_index(g)...)

LightGraphs.edgetype(g::GNNGraph) = Tuple{Int, Int}

function LightGraphs.has_edge(g::GNNGraph{<:COO_T}, i::Integer, j::Integer)
    s, t = edge_index(g)
    return any((s .== i) .& (t .== j))
end

LightGraphs.has_edge(g::GNNGraph{<:ADJMAT_T}, i::Integer, j::Integer) = graph(g)[i,j] != 0

LightGraphs.nv(g::GNNGraph) = g.num_nodes
LightGraphs.ne(g::GNNGraph) = g.num_edges
LightGraphs.has_vertex(g::GNNGraph, i::Int) = 1 <= i <= g.num_nodes
LightGraphs.vertices(g::GNNGraph) = 1:g.num_nodes

function LightGraphs.outneighbors(g::GNNGraph{<:COO_T}, i::Integer)
    s, t = edge_index(g)
    return t[s .== i]
end

function LightGraphs.outneighbors(g::GNNGraph{<:ADJMAT_T}, i::Integer)
    A = graph(g)
    return findall(!=(0), A[i,:])
end

function LightGraphs.inneighbors(g::GNNGraph{<:COO_T}, i::Integer)
    s, t = edge_index(g)
    return s[t .== i]
end

function LightGraphs.inneighbors(g::GNNGraph{<:ADJMAT_T}, i::Integer)
    A = graph(g)
    return findall(!=(0), A[:,i])
end

LightGraphs.is_directed(::GNNGraph) = true
LightGraphs.is_directed(::Type{GNNGraph}) = true

function adjacency_list(g::GNNGraph; dir=:out)
    @assert dir ∈ [:out, :in]
    fneighs = dir == :out ? outneighbors : inneighbors
    return [fneighs(g, i) for i in 1:g.num_nodes]
end

function LightGraphs.adjacency_matrix(g::GNNGraph{<:COO_T}, T::DataType=Int; dir=:out)
    A, n, m = to_sparse(graph(g), T, num_nodes=g.num_nodes)
    @assert size(A) == (n, n)
    return dir == :out ? A : A'
end

function LightGraphs.adjacency_matrix(g::GNNGraph{<:ADJMAT_T}, T::DataType=eltype(graph(g)); dir=:out)
    @assert dir ∈ [:in, :out]
    A = graph(g) 
    A = T != eltype(A) ? T.(A) : A
    return dir == :out ? A : A'
end

function LightGraphs.degree(g::GNNGraph{<:COO_T}, T=Int; dir=:out)
    s, t = edge_index(g)
    degs = fill!(similar(s, T, g.num_nodes), 0)
    o = fill!(similar(s, Int, g.num_edges), 1)
    if dir ∈ [:out, :both]
        NNlib.scatter!(+, degs, o, s)
    end
    if dir ∈ [:in, :both]
        NNlib.scatter!(+, degs, o, t)
    end
    return degs
end

function LightGraphs.degree(g::GNNGraph{<:ADJMAT_T}, T=Int; dir=:out)
    @assert dir ∈ (:in, :out)
    A = adjacency_matrix(g, T)
    return dir == :out ? vec(sum(A, dims=2)) : vec(sum(A, dims=1))
end

# node_feature(g::GNNGraph) = g.ndata["x"]
# edge_feature(g::GNNGraph) = g.edata["e"]
# global_feature(g::GNNGraph) = g.gdata["g"]


"""
    node_feature(g::GNNGraph)

Return the node features of `g`.
"""
node_feature(g::GNNGraph) = g.nf

"""
    edge_feature(g::GNNGraph)

Return the edge features of `g`.
"""
edge_feature(g::GNNGraph) = g.ef

"""
    global_feature(g::GNNGraph)

Return the global features of `g`.
"""
global_feature(g::GNNGraph) = g.gf

# function Base.getproperty(g::GNNGraph, sym::Symbol)
#     if sym === :nf
#         return g.ndata["x"]
#     elseif sym === :ef
#         return g.edata["e"]
#     elseif sym === :gf
#         return g.gdata["g"]
#     else # fallback to getfield
#         return getfield(g, sym)
#     end
# end

function LightGraphs.laplacian_matrix(g::GNNGraph, T::DataType=Int; dir::Symbol=:out)
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
        A += I
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
_eigmax(A) = KrylovKit.eigsolve(Symmetric(A), 1, :LR)[1][1] # also eigs(A, x0, nev, mode) available 

# Eigenvalues for cuarray don't seem to be well supported. 
# https://github.com/JuliaGPU/CUDA.jl/issues/154
# https://discourse.julialang.org/t/cuda-eigenvalues-of-a-sparse-matrix/46851/5

"""
    add_self_loops(g::GNNGraph)

Return a graph with the same features as `g`
but also adding edges connecting the nodes to themselves.

Nodes with already existing
self-loops will obtain a second self-loop.
"""
function add_self_loops(g::GNNGraph{<:COO_T})
    s, t = edge_index(g)
    @assert edge_feature(g) === nothing
    @assert edge_weight(g) === nothing
    n = g.num_nodes
    nodes = convert(typeof(s), [1:n;])
    s = [s; nodes]
    t = [t; nodes]

    GNNGraph((s, t, nothing), g.num_nodes, length(s),
        g.num_graphs, g.graph_indicator,
        node_feature(g), edge_feature(g), global_feature(g))
end

function add_self_loops(g::GNNGraph{<:ADJMAT_T}; add_to_existing=true)
    A = graph(g)
    @assert edge_feature(g) === nothing
    A += I
    num_edges =  g.num_edges + g.num_nodes
    GNNGraph(A, g.num_nodes, num_edges,
        g.num_graphs, g.graph_indicator,
        node_feature(g), edge_feature(g), global_feature(g))
end

function remove_self_loops(g::GNNGraph{<:COO_T})
    s, t = edge_index(g)
    # TODO remove these constraints
    @assert edge_feature(g) === nothing
    @assert edge_weight(g) === nothing
    
    mask_old_loops = s .!= t
    s = s[mask_old_loops]
    t = t[mask_old_loops]

    GNNGraph((s, t, nothing), g.num_nodes, length(s), 
        g.num_graphs, g.graph_indicator,
        node_feature(g), edge_feature(g), global_feature(g))
end

function _catgraphs(g1::GNNGraph{<:COO_T}, g2::GNNGraph{<:COO_T})
    s1, t1 = edge_index(g1)
    s2, t2 = edge_index(g2)
    nv1, nv2 = g1.num_nodes, g2.num_nodes
    s = vcat(s1, nv1 .+ s2)
    t = vcat(t1, nv1 .+ t2)
    w = cat_features(edge_weight(g1), edge_weight(g2))

    ind1 = isnothing(g1.graph_indicator) ? fill!(similar(s1, Int, nv1), 1) : g1.graph_indicator 
    ind2 = isnothing(g2.graph_indicator) ? fill!(similar(s2, Int, nv2), 1) : g2.graph_indicator 
    graph_indicator = vcat(ind1, g1.num_graphs .+ ind2)
    
    GNNGraph(
        (s, t, w),
        nv1 + nv2, g1.num_edges + g2.num_edges, 
        g1.num_graphs + g2.num_graphs, graph_indicator,
        cat_features(node_feature(g1), node_feature(g2)),
        cat_features(edge_feature(g1), edge_feature(g2)),
        cat_features(global_feature(g1), global_feature(g2)),
    )
end

# Cat public interfaces

```
    blockdiag(xs::GNNGraph...)

Batch togheter multiple `GNNGraph`s into a single one 
containing the total number of nodes and edges of the original graphs.

Equivalent to [`Flux.batch`](@ref).
```
function SparseArrays.blockdiag(g1::GNNGraph, gothers::GNNGraph...)
    @assert length(gothers) >= 1
    g = g1
    for go in gothers
        g = _catgraphs(g, go)
    end
    return g
end

```
    batch(xs::Vector{<:GNNGraph})

Batch togheter multiple `GNNGraph`s into a single one 
containing the total number of nodes and edges of the original graphs.

Equivalent to [`SparseArrays.blockdiag`](@ref).
```
Flux.batch(xs::Vector{<:GNNGraph}) = blockdiag(xs...)
#########################

@non_differentiable normalized_laplacian(x...)
@non_differentiable normalized_adjacency(x...)
@non_differentiable scaled_laplacian(x...)
@non_differentiable adjacency_matrix(x...)
@non_differentiable adjacency_list(x...)
@non_differentiable degree(x...)
@non_differentiable add_self_loops(x...)     # TODO this is wrong, since g carries feature arrays, needs rrule
@non_differentiable remove_self_loops(x...)  # TODO this is wrong, since g carries feature arrays, needs rrule

# # delete when https://github.com/JuliaDiff/ChainRules.jl/pull/472 is merged
# function ChainRulesCore.rrule(::typeof(copy), x)
#     copy_pullback(ȳ) = (NoTangent(), ȳ)
#     return copy(x), copy_pullback
# end
