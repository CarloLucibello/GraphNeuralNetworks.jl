#===================================
Define GNNGraph type as a subtype of Graphs' AbstractGraph.
For the core methods to be implemented by any AbstractGraph, see
https://juliagraphs.org/Graphs.jl/latest/types/#AbstractGraph-Type
https://juliagraphs.org/Graphs.jl/latest/developing/#Developing-Alternate-Graph-Types
=============================================#

const COO_T = Tuple{T, T, V} where {T <: AbstractVector{<:Integer}, V}
const ADJLIST_T = AbstractVector{T} where T <: AbstractVector{<:Integer}
const ADJMAT_T = AbstractMatrix
const SPARSE_T = AbstractSparseMatrix # subset of ADJMAT_T
const CUMAT_T = Union{CUDA.AnyCuMatrix, CUDA.CUSPARSE.CuSparseMatrix}


"""
    GNNGraph(data; [graph_type, ndata, edata, gdata, num_nodes, graph_indicator, dir])
    GNNGraph(g::GNNGraph; [ndata, edata, gdata])

A type representing a graph structure that also stores 
feature arrays associated to nodes, edges, and the graph itself. 

A `GNNGraph` can be constructed out of different `data` objects 
expressing the connections inside the graph. The internal representation type
is determined by `graph_type`.

When constructed from another `GNNGraph`, the internal graph representation
is preserved and shared. The node/edge/graph features are retained
as well, unless explicitely set by the keyword arguments
`ndata`, `edata`, and `gdata`.

A `GNNGraph` can also represent multiple graphs batched togheter 
(see [`Flux.batch`](@ref) or [`SparseArrays.blockdiag`](@ref)).
The field `g.graph_indicator` contains the graph membership
of each node.

`GNNGraph`s are always directed graphs, therefore each edge is defined
by a source node and a target node (see [`edge_index`](@ref)).
Self loops (edges connecting a node to itself) and multiple edges
(more than one edge between the same pair of nodes) are supported.

A `GNNGraph` is a Graphs.jl's `AbstractGraph`, therefore it supports most 
functionality from that library.

# Arguments 

- `data`: Some data representing the graph topology. Possible type are 
    - An adjacency matrix
    - An adjacency list.
    - A tuple containing the source and target vectors (COO representation)
    - A Graphs.jl' graph.
- `graph_type`: A keyword argument that specifies 
                the underlying representation used by the GNNGraph. 
                Currently supported values are 
    - `:coo`. Graph represented as a tuple `(source, target)`, such that the `k`-th edge 
              connects the node `source[k]` to node `target[k]`.
              Optionally, also edge weights can be given: `(source, target, weights)`.
    - `:sparse`. A sparse adjacency matrix representation.
    - `:dense`. A dense adjacency matrix representation.  
    Defaults to `:coo`, currently the most supported type.
- `dir`: The assumed edge direction when given adjacency matrix or adjacency list input data `g`. 
        Possible values are `:out` and `:in`. Default `:out`.
- `num_nodes`: The number of nodes. If not specified, inferred from `g`. Default `nothing`.
- `graph_indicator`: For batched graphs, a vector containing the graph assigment of each node. Default `nothing`.  
- `ndata`: Node features. An array or named tuple of arrays whose last dimension has size `num_nodes`.
- `edata`: Edge features. An array or named tuple of arrays whose last dimension has size `num_edges`.
- `gdata`: Graph features. An array or named tuple of arrays whose last dimension has size `num_graphs`. 

# Examples 

```julia
using Flux, GraphNeuralNetworks

# Construct from adjacency list representation
data = [[2,3], [1,4,5], [1], [2,5], [2,4]]
g = GNNGraph(data)

# Number of nodes, edges, and batched graphs
g.num_nodes  # 5
g.num_edges  # 10 
g.num_graphs # 1 

# Same graph in COO representation
s = [1,1,2,2,2,3,4,4,5,5]
t = [2,3,1,4,5,3,2,5,2,4]
g = GNNGraph(s, t)

# From a Graphs' graph
g = GNNGraph(erdos_renyi(100, 20))

# Add 2 node feature arrays
g = GNNGraph(g, ndata = (x=rand(100, g.num_nodes), y=rand(g.num_nodes)))

# Add node features and edge features with default names `x` and `e` 
g = GNNGraph(g, ndata = rand(100, g.num_nodes), edata = rand(16, g.num_edges))

g.ndata.x
g.ndata.e

# Send to gpu
g = g |> gpu

# Collect edges' source and target nodes.
# Both source and target are vectors of length num_edges
source, target = edge_index(g)
```
"""
struct GNNGraph{T<:Union{COO_T,ADJMAT_T}} <: AbstractGraph{Int}
    graph::T
    num_nodes::Int
    num_edges::Int
    num_graphs::Int
    graph_indicator       # vector of ints or nothing
    ndata::NamedTuple
    edata::NamedTuple
    gdata::NamedTuple
end

@functor GNNGraph

function GNNGraph(data::D; 
                        num_nodes = nothing,
                        graph_indicator = nothing, 
                        graph_type = :coo,
                        dir = :out,
                        ndata = (;), 
                        edata = (;), 
                        gdata = (;),
                        ) where D <: Union{COO_T, ADJMAT_T, ADJLIST_T}

    @assert graph_type ∈ [:coo, :dense, :sparse] "Invalid graph_type $graph_type requested"
    @assert dir ∈ [:in, :out]
    
    if graph_type == :coo
        graph, num_nodes, num_edges = to_coo(data; num_nodes, dir)
    elseif graph_type == :dense
        graph, num_nodes, num_edges = to_dense(data; num_nodes, dir)
    elseif graph_type == :sparse
        graph, num_nodes, num_edges = to_sparse(data; num_nodes, dir)
    end
    
    num_graphs = !isnothing(graph_indicator) ? maximum(graph_indicator) : 1
    
    ndata = normalize_graphdata(ndata, default_name=:x, n=num_nodes)
    edata = normalize_graphdata(edata, default_name=:e, n=num_edges, duplicate_if_needed=true)
    gdata = normalize_graphdata(gdata, default_name=:u, n=num_graphs)
    
    GNNGraph(graph, 
            num_nodes, num_edges, num_graphs, 
            graph_indicator,
            ndata, edata, gdata)
end

function (::Type{<:GNNGraph})(num_nodes::T; kws...) where {T<:Integer}
    s, t = T[], T[] 
    return GNNGraph(s, t; num_nodes, kws...)
end

# COO convenience constructors
GNNGraph(s::AbstractVector, t::AbstractVector, v = nothing; kws...) = GNNGraph((s, t, v); kws...)
GNNGraph((s, t)::NTuple{2}; kws...) = GNNGraph((s, t, nothing); kws...)

# GNNGraph(g::AbstractGraph; kws...) = GNNGraph(adjacency_matrix(g, dir=:out); kws...)

function GNNGraph(g::AbstractGraph; kws...)
    s = Graphs.src.(Graphs.edges(g))
    t = Graphs.dst.(Graphs.edges(g))
    if !Graphs.is_directed(g) 
        # add reverse edges since GNNGraph is directed
        s, t = [s; t], [t; s]    
    end
    GNNGraph((s, t); num_nodes=Graphs.nv(g), kws...)
end


function GNNGraph(g::GNNGraph; ndata=g.ndata, edata=g.edata, gdata=g.gdata, graph_type=nothing)

    ndata = normalize_graphdata(ndata, default_name=:x, n=g.num_nodes)
    edata = normalize_graphdata(edata, default_name=:e, n=g.num_edges, duplicate_if_needed=true)
    gdata = normalize_graphdata(gdata, default_name=:u, n=g.num_graphs)

    if !isnothing(graph_type)
        if graph_type == :coo
            graph, num_nodes, num_edges = to_coo(g.graph; g.num_nodes)
        elseif graph_type == :dense
            graph, num_nodes, num_edges = to_dense(g.graph; g.num_nodes)
        elseif graph_type == :sparse
            graph, num_nodes, num_edges = to_sparse(g.graph; g.num_nodes)
        end
        @assert num_nodes == g.num_nodes
        @assert num_edges == g.num_edges
    else
        graph = g.graph
    end
    GNNGraph(graph, 
            g.num_nodes, g.num_edges, g.num_graphs, 
            g.graph_indicator, 
            ndata, edata, gdata) 
end

function Base.show(io::IO, g::GNNGraph)
    print(io, "GNNGraph:
    num_nodes = $(g.num_nodes)
    num_edges = $(g.num_edges)")
    g.num_graphs > 1 && print("\n    num_graphs = $(g.num_graphs)")
    if !isempty(g.ndata)
        print(io, "\n    ndata:")
        for k in keys(g.ndata)
            print(io, "\n        $k => $(size(g.ndata[k]))")
        end
    end
    if !isempty(g.edata)
        print(io, "\n    edata:")
        for k in keys(g.edata)
            print(io, "\n        $k => $(size(g.edata[k]))")
        end
    end
    if !isempty(g.gdata)
        print(io, "\n    gdata:")
        for k in keys(g.gdata)
            print(io, "\n        $k => $(size(g.gdata[k]))")
        end
    end
end

### StatsBase/LearnBase compatibility
StatsBase.nobs(g::GNNGraph) = g.num_graphs 
LearnBase.getobs(g::GNNGraph, i) = getgraph(g, i)

# Flux's Dataloader compatibility. Related PR https://github.com/FluxML/Flux.jl/pull/1683
Flux.Data._nobs(g::GNNGraph) = g.num_graphs
Flux.Data._getobs(g::GNNGraph, i) = getgraph(g, i)

#########################
Base.:(==)(g1::GNNGraph, g2::GNNGraph) = all(k -> getfield(g1,k)==getfield(g2,k), fieldnames(typeof(g1)))
