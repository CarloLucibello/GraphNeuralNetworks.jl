
const EDict{T} = Dict{Tuple{Symbol, Symbol, Symbol}, T}
const NDict{T} = Dict{Symbol, T}


"""
    GNNHeteroGraph(data; ndata, edata, gdata, num_nodes, graph_indicator, dir])
    
A type representing a heterogeneus graph structure.
it is similar [`GNNGraph`](@ref) but node and edges are of different types.


# Arguments 

- `data`: Some data representing the graph topology. Possible type are 
    - A tuple containing the source and target vectors (COO representation)
- `num_nodes`: The number of nodes. If not specified, inferred from `g`. Default `nothing`.
- `graph_indicator`: For batched graphs, a vector containing the graph assigment of each node. Default `nothing`.  
- `ndata`: Node features. An array or named tuple of arrays whose last dimension has size `num_nodes`.
- `edata`: Edge features. A dictionary-like An array or named tuple of arrays whose last dimension has size `num_edges`.
- `gdata`: Graph features. An array or named tuple of arrays whose last dimension has size `num_graphs`. 


!!! note
    `GNNHeteroGraph` is still experimental and not fully supported.
    The interface could be subject to change in the future.
    
# Examples 

```julia
julia> using Flux, GraphNeuralNetworks

julia> nA, nB = 10, 20
(10, 20)

julia> edges1 = rand(1:nA, 20), rand(1:nB, 20)
([4, 8, 6, 3, 4, 7, 2, 7, 3, 2, 3, 4, 9, 4, 2, 9, 10, 1, 3, 9], [6, 4, 20, 8, 16, 7, 12, 16, 5, 4, 6, 20, 11, 19, 17, 9, 12, 2, 18, 12])

julia> edges2 = rand(1:nB, 30), rand(1:nA, 30)
([17, 5, 2, 4, 5, 3, 8, 7, 9, 7  …  19, 8, 20, 7, 16, 2, 9, 15, 8, 13], [1, 1, 3, 1, 1, 3, 2, 7, 4, 4  …  7, 10, 6, 3, 4, 9, 1, 5, 8, 5])

julia> hg = GNNHeteroGraph(((:A, :rel1, :B) => edges1, (:B, :rel2, :A) => edges2))
GNNHeteroGraph:
  num_nodes: (:A => 10, :B => 20)         
  num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)
```

"""
struct GNNHeteroGraph
    graph::EDict
    num_nodes::NDict{Int}
    num_edges::EDict{Int}
    num_graphs::Int
    graph_indicator::Union{Nothing, NDict}
    ndata::NDict{<:NamedTuple}
    edata::EDict{<:NamedTuple}
    gdata::NamedTuple
    ntypes::Vector{Symbol}
    etypes::Vector{Symbol}
end

@functor GNNHeteroGraph

function GNNHeteroGraph(data::EDict; 
                        num_nodes = nothing,
                        graph_indicator = nothing, 
                        graph_type = :coo,
                        dir = :out,
                        ndata = NDict{NamedTuple}(), 
                        edata = EDict{NamedTuple}(), 
                        gdata = (;))
                        

    @assert graph_type ∈ [:coo, :dense, :sparse] "Invalid graph_type $graph_type requested"
    @assert dir ∈ [:in, :out]
    @assert graph_type == :coo "only :coo graph_type is supported for now"

    if num_nodes !== nothing
        num_nodes = Dict(num_nodes)
    end

    ntypes = union([[k[1] for k in keys(data)]; [k[3] for k in keys(data)]])
    etypes = [k[2] for k in keys(data)]

    if graph_type == :coo
        graph, num_nodes, num_edges = to_coo(data; num_nodes, dir)
    elseif graph_type == :dense
        graph, num_nodes, num_edges = to_dense(data; num_nodes, dir)
    elseif graph_type == :sparse
        graph, num_nodes, num_edges = to_sparse(data; num_nodes, dir)
    end
    
    num_graphs = !isnothing(graph_indicator) ? maximum([maximum(gi) for gi in values(graph_indicator)]) : 1
    
    ndata = normalize_heterographdata(ndata, default_name=:x, n=num_nodes)
    edata = normalize_heterographdata(edata, default_name=:e, n=num_edges, duplicate_if_needed=true)
    gdata = normalize_graphdata(gdata, default_name=:u, n=num_graphs)
    
    return GNNHeteroGraph(graph, 
                num_nodes, num_edges, num_graphs, 
                graph_indicator,
                ndata, edata, gdata,
                ntypes, etypes)
end


function Base.show(io::IO, g::GNNHeteroGraph)
    print(io, "GNNHeteroGraph($(g.num_nodes), $(g.num_edges))")
end

function Base.show(io::IO, ::MIME"text/plain", g::GNNHeteroGraph)
    if get(io, :compact, false)
        print(io, "GNNHeteroGraph($(g.num_nodes), $(g.num_edges))")
    else # if the following block is indented the printing is ruined
    print(io, "GNNHeteroGraph:
  num_nodes: $((g.num_nodes...,))         
  num_edges: $((g.num_edges...,))")
  g.num_graphs > 1 && print(io, "\n    num_graphs = $(g.num_graphs)")
  if !isempty(g.ndata)
      print(io, "\n  ndata:")
      for k in keys(g.ndata)
          print(io, "\n    ", _str(k), "  =>  $(shortsummary(g.ndata[k]))")
      end
  end
  if !isempty(g.edata)
      print(io, "\n  edata:")
      for k in keys(g.edata)
          print(io, "\n    $k  =>  $(shortsummary(g.edata[k]))")
      end
  end
  if !isempty(g.gdata)
      print(io, "\n  gdata:")
      print(io, "\n    ")
      shortsummary(io, g.gdata)
  end #else
  end
end

GNNHeteroGraph(data; kws...) = GNNHeteroGraph(Dict(data); kws...)

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

_str(s::Symbol) = ":$s"
_str(s) = "$s"

MLUtils.numobs(g::GNNHeteroGraph) = g.num_graphs 
MLUtils.getobs(g::GNNHeteroGraph, i) = getgraph(g, i)

