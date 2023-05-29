
const EDict{T} = Dict{Tuple{Symbol, Symbol, Symbol}, T}
const NDict{T} = Dict{Symbol, T}

"""
    GNNHeteroGraph(data; ndata, edata, gdata, num_nodes, graph_indicator, dir])

A type representing a heterogeneous graph structure.
It is similar to [`GNNGraph`](@ref) but nodes and edges are of different types.

# Arguments

- `data`: A dictionary or an iterable object that maps (source_type, edge_type, target_type)
    triples to (source, target) index vectors.
- `num_nodes`: The number of nodes for each type. If not specified, inferred from `g`. Default `nothing`.
- `graph_indicator`: For batched graphs, a dictionary of vectors containing the graph assignment of each node. Default `nothing`.
- `ndata`: Node features. A dictionary of arrays or named tuple of arrays.
           The size of the last dimension of each array must be given by `g.num_nodes`.
- `edata`: Edge features. A dictionary of arrays or named tuple of arrays.
           The size of the last dimension of each array must be given by `g.num_edges`.
- `gdata`: Graph features. An array or named tuple of arrays whose last dimension has size `num_graphs`.


!!! warning
    `GNNHeteroGraph` is still experimental and not fully supported.
    The interface could be subject to change in the future.

# Examples

```julia
julia> using Flux, GraphNeuralNetworks

julia> num_nodes = Dict(:A => 10, :B => 20);

julia> edges1 = rand(1:num_nodes[:A], 20), rand(1:num_nodes[:B], 20)
([4, 8, 6, 3, 4, 7, 2, 7, 3, 2, 3, 4, 9, 4, 2, 9, 10, 1, 3, 9], [6, 4, 20, 8, 16, 7, 12, 16, 5, 4, 6, 20, 11, 19, 17, 9, 12, 2, 18, 12])

julia> edges2 = rand(1:num_nodes[:B], 30), rand(1:num_nodes[:A], 30)
([17, 5, 2, 4, 5, 3, 8, 7, 9, 7  …  19, 8, 20, 7, 16, 2, 9, 15, 8, 13], [1, 1, 3, 1, 1, 3, 2, 7, 4, 4  …  7, 10, 6, 3, 4, 9, 1, 5, 8, 5])

julia> eindex = ((:A, :rel1, :B) => edges1, (:B, :rel2, :A) => edges2);

julia> hg = GNNHeteroGraph(eindex; num_nodes)
GNNHeteroGraph:
  num_nodes: (:A => 10, :B => 20)
  num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)

julia> hg.num_edges
Dict{Tuple{Symbol, Symbol, Symbol}, Int64} with 2 entries:
(:A, :rel1, :B) => 20
(:B, :rel2, :A) => 30

# Let's add some node features
julia> ndata = Dict(:A => (x = rand(2, num_nodes[:A]), y = rand(3, num_nodes[:A])),
                    :B => rand(10, num_nodes[:B]));

julia> hg = GNNHeteroGraph(eindex; num_nodes, ndata)
GNNHeteroGraph:
    num_nodes: (:A => 10, :B => 20)
    num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)
    ndata:
    :A  =>  (x = 2×10 Matrix{Float64}, y = 3×10 Matrix{Float64})
    :B  =>  x = 10×20 Matrix{Float64}

# Access features of nodes of type :A
julia> hg.ndata[:A].x
2×10 Matrix{Float64}:
    0.825882  0.0797502  0.245813  0.142281  0.231253  0.685025  0.821457  0.888838  0.571347   0.53165
    0.631286  0.316292   0.705325  0.239211  0.533007  0.249233  0.473736  0.595475  0.0623298  0.159307
```

See also [`GNNGraph`](@ref) for a homogeneous graph type and [`rand_heterograph`](@ref) for a function to generate random heterographs.
"""
struct GNNHeteroGraph
    graph::EDict
    num_nodes::NDict{Int}
    num_edges::EDict{Int}
    num_graphs::Int
    graph_indicator::Union{Nothing, NDict}
    ndata::NDict{DataStore}
    edata::EDict{DataStore}
    gdata::DataStore
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
    @assert graph_type==:coo "only :coo graph_type is supported for now"

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

    num_graphs = !isnothing(graph_indicator) ?
                 maximum([maximum(gi) for gi in values(graph_indicator)]) : 1

    ndata = normalize_heterographdata(ndata, default_name = :x, n = num_nodes)
    edata = normalize_heterographdata(edata, default_name = :e, n = num_edges,
                                      duplicate_if_needed = true)
    gdata = normalize_graphdata(gdata, default_name = :u, n = num_graphs)

    return GNNHeteroGraph(graph,
                          num_nodes, num_edges, num_graphs,
                          graph_indicator,
                          ndata, edata, gdata,
                          ntypes, etypes)
end

function show_sorted_Dict(io::IO, d::Dict, compact::Bool)
    if compact
        print(io, "Dict")
    end
    print(io, "(")
    if !isempty(d)
        if length(keys(d)) == 1
            show(io, keys[1])
            print(io, " => $(d[keys[1]])")
        else
            sorted_keys = sort!(collect(keys(d)))
            for key in sorted_keys[1:(end - 1)]
                show(io, key)
                print(io, " => $(d[key]), ")
            end
            show(io, sorted_keys[end])
            print(io, " => $(d[sorted_keys[end]])")
        end
    end
    print(io, ")")
end

function Base.show(io::IO, g::GNNHeteroGraph)
    print(io, "GNNHeteroGraph(")
    show_sorted_Dict(io, g.num_nodes, true)
    print(io, ", ")
    show_sorted_Dict(io, g.num_edges, true)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", g::GNNHeteroGraph)
    if get(io, :compact, false)
        print(io, "GNNHeteroGraph(")
        show_sorted_Dict(io, g.num_nodes, true)
        print(io, ", ")
        show_sorted_Dict(io, g.num_edges, true)
        print(io, ")")
    else
        print(io, "GNNHeteroGraph:\n num_nodes: ")
        show_sorted_Dict(io, g.num_nodes, false)
        print(io, "\n num_edges: ")
        show_sorted_Dict(io, g.num_edges, false)
        g.num_graphs > 1 && print(io, "\n num_graphs: $(g.num_graphs)")
        if !isempty(g.ndata)
            print(io, "\n ndata:")
            for k in sort(collect(keys(g.ndata)))
                print(io, "\n\t", _str(k), "  =>  $(shortsummary(g.ndata[k]))")
            end
        end
        if !isempty(g.edata)
            print(io, "\n edata:")
            for k in sort(collect(keys(g.edata)))
                print(io, "\n\t$k  =>  $(shortsummary(g.edata[k]))")
            end
        end
        if !isempty(g.gdata)
            print(io, "\n gdata:\n\t")
            shortsummary(io, g.gdata)
        end
    end
end

GNNHeteroGraph(data; kws...) = GNNHeteroGraph(Dict(data); kws...)

_str(s::Symbol) = ":$s"
_str(s) = "$s"

MLUtils.numobs(g::GNNHeteroGraph) = g.num_graphs
# MLUtils.getobs(g::GNNHeteroGraph, i) = getgraph(g, i)
