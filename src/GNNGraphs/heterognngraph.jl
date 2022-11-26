
const EDict{T} = Dict{Tuple{Symbol, Symbol, Symbol}, T}
const NDict{T} = Dict{Symbol, T}

struct HeteroGNNGraph
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

@functor HeteroGNNGraph

function HeteroGNNGraph(data::EDict; 
                        num_nodes = nothing,
                        graph_indicator = nothing, 
                        graph_type = :coo,
                        dir = :out,
                        ndata = NDict{NamedTuple}(), 
                        edata = EDict{NamedTuple}(), 
                        gdata = (;),
                        )
                        

    @assert graph_type ∈ [:coo, :dense, :sparse] "Invalid graph_type $graph_type requested"
    @assert dir ∈ [:in, :out]
    @assert graph_type == :coo "only :coo graph_type is supported for now"

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
    
    return HeteroGNNGraph(graph, 
                num_nodes, num_edges, num_graphs, 
                graph_indicator,
                ndata, edata, gdata,
                ntypes, etypes)
end


function Base.show(io::IO, g::HeteroGNNGraph)
    print(io, "HeteroGNNGraph($(g.num_nodes), $(g.num_edges))")
end

function Base.show(io::IO, ::MIME"text/plain", g::HeteroGNNGraph)
    if get(io, :compact, false)
        print(io, "HeteroGNNGraph($(g.num_nodes), $(g.num_edges))")
    else # if the following block is indented the printing is ruined
    print(io, "HeteroGNNGraph:
    num_nodes = $((g.num_nodes...,))         
    num_edges = $((g.num_edges...,))")
    g.num_graphs > 1 && print(io, "\n    num_graphs = $(g.num_graphs)")
    if !isempty(g.ndata)
        print(io, "\n    ndata:")
        for k in keys(g.ndata)
            print(io, "\n        $k => $(summary(g.ndata[k]))")
        end
    end
    if !isempty(g.edata)
        print(io, "\n    edata:")
        for k in keys(g.edata)
            print(io, "\n        $k => $(summary(g.edata[k]))")
        end
    end
    if !isempty(g.gdata)
        print(io, "\n    gdata:")
        for k in keys(g.gdata)
            print(io, "\n        $k => $(summary(g.gdata[k]))")
        end
    end
    end #else
end

MLUtils.numobs(g::HeteroGNNGraph) = g.num_graphs 
MLUtils.getobs(g::HeteroGNNGraph, i) = getgraph(g, i)

