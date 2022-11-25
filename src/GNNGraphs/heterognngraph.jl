
const EDict{T} = Dict{Tuple{String, String, String}, T}
const NDict{T} = Dict{String, T}

struct HeteroGNNGraph
    graph::EDict
    num_nodes::NDict{Int}
    num_edges::EDict{Int}
    num_graphs::Int
    graph_indicator::Union{Nothing, NDict}
    ndata::NDict{NamedTuple}
    edata::EDict{NamedTuple}
    gdata::NamedTuple
    ntypes::Vector{String}
    etypes::Vector{String}
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
    @assert length(union(etypes)) == length(etypes)

    if graph_type == :coo
        graph, num_nodes, num_edges = to_coo(data; num_nodes, dir)
    elseif graph_type == :dense
        graph, num_nodes, num_edges = to_dense(data; num_nodes, dir)
    elseif graph_type == :sparse
        graph, num_nodes, num_edges = to_sparse(data; num_nodes, dir)
    end
    
    num_graphs = !isnothing(graph_indicator) ? maximum([maximum(gi) for gi in values(graph_indicator)]) : 1
    
    # ndata = normalize_graphdata(ndata, default_name=:x, n=num_nodes)
    # edata = normalize_graphdata(edata, default_name=:e, n=num_edges, duplicate_if_needed=true)
    # gdata = normalize_graphdata(gdata, default_name=:u, n=num_graphs)
    
    HeteroGNNGraph(graph, 
            num_nodes, num_edges, num_graphs, 
            graph_indicator,
            ndata, edata, gdata,
            ntypes, etypes)
end