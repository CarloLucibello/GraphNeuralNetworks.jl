struct HeteroGraph
    graph::NamedTuple
    num_nodes::NamedTuple
    num_edges::NamedTuple
    num_graphs::Int
    graph_indicator::NamedTuple  # vector of ints or nothing
    ndata::NamedTuple
    edata::NamedTuple
    gdata::NamedTuple
    ntypes
    etypes
end

@functor HeteroGraph

function HeteroGraph(data::Dict; 
                        num_nodes = nothing,
                        graph_indicator = nothing, 
                        graph_type = :coo,
                        dir = :out,
                        ndata = (;), 
                        edata = (;), 
                        gdata = (;),
                        )

    @assert graph_type ∈ [:coo, :dense, :sparse] "Invalid graph_type $graph_type requested"
    @assert dir ∈ [:in, :out]

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
    
    num_graphs = !isnothing(graph_indicator) ? maximum(graph_indicator) : 1
    
    ndata = normalize_graphdata(ndata, default_name=:x, n=num_nodes)
    edata = normalize_graphdata(edata, default_name=:e, n=num_edges, duplicate_if_needed=true)
    gdata = normalize_graphdata(gdata, default_name=:u, n=num_graphs)
    
    HeteroGraph(graph, 
            num_nodes, num_edges, num_graphs, 
            graph_indicator,
            ndata, edata, gdata,
            ntypes, etypes)
end
