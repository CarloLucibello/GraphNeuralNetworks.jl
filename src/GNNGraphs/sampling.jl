"""
    sample_neighbors(g, nodes, K=-1; dir=:in)

Sample neighboring edges of the given nodes and return the induced subgraph.
For each node, a number of inbound (or outbound when `dir = :out``) edges will be randomly chosen. 
The graph returned will then contain all the nodes in the original graph, but only the sampled edges.

# Arguments

- `g`. The graph.
- `nodes`. A list of node IDs to sample neighbors from.
- `K`. The maximum number of edges to be sampled for each node.
         If -1, all the neighboring edges will be selected.
- `dir`. Determines whether to sample inbound (`:in`) or outbound (``:out`) edges (Default `:in`).
- `replace`. If `true`, sample with replacement.
"""
using StatsBase: sample

function sample_neighbors(g::GNNGraph{<:COO_T}, nodes, K=-1; dir=:in; replace=true)
    @assert dir ∈ (:in, :out)
    _, eidlist = adjacency_list(g, nodes; dir, with_eid=true)
    for i in 1:length(eidlist)
        if replace 
            k = K > 0 ? K : length(eidlist[i])
        else
            k = K > 0 ? min(length(eidlist[i]), K) : length(eidlist[i])
        end
        eidlist[i] = sample(eidlist[i], k; replace)
    end
    eids = reduce(vcat, eidlist)
    s, t = edge_index(g)
    w = get_edge_weight(g)
    s = s[eids]
    t = t[eids]
    w = isnothing(w) ? nothing : w[eids]
    graph = (s, t, w)
    
    edata = getobs(g.edata, eids)
    
    num_edges = length(eids)
    
    gnew = GNNGraph(graph, 
                g.num_nodes, num_edges, g.num_graphs,
                g.graph_indicator,
                g.ndata, edata, g.gdata)
    return gnew
end
