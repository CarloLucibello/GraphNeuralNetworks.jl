"""
    sample_neighbors(g, nodes, fanout=-1; dir=:in)

Sample neighboring edges of the given nodes and return the induced subgraph.
For each node, a number of inbound (or outbound when `dir = :out``) edges will be randomly chosen. 
The graph returned will then contain all the nodes in the original graph, but only the sampled edges.

# Arguments

- `g`. The graph.
- `nodes`. A list of node IDs to sample neighbors from.
- `fan`. The maximum number of edges to be sampled for each node.
         If -1, all the neighboring edges will be selected.
- `dir`. Determines whether to sample inbound (`:in`) or outbound (``:out`) edges (Default `:in`).
- `replace`. If `true`, sample with replacement.
"""
function sample_neighbors(g::GNNGraph{<:COO_T}, nodes, fanout=-1; dir=:in)
    @assert dir ∈ (:in, :out)
    @assert fanout == -1
    s, t = edge_index(g)
    w = get_edge_weight(g)
    if dir == :out 
        edge_mask = s .∈ Ref(nodes) 
    else # :in
        edge_mask = t .∈ Ref(nodes) 
    end

    s = s[edge_mask]
    t = t[edge_mask]
    w = isnothing(w) ? nothing : w[edge_mask]
    graph = (s, t, w)
    
    edata = getobs(g.edata, edge_mask)
    
    num_edges = sum(edge_mask)
    
    gnew = GNNGraph(graph, 
                g.num_nodes, num_edges, g.num_graphs,
                g.graph_indicator,
                g.ndata, edata, g.gdata)
    gnew
end
