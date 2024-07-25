# 2 or more args graph operators
""""
    intersect(g1::GNNGraph, g2::GNNGraph)

Intersect two graphs by keeping only the common edges.
"""
function Base.intersect(g1::GNNGraph, g2::GNNGraph)
    @assert g1.num_nodes == g2.num_nodes
    @assert graph_type_symbol(g1) == graph_type_symbol(g2)
    graph_type = graph_type_symbol(g1)
    num_nodes = g1.num_nodes

    idx1, _ = edge_encoding(edge_index(g1)..., num_nodes)
    idx2, _ = edge_encoding(edge_index(g2)..., num_nodes)
    idx = intersect(idx1, idx2)
    s, t = edge_decoding(idx, num_nodes)
    return GNNGraph(s, t; num_nodes, graph_type)
end
