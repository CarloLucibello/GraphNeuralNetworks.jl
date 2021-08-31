"""
    bypass_graph(nf_func, ef_func, gf_func)

Bypassing graph in GNNGraph and let other layer process (node, edge and global)features only.
"""
function bypass_graph(nf_func=identity, ef_func=identity, gf_func=identity)
    return function (g::GNNGraph)
        GNNGraph(g,
                      nf=nf_func(node_feature(g)),
                      ef=ef_func(edge_feature(g)),
                      gf=gf_func(global_feature(g)))
    end
end
