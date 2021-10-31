"""
    rand_graph(n, m; directed=false, kws...)

Generate a random (Erdós-Renyi) `GNNGraph` with `n` nodes.

If `directed=false` the output will contain `2m` edges:
the reverse edge of each edge will be present.
If `directed=true` instead, `m` unrelated edges are generated.

Additional keyword argument  will be fed to the [`GNNGraph`](@ref) constructor.
"""
function rand_graph(n::Integer, m::Integer; directed=false, kws...)
    return GNNGraph(Graphs.erdos_renyi(n, m, is_directed=directed); kws...)    
end
