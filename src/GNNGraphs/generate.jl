"""
    rand_graph(n, m; bidirected=true, seed=-1, kws...)

Generate a random (Erdós-Renyi) `GNNGraph` with `n` nodes
and `m` edges.

If `bidirected=true` the reverse edge of each edge will be present.
If `bidirected=false` instead, `m` unrelated edges are generated.
In any case, the output graph will contain no self-loops or multi-edges.

Use a `seed > 0` for reproducibility.

Additional keyword arguments will be passed to the [`GNNGraph`](@ref) constructor.

# Usage

```juliarepl
julia> g = rand_graph(5, 4, bidirected=false)
GNNGraph:
    num_nodes = 5
    num_edges = 4

julia> edge_index(g)
([1, 3, 3, 4], [5, 4, 5, 2])

# In the bidirected case, edge data will be duplicated on the reverse edges if needed.
julia> g = rand_graph(5, 4, edata=rand(16, 2))
GNNGraph:
    num_nodes = 5
    num_edges = 4
    edata:
        e => (16, 4)

# Each edge has a reverse
julia> edge_index(g)
([1, 3, 3, 4], [3, 4, 1, 3])

```
"""
function rand_graph(n::Integer, m::Integer; bidirected=true, seed=-1, kws...)
    if bidirected
        @assert iseven(m) "Need even number of edges for bidirected graphs, given m=$m."
    end
    m2 = bidirected ? m÷2 : m
    return GNNGraph(Graphs.erdos_renyi(n, m2; is_directed=!bidirected, seed); kws...)    
end
