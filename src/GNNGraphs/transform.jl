
"""
    add_self_loops(g::GNNGraph)

Return a graph with the same features as `g`
but also adding edges connecting the nodes to themselves.

Nodes with already existing
self-loops will obtain a second self-loop.
"""
function add_self_loops(g::GNNGraph{<:COO_T})
    s, t = edge_index(g)
    @assert g.edata === (;)
    @assert edge_weight(g) === nothing
    n = g.num_nodes
    nodes = convert(typeof(s), [1:n;])
    s = [s; nodes]
    t = [t; nodes]

    GNNGraph((s, t, nothing), 
        g.num_nodes, length(s), g.num_graphs, 
        g.graph_indicator,
        g.ndata, g.edata, g.gdata)
end

function add_self_loops(g::GNNGraph{<:ADJMAT_T})
    A = g.graph
    @assert g.edata === (;)
    num_edges = g.num_edges + g.num_nodes
    A = A + I
    GNNGraph(A, 
            g.num_nodes, num_edges, g.num_graphs, 
            g.graph_indicator,
            g.ndata, g.edata, g.gdata)
end


function remove_self_loops(g::GNNGraph{<:COO_T})
    s, t = edge_index(g)
    # TODO remove these constraints
    @assert g.edata === (;)
    @assert edge_weight(g) === nothing
    
    mask_old_loops = s .!= t
    s = s[mask_old_loops]
    t = t[mask_old_loops]

    GNNGraph((s, t, nothing), 
            g.num_nodes, length(s), g.num_graphs, 
            g.graph_indicator,
            g.ndata, g.edata, g.gdata)
end

"""
    add_edges(g::GNNGraph, s::AbstractVector, t::AbstractVector; [edata])

Add to graph `g` the edges with source nodes `s` and target nodes `t`.
"""
function add_edges(g::GNNGraph{<:COO_T}, 
        snew::AbstractVector{<:Integer}, 
        tnew::AbstractVector{<:Integer};
        edata=nothing)

    @assert length(snew) == length(tnew)
    # TODO remove this constraint
    @assert edge_weight(g) === nothing
    
    edata = normalize_graphdata(edata, default_name=:e, n=length(snew))
    edata = cat_features(g.edata, edata)
    
    s, t = edge_index(g)
    s = [s; snew]
    t = [t; tnew]

    GNNGraph((s, t, nothing), 
            g.num_nodes, length(s), g.num_graphs, 
            g.graph_indicator,
            g.ndata, edata, g.gdata)
end


"""
    add_nodes(g::GNNGraph, n; [ndata])

Add `n` new nodes to graph `g`. In the 
new graph, these nodes will have indexes from `g.num_nodes + 1`
to `g.num_nodes + n`.
"""
function add_nodes(g::GNNGraph{<:COO_T}, n::Integer; ndata=(;))
    ndata = normalize_graphdata(ndata, default_name=:x, n=n)
    ndata = cat_features(g.ndata, ndata)

    GNNGraph(g.graph, 
            g.num_nodes + n, g.num_edges, g.num_graphs, 
            g.graph_indicator,
            ndata, g.edata, g.gdata)
end


function SparseArrays.blockdiag(g1::GNNGraph, g2::GNNGraph)
    nv1, nv2 = g1.num_nodes, g2.num_nodes
    if g1.graph isa COO_T
        s1, t1 = edge_index(g1)
        s2, t2 = edge_index(g2)
        s = vcat(s1, nv1 .+ s2)
        t = vcat(t1, nv1 .+ t2)
        w = cat_features(edge_weight(g1), edge_weight(g2))
        graph = (s, t, w)
        ind1 = isnothing(g1.graph_indicator) ? ones_like(s1, Int, nv1) : g1.graph_indicator 
        ind2 = isnothing(g2.graph_indicator) ? ones_like(s2, Int, nv2) : g2.graph_indicator     
    elseif g1.graph isa ADJMAT_T        
        graph = blockdiag(g1.graph, g2.graph)
        ind1 = isnothing(g1.graph_indicator) ? ones_like(graph, Int, nv1) : g1.graph_indicator 
        ind2 = isnothing(g2.graph_indicator) ? ones_like(graph, Int, nv2) : g2.graph_indicator     
    end
    graph_indicator = vcat(ind1, g1.num_graphs .+ ind2)
    
    GNNGraph(graph,
            nv1 + nv2, g1.num_edges + g2.num_edges, g1.num_graphs + g2.num_graphs, 
            graph_indicator,
            cat_features(g1.ndata, g2.ndata),
            cat_features(g1.edata, g2.edata),
            cat_features(g1.gdata, g2.gdata))
end

# PIRACY
function SparseArrays.blockdiag(A1::AbstractMatrix, A2::AbstractMatrix)
    m1, n1 = size(A1)
    @assert m1 == n1
    m2, n2 = size(A2)
    @assert m2 == n2
    O1 = fill!(similar(A1, eltype(A1), (m1, n2)), 0)
    O2 = fill!(similar(A1, eltype(A1), (m2, n1)), 0)
    return [A1 O1
            O2 A2]
end

"""
    blockdiag(xs::GNNGraph...)

Equivalent to [`Flux.batch`](@ref).
"""
function SparseArrays.blockdiag(g1::GNNGraph, gothers::GNNGraph...)
    g = g1
    for go in gothers
        g = blockdiag(g, go)
    end
    return g
end

"""
    batch(gs::Vector{<:GNNGraph})

Batch together multiple `GNNGraph`s into a single one 
containing the total number of original nodes and edges.

Equivalent to [`SparseArrays.blockdiag`](@ref).
See also [`Flux.unbatch`](@ref).

# Usage

```juliarepl
julia> g1 = rand_graph(4, 6, ndata=ones(8, 4))
GNNGraph:
    num_nodes = 4
    num_edges = 6
    num_graphs = 1
    ndata:
        x => (8, 4)
    edata:
    gdata:


julia> g2 = rand_graph(7, 4, ndata=zeros(8, 7))
GNNGraph:
    num_nodes = 7
    num_edges = 4
    num_graphs = 1
    ndata:
        x => (8, 7)
    edata:
    gdata:


julia> g12 = Flux.batch([g1, g2])
GNNGraph:
    num_nodes = 11
    num_edges = 10
    num_graphs = 2
    ndata:
        x => (8, 11)
    edata:
    gdata:


julia> g12.ndata.x
8×11 Matrix{Float64}:
 1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
Flux.batch(gs::Vector{<:GNNGraph}) = blockdiag(gs...)


"""
    unbatch(g::GNNGraph)

Opposite of the [`Flux.batch`](@ref) operation, returns 
an array of the individual graphs batched together in `g`.

See also [`Flux.batch`](@ref) and [`getgraph`](@ref).

# Usage

```juliarepl
julia> gbatched = Flux.batch([rand_graph(5, 6), rand_graph(10, 8), rand_graph(4,2)])
GNNGraph:
    num_nodes = 19
    num_edges = 16
    num_graphs = 3
    ndata:
    edata:
    gdata:

julia> Flux.unbatch(gbatched)
3-element Vector{GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}}:
 GNNGraph:
    num_nodes = 5
    num_edges = 6
    num_graphs = 1
    ndata:
    edata:
    gdata:

 GNNGraph:
    num_nodes = 10
    num_edges = 8
    num_graphs = 1
    ndata:
    edata:
    gdata:

 GNNGraph:
    num_nodes = 4
    num_edges = 2
    num_graphs = 1
    ndata:
    edata:
    gdata:
```
"""
function Flux.unbatch(g::GNNGraph) 
    [getgraph(g, i) for i in 1:g.num_graphs]
end


"""
    getgraph(g::GNNGraph, i; nmap=false)

Return the subgraph of `g` induced by those nodes `j`
for which `g.graph_indicator[j] == i` or,
if `i` is a collection, `g.graph_indicator[j] ∈ i`. 
In other words, it extract the component graphs from a batched graph. 

If `nmap=true`, return also a vector `v` mapping the new nodes to the old ones. 
The node `i` in the subgraph will correspond to the node `v[i]` in `g`.
"""
getgraph(g::GNNGraph, i::Int; kws...) = getgraph(g, [i]; kws...)

function getgraph(g::GNNGraph, i::AbstractVector{Int}; nmap=false)
    if g.graph_indicator === nothing
        @assert i == [1]
        if nmap
            return g, 1:g.num_nodes
        else
            return g
        end
    end

    node_mask = g.graph_indicator .∈ Ref(i)
    
    nodes = (1:g.num_nodes)[node_mask]
    nodemap = Dict(v => vnew for (vnew, v) in enumerate(nodes))

    graphmap = Dict(i => inew for (inew, i) in enumerate(i))
    graph_indicator = [graphmap[i] for i in g.graph_indicator[node_mask]]
    
    s, t = edge_index(g)
    w = edge_weight(g)
    edge_mask = s .∈ Ref(nodes) 
    
    if g.graph isa COO_T 
        s = [nodemap[i] for i in s[edge_mask]]
        t = [nodemap[i] for i in t[edge_mask]]
        w = isnothing(w) ? nothing : w[edge_mask]
        graph = (s, t, w)
    elseif g.graph isa ADJMAT_T
        graph = g.graph[nodes, nodes]
    end

    ndata = getobs(g.ndata, node_mask)
    edata = getobs(g.edata, edge_mask)
    gdata = getobs(g.gdata, i)
    
    num_edges = sum(edge_mask)
    num_nodes = length(graph_indicator)
    num_graphs = length(i)

    gnew = GNNGraph(graph, 
                num_nodes, num_edges, num_graphs,
                graph_indicator,
                ndata, edata, gdata)

    if nmap
        return gnew, nodes
    else
        return gnew
    end
end

@non_differentiable add_self_loops(x...)     # TODO this is wrong, since g carries feature arrays, needs rrule
@non_differentiable remove_self_loops(x...)  # TODO this is wrong, since g carries feature arrays, needs rrule
