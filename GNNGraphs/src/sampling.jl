"""
    sample_neighbors(g, nodes, K=-1; dir=:in, replace=false, dropnodes=false)

Sample neighboring edges of the given nodes and return the induced subgraph.
For each node, a number of inbound (or outbound when `dir = :out``) edges will be randomly chosen. 
If `dropnodes=false`, the graph returned will then contain all the nodes in the original graph, 
but only the sampled edges.

The returned graph will contain an edge feature `EID` corresponding to the id of the edge
in the original graph. If `dropnodes=true`, it will also contain a node feature `NID` with
the node ids in the original graph.

# Arguments

- `g`. The graph.
- `nodes`. A list of node IDs to sample neighbors from.
- `K`. The maximum number of edges to be sampled for each node.
       If -1, all the neighboring edges will be selected.
- `dir`. Determines whether to sample inbound (`:in`) or outbound (``:out`) edges (Default `:in`).
- `replace`. If `true`, sample with replacement.
- `dropnodes`. If `true`, the resulting subgraph will contain only the nodes involved in the sampled edges.
     
# Examples

```julia
julia> g = rand_graph(20, 100)
GNNGraph:
    num_nodes = 20
    num_edges = 100

julia> sample_neighbors(g, 2:3)
GNNGraph:
    num_nodes = 20
    num_edges = 9
    edata:
        EID => (9,)

julia> sg = sample_neighbors(g, 2:3, dropnodes=true)
GNNGraph:
    num_nodes = 10
    num_edges = 9
    ndata:
        NID => (10,)
    edata:
        EID => (9,)

julia> sg.ndata.NID
10-element Vector{Int64}:
  2
  3
 17
 14
 18
 15
 16
 20
  7
 10

julia> sample_neighbors(g, 2:3, 5, replace=true)
GNNGraph:
    num_nodes = 20
    num_edges = 10
    edata:
        EID => (10,)
```
"""
function sample_neighbors(g::GNNGraph{<:COO_T}, nodes, K = -1;
                          dir = :in, replace = false, dropnodes = false)
    @assert dir ∈ (:in, :out)
    _, eidlist = adjacency_list(g, nodes; dir, with_eid = true)
    for i in 1:length(eidlist)
        if replace
            k = K > 0 ? K : length(eidlist[i])
        else
            k = K > 0 ? min(length(eidlist[i]), K) : length(eidlist[i])
        end
        eidlist[i] = StatsBase.sample(eidlist[i], k; replace)
    end
    eids = reduce(vcat, eidlist)
    s, t = edge_index(g)
    w = get_edge_weight(g)
    s = s[eids]
    t = t[eids]
    w = isnothing(w) ? nothing : w[eids]

    edata = getobs(g.edata, eids)
    edata.EID = eids

    num_edges = length(eids)

    if !dropnodes
        graph = (s, t, w)

        gnew = GNNGraph(graph,
                        g.num_nodes, num_edges, g.num_graphs,
                        g.graph_indicator,
                        g.ndata, edata, g.gdata)
    else
        nodes_other = dir == :in ? setdiff(s, nodes) : setdiff(t, nodes)
        nodes_all = [nodes; nodes_other]
        nodemap = Dict(n => i for (i, n) in enumerate(nodes_all))
        s = [nodemap[s] for s in s]
        t = [nodemap[t] for t in t]
        graph = (s, t, w)
        graph_indicator = g.graph_indicator !== nothing ? g.graph_indicator[nodes_all] :
                          nothing
        num_nodes = length(nodes_all)
        ndata = getobs(g.ndata, nodes_all)
        ndata.NID = nodes_all

        gnew = GNNGraph(graph,
                        num_nodes, num_edges, g.num_graphs,
                        graph_indicator,
                        ndata, edata, g.gdata)
    end
    return gnew
end


"""
    induced_subgraph(graph, nodes)

Generates a subgraph from the original graph using the provided `nodes`. 
The function includes the nodes' neighbors and creates edges between nodes that are connected in the original graph. 
If a node has no neighbors, an isolated node will be added to the subgraph. 
Returns A new `GNNGraph` containing the subgraph with the specified nodes and their features.

# Arguments

- `graph`. The original GNNGraph containing nodes, edges, and node features.
- `nodes``. A vector of node indices to include in the subgraph.
     
# Examples

```julia
julia> s = [1, 2]
2-element Vector{Int64}:
 1
 2

julia> t = [2, 3]
2-element Vector{Int64}:
 2
 3

julia> graph = GNNGraph((s, t), ndata = (; x=rand(Float32, 32, 3), y=rand(Float32, 3)), edata = rand(Float32, 2))
GNNGraph:
  num_nodes: 3
  num_edges: 2
  ndata:
        y = 3-element Vector{Float32}
        x = 32×3 Matrix{Float32}
  edata:
        e = 2-element Vector{Float32}

julia> nodes = [1, 2]
2-element Vector{Int64}:
 1
 2

julia> subgraph = Graphs.induced_subgraph(graph, nodes)
GNNGraph:
  num_nodes: 2
  num_edges: 1
  ndata:
        y = 2-element Vector{Float32}
        x = 32×2 Matrix{Float32}
  edata:
        e = 1-element Vector{Float32}
```
"""
function Graphs.induced_subgraph(graph::GNNGraph, nodes::Vector{Int})
    if isempty(nodes)
        return GNNGraph()  # Return empty graph if no nodes are provided
    end

    node_map = Dict(node => i for (i, node) in enumerate(nodes))

    # Collect edges to add
    source = Int[]
    target = Int[]
    eindices = Int[]
    for node in nodes
        neighbors = Graphs.neighbors(graph, node, dir = :in)
        for neighbor in neighbors
            if neighbor in keys(node_map)
                push!(target, node_map[node])
                push!(source, node_map[neighbor])

                eindex = findfirst(x -> x == [neighbor, node], edge_index(graph))
                push!(eindices, eindex)
            end
        end
    end

    # Extract features for the new nodes
    new_ndata = getobs(graph.ndata, nodes)
    new_edata = getobs(graph.edata, eindices)

    return GNNGraph(source, target, num_nodes = length(node_map), ndata = new_ndata, edata = new_edata) 
end
