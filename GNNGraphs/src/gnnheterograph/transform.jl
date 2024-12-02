"""
    add_self_loops(g::GNNHeteroGraph, edge_t::EType)
    add_self_loops(g::GNNHeteroGraph)

If the source node type is the same as the destination node type in `edge_t`,
return a graph with the same features as `g` but also add self-loops 
of the specified type, `edge_t`. Otherwise, it returns `g` unchanged.

Nodes with already existing self-loops of type `edge_t` will obtain 
a second set of self-loops of the same type.

If the graph has edge weights for edges of type `edge_t`, the new edges will have weight 1.

If no edges of type `edge_t` exist, or all existing edges have no weight, 
then all new self loops will have no weight.

If `edge_t` is not passed as argument, for the entire graph self-loop is added to each node for every edge type in the graph where the source and destination node types are the same. 
This iterates over all edge types present in the graph, applying the self-loop addition logic to each applicable edge type.
"""
function add_self_loops(g::GNNHeteroGraph{<:COO_T}, edge_t::EType)

    function get_edge_weight_nullable(g::GNNHeteroGraph{<:COO_T}, edge_t::EType)
        get(g.graph, edge_t, (nothing, nothing, nothing))[3]
    end

    src_t, _, tgt_t = edge_t
    (src_t === tgt_t) ||
        return g
    
    n = get(g.num_nodes, src_t, 0)

    if haskey(g.graph, edge_t)
        s, t = g.graph[edge_t][1:2]
        nodes = convert(typeof(s), [1:n;])
        s = [s; nodes]
        t = [t; nodes]
    else
        if !isempty(g.graph)
            T = typeof(first(values(g.graph))[1])
            nodes = convert(T, [1:n;])
        else
            nodes = [1:n;]
        end
        s = nodes
        t = nodes
    end

    graph = g.graph |> copy
    ew = get(g.graph, edge_t, (nothing, nothing, nothing))[3]

    if ew !== nothing
        ew = [ew; fill!(similar(ew, n), 1)]
    end

    graph[edge_t] = (s, t, ew)
    edata = g.edata |> copy
    ndata = g.ndata |> copy
    ntypes = g.ntypes |> copy
    etypes = g.etypes |> copy
    num_nodes = g.num_nodes |> copy
    num_edges = g.num_edges |> copy
    num_edges[edge_t] = length(get(graph, edge_t, ([],[]))[1])

    return GNNHeteroGraph(graph,
             num_nodes, num_edges, g.num_graphs,
             g.graph_indicator,
             ndata, edata, g.gdata,
             ntypes, etypes)
end

function add_self_loops(g::GNNHeteroGraph)
    for edge_t in keys(g.graph)
        g = add_self_loops(g, edge_t)
    end
    return g
end

"""
    add_edges(g::GNNHeteroGraph, edge_t, s, t; [edata, num_nodes])
    add_edges(g::GNNHeteroGraph, edge_t => (s, t); [edata, num_nodes])
    add_edges(g::GNNHeteroGraph, edge_t => (s, t, w); [edata, num_nodes])

Add to heterograph `g` edges of type `edge_t` with source node vector `s` and target node vector `t`.
Optionally, pass the  edge weights `w` or the features  `edata` for the new edges.
`edge_t` is a triplet of symbols `(src_t, rel_t, dst_t)`. 

If the edge type is not already present in the graph, it is added. 
If it involves new node types, they are added to the graph as well.
In this case, a dictionary or named tuple of `num_nodes` can be passed to specify the number of nodes of the new types,
otherwise the number of nodes is inferred from the maximum node id in `s` and `t`.
"""
add_edges(g::GNNHeteroGraph{<:COO_T}, edge_t::EType, snew::AbstractVector, tnew::AbstractVector; kws...) = add_edges(g, edge_t => (snew, tnew, nothing); kws...)
add_edges(g::GNNHeteroGraph{<:COO_T}, data::Pair{EType, <:Tuple{<:AbstractVector, <:AbstractVector}}; kws...) = add_edges(g, data.first => (data.second..., nothing); kws...)

function add_edges(g::GNNHeteroGraph{<:COO_T},
                   data::Pair{EType, <:COO_T};
                   edata = nothing,
                   num_nodes = Dict{Symbol,Int}())
    edge_t, (snew, tnew, wnew) = data
    @assert length(snew) == length(tnew)
    if length(snew) == 0
        return g
    end
    @assert minimum(snew) >= 1
    @assert minimum(tnew) >= 1

    is_existing_rel = haskey(g.graph, edge_t)

    edata = normalize_graphdata(edata, default_name = :e, n = length(snew))
    _edata = g.edata |> copy
    if haskey(_edata, edge_t)
        _edata[edge_t] = cat_features(g.edata[edge_t], edata)
    else
        _edata[edge_t] = edata
    end

    graph = g.graph |> copy
    etypes = g.etypes |> copy
    ntypes = g.ntypes |> copy
    _num_nodes = g.num_nodes |> copy
    ndata = g.ndata |> copy
    if !is_existing_rel
        for (node_t, st) in [(edge_t[1], snew), (edge_t[3], tnew)]
            if node_t ∉ ntypes
                push!(ntypes, node_t)
                if haskey(num_nodes, node_t)
                    _num_nodes[node_t] = num_nodes[node_t]
                else
                    _num_nodes[node_t] = maximum(st)
                end
                ndata[node_t] = DataStore(_num_nodes[node_t])
            end
        end
        push!(etypes, edge_t)
    else
        s, t = edge_index(g, edge_t)
        snew = [s; snew]
        tnew = [t; tnew]
        w = get_edge_weight(g, edge_t)
        wnew = cat_features(w, wnew, length(s), length(snew))
    end
    
    if maximum(snew) > _num_nodes[edge_t[1]]
        ndata_new = normalize_graphdata((;), default_name = :x, n = maximum(snew) - _num_nodes[edge_t[1]])
        ndata[edge_t[1]] = cat_features(ndata[edge_t[1]], ndata_new)
        _num_nodes[edge_t[1]] = maximum(snew)
    end
    if maximum(tnew) > _num_nodes[edge_t[3]]
        ndata_new = normalize_graphdata((;), default_name = :x, n = maximum(tnew) - _num_nodes[edge_t[3]])
        ndata[edge_t[3]] = cat_features(ndata[edge_t[3]], ndata_new)
        _num_nodes[edge_t[3]] = maximum(tnew)
    end

    graph[edge_t] = (snew, tnew, wnew)
    num_edges = g.num_edges |> copy
    num_edges[edge_t] = length(graph[edge_t][1])

    return GNNHeteroGraph(graph,
             _num_nodes, num_edges, g.num_graphs,
             g.graph_indicator,
             ndata, _edata, g.gdata,
             ntypes, etypes)
end

function MLUtils.batch(gs::AbstractVector{<:GNNHeteroGraph})
    function edge_index_nullable(g::GNNHeteroGraph{<:COO_T}, edge_t::EType)
        if haskey(g.graph, edge_t)
            g.graph[edge_t][1:2]
        else
            nothing
        end
    end

    function get_edge_weight_nullable(g::GNNHeteroGraph{<:COO_T}, edge_t::EType)
        get(g.graph, edge_t, (nothing, nothing, nothing))[3]
    end

    @assert length(gs) > 0
    ntypes = union([g.ntypes for g in gs]...)
    etypes = union([g.etypes for g in gs]...)
    
    v_num_nodes = Dict(node_t => [get(g.num_nodes, node_t, 0) for g in gs] for node_t in ntypes)
    num_nodes = Dict(node_t => sum(v_num_nodes[node_t]) for node_t in ntypes)
    num_edges = Dict(edge_t => sum(get(g.num_edges, edge_t, 0) for g in gs) for edge_t in etypes)
    edge_indices = edge_indices = Dict(edge_t => [edge_index_nullable(g, edge_t) for g in gs] for edge_t in etypes)
    nodesum = Dict(node_t => cumsum([0; v_num_nodes[node_t]])[1:(end - 1)] for node_t in ntypes)
    graphs = []
    for edge_t in etypes
        src_t, _, dst_t = edge_t
        # @show edge_t edge_indices[edge_t] first(edge_indices[edge_t])
        # for ei in edge_indices[edge_t]
        #     @show ei[1]
        # end 
        # # [ei[1] for (ii, ei) in enumerate(edge_indices[edge_t])]
        s = cat_features([ei[1] .+ nodesum[src_t][ii] for (ii, ei) in enumerate(edge_indices[edge_t]) if ei !== nothing])
        t = cat_features([ei[2] .+ nodesum[dst_t][ii] for (ii, ei) in enumerate(edge_indices[edge_t]) if ei !== nothing])
        w = cat_features(filter(x -> x !== nothing, [get_edge_weight_nullable(g, edge_t) for g in gs]))
        push!(graphs, edge_t => (s, t, w))
    end
    graph = Dict(graphs...)

    #TODO relax this restriction
    @assert all(g -> g.num_graphs == 1, gs) 

    s = edge_index(gs[1], gs[1].etypes[1])[1] # grab any source vector

    function materialize_graph_indicator(g, node_t)
        n = get(g.num_nodes, node_t, 0)
        return ones_like(s, n)
    end
    v_gi = Dict(node_t => [materialize_graph_indicator(g, node_t) for g in gs] for node_t in ntypes)
    v_num_graphs = [g.num_graphs for g in gs]
    graphsum = cumsum([0; v_num_graphs])[1:(end - 1)]
    v_gi = Dict(node_t => [ng .+ gi for (ng, gi) in zip(graphsum, v_gi[node_t])] for node_t in ntypes)
    graph_indicator = Dict(node_t => cat_features(v_gi[node_t]) for node_t in ntypes)

    function data_or_else(data, types)
        Dict(type => get(data, type, DataStore(0)) for type in types)
    end

    return  GNNHeteroGraph(graph,
                num_nodes,
                num_edges,
                sum(v_num_graphs),
                graph_indicator,
                cat_features([data_or_else(g.ndata, ntypes) for g in gs]),
                cat_features([data_or_else(g.edata, etypes) for g in gs]),
                cat_features([g.gdata for g in gs]),
                ntypes, etypes)
end
