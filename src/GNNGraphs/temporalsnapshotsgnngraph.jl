"""
    TemporalSnapshotsGNNGraph(snapshots::AbstractVector{<:GNNGraph})

A type representing a temporal graph as a sequence of snapshots, in this case a snapshot is a [`GNNGraph`](@ref).

It stores the feature array associated to the graph itself as a [`DataStore`](@ref) object, and it uses the [`DataStore`](@ref) objects of each snapshot for the node and edge features.
The features can be passed at construction time or added later.

# Arguments
- snapshot: a vector of snapshots, each snapshot must have the same number of nodes.
"""
struct TemporalSnapshotsGNNGraph
    num_nodes::Vector{Int}   
    num_edges::Vector{Int}
    num_snapshots::Int
    snapshots::Vector{<:GNNGraph}
    tgdata::DataStore   
end

function TemporalSnapshotsGNNGraph(snapshots::AbstractVector{<:GNNGraph})
    @assert all([s.num_nodes == snapshots[1].num_nodes for s in snapshots]) "all snapshots must have the same number of nodes"
    return TemporalSnapshotsGNNGraph(
        [s.num_nodes for s in snapshots],
        [s.num_edges for s in snapshots],
        length(snapshots),
        snapshots,
        DataStore()
    )
end

function Base.:(==)(tsg1::TemporalSnapshotsGNNGraph, tsg2::TemporalSnapshotsGNNGraph)
    tsg1 === tsg2 && return true
    for k in fieldnames(typeof(tsg1))
        getfield(tsg1, k) != getfield(tsg2, k) && return false
    end
    return true
end

function Base.getindex(tg::TemporalSnapshotsGNNGraph, t::Int)
    return tg.snapshots[t]
end

function Base.getindex(tg::TemporalSnapshotsGNNGraph, t::AbstractVector)
    return TemporalSnapshotsGNNGraph(tg.num_nodes[t], tg.num_edges[t], length(t), tg.snapshots[t], tg.tgdata)
end

function add_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)
    @assert g.num_nodes == tg.num_nodes[t] "number of nodes must match"
    num_nodes= tg.num_nodes
    num_edges = tg.num_edges
    snapshots = tg.snapshots
    num_snapshots = tg.num_snapshots + 1
    insert!(num_nodes, t, g.num_nodes)
    insert!(num_edges, t, g.num_edges)
    insert!(snapshots, t, g)
    return TemporalSnapshotsGNNGraph(num_nodes, num_edges, num_snapshots, snapshots, tg.tgdata) 
end

function remove_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int)
    num_nodes= tg.num_nodes
    num_edges = tg.num_edges
    snapshots = tg.snapshots
    num_snapshots = tg.num_snapshots - 1
    deleteat!(num_nodes, t)
    deleteat!(num_edges, t)
    deleteat!(snapshots, t)
    return TemporalSnapshotsGNNGraph(num_nodes, num_edges, num_snapshots, snapshots, tg.tgdata) 
end

function Base.show(io::IO, tsg::TemporalSnapshotsGNNGraph)
    print(io, "TemporalSnapshotsGNNGraph($(tsg.num_snapshots)) with ")
    print_feature_t(io, tsg.tgdata)
    print(io, " data")
end

function Base.show(io::IO, ::MIME"text/plain", tsg::TemporalSnapshotsGNNGraph)
    if get(io, :compact, false)
        print(io, "TemporalSnapshotsGNNGraph($(tsg.num_snapshots)) with ")
        print_feature_t(io, tsg.tgdata)
        print(io, " data")
    else
        print(io,
              "TemporalSnapshotsGNNGraph:\n  num_nodes: $(tsg.num_nodes)\n  num_edges: $(tsg.num_edges)\n  num_snapshots: $(tsg.num_snapshots)")
        if !isempty(tsg.tgdata)
            print(io, "\n  tgdata:")
            for k in keys(tsg.tgdata)
                print(io, "\n\t$k = $(shortsummary(tsg.tgdata[k]))")
            end
        end
    end
end


function print_feature_t(io::IO, feature)
    if !isempty(feature)
        if length(keys(feature)) == 1
            k = first(keys(feature))
            v = first(values(feature))
            print(io, "$(k): $(dims2string(size(v)))")
        else
            print(io, "(")
            for (i, (k, v)) in enumerate(pairs(feature))
                print(io, "$k: $(dims2string(size(v)))")
                if i == length(feature)
                    print(io, ")")
                else
                    print(io, ", ")
                end
            end
        end
    else 
        print(io, "no")
    end
end