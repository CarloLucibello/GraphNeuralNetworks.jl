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

function Base.getindex(tg::TemporalSnapshotsGNNGraph, t::Int)
    return tg.snapshots[t]
end

function Base.getindex(tg::TemporalSnapshotsGNNGraph, t::AbstractVector)
    return TemporalSnapshotsGNNGraph(tg.num_nodes[t], tg.num_edges[t], length(t), tg.snapshots[t], tg.tgdata)
end