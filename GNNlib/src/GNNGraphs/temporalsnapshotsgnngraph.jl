"""
    TemporalSnapshotsGNNGraph(snapshots::AbstractVector{<:GNNGraph})

A type representing a temporal graph as a sequence of snapshots. In this case a snapshot is a [`GNNGraph`](@ref).

`TemporalSnapshotsGNNGraph` can store the feature array associated to the graph itself as a [`DataStore`](@ref) object, 
and it uses the [`DataStore`](@ref) objects of each snapshot for the node and edge features.
The features can be passed at construction time or added later.

# Constructor Arguments

- `snapshot`: a vector of snapshots, where each snapshot must have the same number of nodes.

# Examples

```julia
julia> using GraphNeuralNetworks

julia> snapshots = [rand_graph(10,20) for i in 1:5];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [20, 20, 20, 20, 20]
  num_snapshots: 5

julia> tg.tgdata.x = rand(4); # add temporal graph feature

julia> tg # show temporal graph with new feature
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [20, 20, 20, 20, 20]
  num_snapshots: 5
  tgdata:
        x = 4-element Vector{Float64}
```
"""
struct TemporalSnapshotsGNNGraph
    num_nodes::AbstractVector{Int}   
    num_edges::AbstractVector{Int}
    num_snapshots::Int
    snapshots::AbstractVector{<:GNNGraph}
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

"""
    add_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)

Return a `TemporalSnapshotsGNNGraph` created starting from `tg` by adding the snapshot `g` at time index `t`.

# Examples

```jldoctest
julia> using GraphNeuralNetworks

julia> snapshots = [rand_graph(10, 20) for i in 1:5];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [20, 20, 20, 20, 20]
  num_snapshots: 5

julia> new_tg = add_snapshot(tg, 3, rand_graph(10, 16)) # add a new snapshot at time 3
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10, 10]
  num_edges: [20, 20, 16, 20, 20, 20]
  num_snapshots: 6
```
"""
function add_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)
    if tg.num_snapshots > 0
        @assert g.num_nodes == first(tg.num_nodes) "number of nodes must match"
    end
    @assert t <= tg.num_snapshots + 1 "cannot add snapshot at time $t, the temporal graph has only $(tg.num_snapshots) snapshots"
    num_nodes = tg.num_nodes |> copy
    num_edges = tg.num_edges |> copy
    snapshots = tg.snapshots |> copy
    num_snapshots = tg.num_snapshots + 1
    insert!(num_nodes, t, g.num_nodes)
    insert!(num_edges, t, g.num_edges)
    insert!(snapshots, t, g)
    return TemporalSnapshotsGNNGraph(num_nodes, num_edges, num_snapshots, snapshots, tg.tgdata) 
end

# """
#     add_snapshot!(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)

# Add to `tg` the snapshot `g` at time index `t`.

# See also [`add_snapshot`](@ref) for a non-mutating version.
# """
# function add_snapshot!(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)
#     if t > tg.num_snapshots + 1
#         error("cannot add snapshot at time $t, the temporal graph has only $(tg.num_snapshots) snapshots")
#     end
#     if tg.num_snapshots > 0
#         @assert g.num_nodes == first(tg.num_nodes) "number of nodes must match"
#     end
#     insert!(tg.num_nodes, t, g.num_nodes)
#     insert!(tg.num_edges, t, g.num_edges)
#     insert!(tg.snapshots, t, g)
#     return tg 
# end

"""
    remove_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int)

Return a [`TemporalSnapshotsGNNGraph`](@ref) created starting from `tg` by removing the snapshot at time index `t`.

# Examples

```jldoctest
julia> using GraphNeuralNetworks

julia> snapshots = [rand_graph(10,20), rand_graph(10,14), rand_graph(10,22)];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 14, 22]
  num_snapshots: 3

julia> new_tg = remove_snapshot(tg, 2) # remove snapshot at time 2
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10]
  num_edges: [20, 22]
  num_snapshots: 2
```
"""
function remove_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int)
    num_nodes = tg.num_nodes |> copy
    num_edges = tg.num_edges |> copy
    snapshots = tg.snapshots |> copy
    num_snapshots = tg.num_snapshots - 1
    deleteat!(num_nodes, t)
    deleteat!(num_edges, t)
    deleteat!(snapshots, t)
    return TemporalSnapshotsGNNGraph(num_nodes, num_edges, num_snapshots, snapshots, tg.tgdata) 
end

# """
#     remove_snapshot!(tg::TemporalSnapshotsGNNGraph, t::Int)

# Remove the snapshot at time index `t` from `tg` and return `tg`.

# See [`remove_snapshot`](@ref) for a non-mutating version.
# """
# function remove_snapshot!(tg::TemporalSnapshotsGNNGraph, t::Int)
#     @assert t <= tg.num_snapshots "snapshot index $t out of bounds"
#     tg.num_snapshots -= 1
#     deleteat!(tg.num_nodes, t)
#     deleteat!(tg.num_edges, t)
#     deleteat!(tg.snapshots, t)
#     return tg
# end

function Base.getproperty(tg::TemporalSnapshotsGNNGraph, prop::Symbol)
    if prop ∈ fieldnames(TemporalSnapshotsGNNGraph)
        return getfield(tg, prop)
    elseif prop == :ndata
        return [s.ndata for s in tg.snapshots]
    elseif prop == :edata
        return [s.edata for s in tg.snapshots]
    elseif prop == :gdata
        return [s.gdata for s in tg.snapshots]
    else 
        return [getproperty(s,prop) for s in tg.snapshots]
    end
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

@functor TemporalSnapshotsGNNGraph
