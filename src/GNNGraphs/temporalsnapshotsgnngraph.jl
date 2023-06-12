"""
    TemporalSnapshotsGNNGraph(snapshots::AbstractVector{<:GNNGraph})

A type representing a temporal graph as a sequence of snapshots, in this case a snapshot is a [`GNNGraph`](@ref).

It stores the feature array associated to the graph itself as a [`DataStore`](@ref) object, and it uses the [`DataStore`](@ref) objects of each snapshot for the node and edge features.
The features can be passed at construction time or added later.

# Arguments
- snapshot: a vector of snapshots, each snapshot must have the same number of nodes.

# Examples
```julia
julia> using GraphNeuralNetworks

julia> snapshots = [rand_graph(10,20) for i in 1:5];

julia> tgs = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [20, 20, 20, 20, 20]
  num_snapshots: 5

julia> tgs.tgdata.x = rand(4); # add temporal graph feature

julia> tgs # show temporal graph with new feature
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [20, 20, 20, 20, 20]
  num_snapshots: 5
  tgdata:
        x = 4-element Vector{Float64}
```
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

"""
    add_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)

Return a `TemporalSnapshotsGNNGraph` created starting from `tg` by adding the snapshot `g` at time index `t`.

# Example
```julia
julia> using GraphNeuralNetworks

julia> snapshots = [rand_graph(10,20) for i in 1:5];

julia> tgs = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [20, 20, 20, 20, 20]
  num_snapshots: 5

julia> new_tgs = add_snapshot(tgs, 3, rand_graph(10,16)) # add a new snapshot at time 3
TemporalSnapshotsGNNGraph:
num_nodes: [10, 10, 10, 10, 10, 10]
num_edges: [20, 20, 16, 20, 20, 20]
num_snapshots: 6
```
"""
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

"""
    remove_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int)

Return a `TemporalSnapshotsGNNGraph` created starting from `tg` by removing the snapshot at time index `t`.

# Example
```julia
julia> using GraphNeuralNetworks

julia> snapshots = [rand_graph(10,20), rand_graph(10,14), rand_graph(10,22)];

julia> tgs = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 14, 22]
  num_snapshots: 3

julia> new_tgs = remove_snapshot(tgs,2) # remove snapshot at time 2
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10]
  num_edges: [20, 22]
  num_snapshots: 2
```
"""
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

"""
    getproperty(tg::TemporalSnapshotsGNNGraph, prop::Symbol)

If `prop` is a field of `TemporalSnapshotsGNNGraph` return the corresponding value, if `prop` is `:ndata`, `:edata` or `:gdata` return the corresponding array of `DataStore`, otherwise return an array containing the `prop` feature of each snapshot.

# Examples
```julia
julia> snaps=[rand_graph(10,20,ndata = rand(3,10)) for i in 1:3];

julia> tgs = TemporalSnapshotsGNNGraph(snaps)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 20, 20]
  num_snapshots: 3

julia> tgs.ndata
3-element Vector{DataStore}:
 DataStore(10) with 1 element:
  x = 3×10 Matrix{Float64}
 DataStore(10) with 1 element:
  x = 3×10 Matrix{Float64}
 DataStore(10) with 1 element:
  x = 3×10 Matrix{Float64}

julia> tgs.ndata.x
3-element Vector{Matrix{Float64}}:
 [0.8673509190658151 0.44507039178583574 … 0.3540406246655291 0.32301290218226175; 0.2657940264407055 0.8955046116193814 … 0.33941467211298426 0.38485049221502465; 0.48030055946962036 0.7127377333270681 … 0.21132801599438156 0.8045821310635392]
 [0.8195347950242018 0.9823449883057142 … 0.530094072794728 0.49424179101438703; 0.16734985599253294 0.7669357123643717 … 0.3501525426697579 0.5951573310727881; 0.23798757760327838 0.8353144950964572 … 0.04964083551409626 0.4725336527008097]
 [0.9655069785658686 0.8354570027415478 … 0.8238130292494555 0.8478137086159969; 0.2445514712259027 0.19057411837712268 … 0.8214337153973568 0.9790470076645307; 0.6972044915443449 0.22701496021424217 … 0.5837902745512978 0.6671225562067188]
```
"""
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