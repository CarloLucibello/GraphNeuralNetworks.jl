# Temporal Graphs

Temporal Graphs are graphs with time varying topologies and  features. In GNNGraphs.jl, temporal graphs with fixed number of nodes over time are supported by the [`TemporalSnapshotsGNNGraph`](@ref) type.

## Creating a TemporalSnapshotsGNNGraph

A temporal graph can be created by passing a list of snapshots to the constructor. Each snapshot is a [`GNNGraph`](@ref). 

```jldoctest
julia> snapshots = [rand_graph(10,20) for i in 1:5];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [20, 20, 20, 20, 20]
  num_snapshots: 5
```

A new temporal graph can be created by adding or removing snapshots to an existing temporal graph. 

```jldoctest
julia> new_tg = add_snapshot(tg, 3, rand_graph(10, 16)) # add a new snapshot at time 3
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10, 10]
  num_edges: [20, 20, 16, 20, 20, 20]
  num_snapshots: 6
```
```jldoctest
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

See [`rand_temporal_radius_graph`](@ref) and [`rand_temporal_hyperbolic_graph`](@ref) for generating random temporal graphs. 

```jldoctest
julia> tg = rand_temporal_radius_graph(10, 3, 0.1, 0.5)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [32, 30, 34]
  num_snapshots: 3
``` 

## Basic Queries

Basic queries are similar to those for [`GNNGraph`](@ref)s:
```jldoctest
julia> snapshots = [rand_graph(10,20), rand_graph(10,14), rand_graph(10,22)];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 14, 22]
  num_snapshots: 3

julia> tg.num_nodes         # number of nodes in each snapshot
3-element Vector{Int64}:
 10
 10
 10

julia> tg.num_edges         # number of edges in each snapshot
3-element Vector{Int64}:
 20
 14
 22

julia> tg.num_snapshots     # number of snapshots
3

julia> tg.snapshots         # list of snapshots
3-element Vector{GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}}:
 GNNGraph(10, 20) with no data
 GNNGraph(10, 14) with no data
 GNNGraph(10, 22) with no data

julia> tg.snapshots[1]      # first snapshot, same as tg[1]
GNNGraph:
  num_nodes: 10
  num_edges: 20
```

## Data Features
A temporal graph can store global feature for the entire time series in the `tgdata` filed.
Also, each snapshot can store node, edge, and graph features in the `ndata`, `edata`, and `gdata` fields, respectively. 

```jldoctest
julia> snapshots = [rand_graph(10,20; ndata = rand(3,10)), rand_graph(10,14; ndata = rand(4,10)), rand_graph(10,22; ndata = rand(5,10))]; # node features at construction time

julia> tg = TemporalSnapshotsGNNGraph(snapshots);

julia> tg.tgdata.y = rand(3,1); # add global features after construction

julia> tg
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 14, 22]
  num_snapshots: 3
  tgdata:
        y = 3×1 Matrix{Float64}

julia> tg.ndata # vector of DataStore containing node features for each snapshot
3-element Vector{DataStore}:
 DataStore(10) with 1 element:
  x = 3×10 Matrix{Float64}
 DataStore(10) with 1 element:
  x = 4×10 Matrix{Float64}
 DataStore(10) with 1 element:
  x = 5×10 Matrix{Float64}

julia> [ds.x for ds in tg.ndata]; # vector containing the x feature of each snapshot

julia> [g.x for g in tg.snapshots]; # same vector as above, now accessing 
                                   # the x feature directly from the snapshots
```

## Graph convolutions on TemporalSnapshotsGNNGraph

A graph convolutional layer can be applied to each snapshot independently, in the next example we apply a `GINConv` layer to each snapshot of a `TemporalSnapshotsGNNGraph`.  

```jldoctest
julia> using GNNGraphs, Flux

julia> snapshots = [rand_graph(10, 20; ndata = rand(3, 10)), rand_graph(10, 14; ndata = rand(3, 10))];

julia> tg = TemporalSnapshotsGNNGraph(snapshots);

julia> m = GINConv(Dense(3 => 1), 0.4);

julia> output = m(tg, tg.ndata.x);

julia> size(output[1])
(1, 10)
```

