# Static Graphs 

The fundamental graph type in GNNGraphs.jl is the [`GNNGraph`](@ref).
A GNNGraph `g` is a directed graph with nodes labeled from 1 to `g.num_nodes`.
The underlying implementation allows for efficient application of graph neural network
operators, gpu movement, and storage of node/edge/graph related feature arrays.

`GNNGraph` inherits from [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl)'s `AbstractGraph`,
therefore it supports most functionality from that library. 

## Graph Creation
A GNNGraph can be created from several different data sources encoding the graph topology:

```julia
using GNNGraphs, Graphs, SparseArrays


# Construct a GNNGraph from from a Graphs.jl's graph
lg = erdos_renyi(10, 30)
g = GNNGraph(lg)

# Same as above using convenience method rand_graph
g = rand_graph(10, 60)

# From an adjacency matrix
A = sprand(10, 10, 0.3)
g = GNNGraph(A)

# From an adjacency list
adjlist = [[2,3], [1,3], [1,2,4], [3]]
g = GNNGraph(adjlist)

# From COO representation
source = [1,1,2,2,3,3,3,4]
target = [2,3,1,3,1,2,4,3]
g = GNNGraph(source, target)
```

See also the related methods [`Graphs.adjacency_matrix`](@ref), [`edge_index`](@ref), and [`adjacency_list`](@ref).

## Basic Queries

```julia
julia> source = [1,1,2,2,3,3,3,4];

julia> target = [2,3,1,3,1,2,4,3];

julia> g = GNNGraph(source, target)
GNNGraph:
  num_nodes: 4
  num_edges: 8


julia> @assert g.num_nodes == 4   # number of nodes

julia> @assert g.num_edges == 8   # number of edges

julia> @assert g.num_graphs == 1  # number of subgraphs (a GNNGraph can batch many graphs together)

julia> is_directed(g)      # a GNNGraph is always directed
true

julia> is_bidirected(g)      # for each edge, also the reverse edge is present
true

julia> has_self_loops(g)
false

julia> has_multi_edges(g)      
false
```

## Data Features

One or more arrays can be associated to nodes, edges, and (sub)graphs of a `GNNGraph`.
They will be stored in the fields `g.ndata`, `g.edata`, and `g.gdata` respectively.

The data fields are [`DataStore`](@ref) objects. [`DataStore`](@ref)s conveniently offer an interface similar to both dictionaries and named tuples. Similarly to dictionaries, DataStores support addition of new features after creation time.

The array contained in the datastores have last dimension equal to `num_nodes` (in `ndata`), `num_edges` (in `edata`), or `num_graphs` (in `gdata`) respectively.

```julia
# Create a graph with a single feature array `x` associated to nodes
g = rand_graph(10,  60, ndata = (; x = rand(Float32, 32, 10)))

g.ndata.x  # access the features

# Equivalent definition passing directly the array
g = rand_graph(10,  60, ndata = rand(Float32, 32, 10))

g.ndata.x  # `:x` is the default name for node features

g.ndata.z = rand(Float32, 3, 10)  # add new feature array `z`

# For convenience, we can access the features through the shortcut
g.x 

# You can have multiple feature arrays
g = rand_graph(10,  60, ndata = (; x=rand(Float32, 32, 10), y=rand(Float32, 10)))

g.ndata.y, g.ndata.x   # or g.x, g.y

# Attach an array with edge features.
# Since `GNNGraph`s are directed, the number of edges
# will be double that of the original Graphs' undirected graph.
g = GNNGraph(erdos_renyi(10,  30), edata = rand(Float32, 60))
@assert g.num_edges == 60

g.edata.e  # or g.e

# If we pass only half of the edge features, they will be copied
# on the reversed edges.
g = GNNGraph(erdos_renyi(10,  30), edata = rand(Float32, 30))


# Create a new graph from previous one, inheriting edge data
# but replacing node data
g′ = GNNGraph(g, ndata =(; z = ones(Float32, 16, 10)))

g′.z
g′.e
```

## Edge weights

It is common to denote scalar edge features as edge weights. The `GNNGraph` has specific support
for edge weights: they can be stored as part of internal representations of the graph (COO or adjacency matrix). Some graph convolutional layers, most notably the `GCNConv`, can use the edge weights to perform weighted sums over the nodes' neighborhoods.

```julia
julia> source = [1, 1, 2, 2, 3, 3];

julia> target = [2, 3, 1, 3, 1, 2];

julia> weight = [1.0, 0.5, 2.1, 2.3, 4, 4.1];

julia> g = GNNGraph(source, target, weight)
GNNGraph:
  num_nodes: 3
  num_edges: 6

julia> get_edge_weight(g)
6-element Vector{Float64}:
 1.0
 0.5
 2.1
 2.3
 4.0
 4.1
```

## Batches and Subgraphs

Multiple `GNNGraph`s can be batched together into a single graph
that contains the total number of the original nodes 
and where the original graphs are disjoint subgraphs.

```julia
using MLUtils

data = [rand_graph(10, 30, ndata=rand(Float32, 3, 10)) for _ in 1:160]
gall = MLUtils.batch(data)

# gall is a GNNGraph containing many graphs
@assert gall.num_graphs == 160 
@assert gall.num_nodes == 1600   # 10 nodes x 160 graphs
@assert gall.num_edges == 4800  # 30 undirected edges x 160 graphs

# Let's create a mini-batch from gall
g23 = getgraph(gall, 2:3)
@assert g23.num_graphs == 2
@assert g23.num_nodes == 20   # 10 nodes x 2 graphs
@assert g23.num_edges == 60  # 30 undirected edges X 2 graphs

# We can pass a GNNGraph to MLUtils' DataLoader
train_loader = DataLoader(gall, batchsize=16, shuffle=true)

for g in train_loader
    @assert g.num_graphs == 16
    @assert g.num_nodes == 160
    @assert size(g.ndata.x) = (3, 160)    
    # .....
end

# Access the nodes' graph memberships 
graph_indicator(gall)
```

## DataLoader and mini-batch iteration

While constructing a batched graph and passing it to the `DataLoader` is always 
an option for mini-batch iteration, the recommended way for better performance is
to pass an array of graphs directly and set the `collate` option to `true`:

```julia
using MLUtils: DataLoader

data = [rand_graph(10, 30, ndata=rand(Float32, 3, 10)) for _ in 1:320]

train_loader = DataLoader(data, batchsize=16, shuffle=true, collate=true)

for g in train_loader
    @assert g.num_graphs == 16
    @assert g.num_nodes == 160
    @assert size(g.ndata.x) = (3, 160)    
    # .....
end
```

## Graph Manipulation

```julia
g′ = add_self_loops(g)
g′ = remove_self_loops(g)
g′ = add_edges(g, [1, 2], [2, 3]) # add edges 1->2 and 2->3
```

## GPU movement

Move a `GNNGraph` to a CUDA device using `Flux.gpu` method. 

```julia
using Flux, CUDA # or using Metal or using AMDGPU 

g_gpu = g |> Flux.gpu
```

## Integration with Graphs.jl

Since `GNNGraph <: Graphs.AbstractGraph`, we can use any functionality from [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) for querying and analyzing the graph structure. 
Moreover, a `GNNGraph` can be easily constructed from a `Graphs.Graph` or a `Graphs.DiGraph`:

```julia
julia> import Graphs

julia> using GNNGraphs

# A Graphs.jl undirected graph
julia> gu = Graphs.erdos_renyi(10, 20)    
{10, 20} undirected simple Int64 graph

# Since GNNGraphs are undirected, the edges are doubled when converting 
# to GNNGraph
julia> GNNGraph(gu)
GNNGraph:
  num_nodes: 10
  num_edges: 40

# A Graphs.jl directed graph
julia> gd = Graphs.erdos_renyi(10, 20, is_directed=true)
{10, 20} directed simple Int64 graph

julia> GNNGraph(gd)
GNNGraph:
  num_nodes: 10
  num_edges: 20
```
