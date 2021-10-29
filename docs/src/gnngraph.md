# Graphs

The fundamental graph type in GraphNeuralNetworks.jl is the [`GNNGraph`](@ref).
A GNNGraph `g` is a directed graph with nodes labeled from 1 to `g.num_nodes`.
The underlying implementation allows for efficient application of graph neural network
operators, gpu movement, and storage of node/edge/graph related feature arrays.

`GNNGraph` inherits from [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl)'s `AbstractGraph`,
therefore it supports most functionality from that library. 

## Graph Creation
A GNNGraph can be created from several different data sources encoding the graph topology:

```julia
using GraphNeuralNetworks, Graphs, SparseArrays


# Construct GNNGraph from From Graphs's graph
lg = erdos_renyi(10, 30)
g = GNNGraph(lg)

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

See also the related methods [`adjacency_matrix`](@ref), [`edge_index`](@ref), and [`adjacency_list`](@ref).

## Basic Queries

```julia
source = [1,1,2,2,3,3,3,4]
target = [2,3,1,3,1,2,4,3]
g = GNNGraph(source, target)

@assert g.num_nodes == 4   # number of nodes
@assert g.num_edges == 8   # number of edges
@assert g.num_graphs == 1  # number of subgraphs (a GNNGraph can batch many graphs together)
is_directed(g)      # a GGNGraph is always directed
```

## Data Features

One or more arrays can be associated to nodes, edges, and (sub)graphs of a `GNNGraph`.
They will be stored in the fields `g.ndata`, `g.edata`, and `g.gdata` respectivaly.
The data fields are `NamedTuple`s. The array they contain must have last dimension
equal to `num_nodes` (in `ndata`), `num_edges` (in `edata`), or `num_graphs` (in `gdata`).

```julia
# Create a graph with a single feature array `x` associated to nodes
g = GNNGraph(erdos_renyi(10,  30), ndata = (; x = rand(Float32, 32, 10)))

g.ndata.x  # access the features

# Equivalent definition passing directly the array
g = GNNGraph(erdos_renyi(10,  30), ndata = rand(Float32, 32, 10))

g.ndata.x  # `:x` is the default name for node features

# You can have multiple feature arrays
g = GNNGraph(erdos_renyi(10,  30), ndata = (; x=rand(Float32, 32, 10), y=rand(Float32, 10)))

g.ndata.y, g.ndata.x

# Attach an array with edge features.
# Since `GNNGraph`s are directed, the number of edges
# will be double that of the original Graphs' undirected graph.
g = GNNGraph(erdos_renyi(10,  30), edata = rand(Float32, 60))
@assert g.num_edges == 60

g.edata.e

# If we pass only half of the edge features, they will be copied
# on the reversed edges.
g = GNNGraph(erdos_renyi(10,  30), edata = rand(Float32, 30))


# Create a new graph from previous one, inheriting edge data
# but replacing node data
g′ = GNNGraph(g, ndata =(; z = ones(Float32, 16, 10)))

g.ndata.z
g.edata.e
```

## Batches and Subgraphs

Multiple `GNNGraph`s can be batched togheter into a single graph
containing the total number of the original nodes 
and where the original graphs are disjoint subgraphs.

```julia
using Flux

gall = Flux.batch([GNNGraph(erdos_renyi(10, 30), ndata=rand(Float32,3,10)) for _ in 1:160])

@assert gall.num_graphs == 160 
@assert gall.num_nodes == 1600   # 10 nodes x 160 graphs
@assert gall.num_edges == 9600  # 30 undirected edges x 2 directions x 160 graphs

g23, _ = getgraph(gall, 2:3)
@assert g23.num_graphs == 2
@assert g23.num_nodes == 20   # 10 nodes x 160 graphs
@assert g23.num_edges == 120  # 30 undirected edges x 2 directions x 2 graphs x


# DataLoader compatibility
train_loader = Flux.Data.DataLoader(gall, batchsize=16, shuffle=true)

for g in train_loader
    @assert g.num_graphs == 16
    @assert g.num_nodes == 160
    @assert size(g.ndata.x) = (3, 160)    
    .....
end

# Access the nodes' graph memberships through 
gall.graph_indicator
```

## Graph Manipulation

```julia
g′ = add_self_loops(g)

g′ = remove_self_loops(g)
```

## JuliaGraphs ecosystem integration

Since `GNNGraph <: Graphs.AbstractGraph`, we can use any functionality from Graphs. 

```julia
@assert Graphs.isdirected(g)
```

## GPU movement

Move a `GNNGraph` to a CUDA device using `Flux.gpu` method. 

```julia
using Flux: gpu

g_gpu = g |> gpu
```
