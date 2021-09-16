# Graphs

The fundamental graph type in GraphNeuralNetworks.jl is the [`GNNGraph`](@ref), 
A GNNGraph `g` is a directed graph with nodes labeled from 1 to `g.num_nodes`.
The underlying implementation allows for efficient application of graph neural network
operators, gpu movement, and storage of node/edge/graph related feature arrays.

## Graph Creation
A GNNGraph can be created from several different data sources encoding the graph topology:

```julia
using GraphNeuralNetworks, LightGraphs, SparseArrays


# Construct GNNGraph from From LightGraphs's graph
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


## Data Features

```julia
# Create a graph with a single feature array `x` associated to nodes
g = GNNGraph(erdos_renyi(10,  30), ndata = (; x = rand(Float32, 32, 10)))
# Equivalent definition
g = GNNGraph(erdos_renyi(10,  30), ndata = rand(Float32, 32, 10))

# You can have multiple feature arrays
g = GNNGraph(erdos_renyi(10,  30), ndata = (; x=rand(Float32, 32, 10), y=rand(Float32, 10)))


# Attach an array with edge features
g = GNNGraph(erdos_renyi(10,  30), edata = rand(Float32, 30))

# Create a new graph from previous one, inheriting edge data
# but replacing node data
g′ = GNNGraph(g, ndata =(; z = ones(Float32, 16, 10)))
```


## Graph Manipulation

```julia
g′ = add_self_loops(g)

g′ = remove_self_loops(g)
```

## Batches and Subgraphs

```julia
using Flux

gall = Flux.batch([GNNGraph(erdos_renyi(10, 30), ndata=rand(Float32,3,10)) for _ in 1:160])

g23 = getgraph(gall, 2:3)
@assert g23.num_graphs == 16
@assert g23.num_nodes == 32
@assert g23.num_edges == 60


# DataLoader compatibility
train_loader = Flux.Data.DataLoader(gall, batchsize=16, shuffle=true)

for g in train_loader
    @assert g.num_graphs == 16
    @assert g.num_nodes == 160
    @assert size(g.ndata.x) = (3, 160)    
    .....
end
```

## JuliaGraphs ecosystem integration

Since `GNNGraph <: LightGraphs.AbstractGraph`, we can use any functionality from LightGraphs. 

```julia
@assert LightGraphs.isdirected(g)
```

## GPU movement

Move a `GNNGraph` to a CUDA device using `Flux.gpu` method. 

```julia
using Flux: gpu

g_gpu = g |> gpu
```
