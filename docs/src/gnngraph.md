# Graphs

TODO

## Graph Creation


```julia
using GraphNeuralNetworks, LightGraphs, SparseArrays


# From LightGraphs's graph
lg_graph = erdos_renyi(10, 0.3)
g = GNNGraph(lg_graph)


# From adjacency matrix
A = sprand(10, 10, 0.3)

g = GNNGraph(A)

@assert adjacency_matrix(g) == A

# From adjacency list
adjlist = [[] [] [] ]

g = GNNGraph(adjlist)

@assert sort.(adjacency_list(g)) == sort.(adjlist)

# From COO representation
source = []
target = []
g = GNNGraph(source, target)
@assert edge_index(g) == (source, target)
```

We have also seen some useful methods such as [`adjacency_matrix`](@ref) and [`edge_index`](@ref).



## Data Features

```julia
GNNGraph(erods_renyi(10,  30), ndata = (; X=rand(Float32, 32, 10)))
# or equivalently
GNNGraph(sprand(10, 0.3), ndata=rand(Float32, 32, 10))

g = GNNGraph(sprand(10, 0.3), ndata = (X=rand(Float32, 32, 10), y=rand(Float32, 10)))

g = GNNGraph(g, edata=rand(Float32, 6, g.num_edges))
```


## Graph Manipulation

```julia
g = add_self_loops(g)

g = remove_self_loops(g)
```

## Batches and Subgraphs

```julia
using Flux

gall = Flux.batch([GNNGraph(erdos_renyi(10, 30), ndata=rand(3,10)) for _ in 1:100])

subgraph(gall, 2:3)


# DataLoader compatibility
train_loader = Flux.Data.DataLoader(gall, batchsize=16, shuffle=true)

for g for gall
    @assert g.num_graphs == 16
    @assert g.num_nodes == 160
    @assert size(g.ndata.X) = (3, 160)    
    .....
end
```

## LightGraphs integration

```julia
@assert LightGraphs.isdirected(g)
```

## GPU movement

```julia
using Flux: gpu

g |> gpu
```

## Other methods
