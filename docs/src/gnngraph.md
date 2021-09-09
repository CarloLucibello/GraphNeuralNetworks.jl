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
GNNGraph(sprand(10, 0.3), ndata = (; X=rand(32, 10)))
# or equivalently
GNNGraph(sprand(10, 0.3), ndata=rand(32, 10))


g = GNNGraph(sprand(10, 0.3), ndata = (X=rand(32, 10), y=rand(10)))

g = GNNGraph(g, edata=rand(6, g.num_edges))
```


## Graph Manipulation

```julia
g = add_self_loops(g)

g = remove_self_loops(g)
```

## Batches and Subgraphs

```julia
g = Flux.batch([g1, g2, g3])

subgraph(g, 2:3)
```


## LightGraphs integration

```julia
@assert LightGraphs.isdirected(g)
```

## Other methods
