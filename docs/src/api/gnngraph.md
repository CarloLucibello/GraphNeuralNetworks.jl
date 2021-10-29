```@meta
CurrentModule = GraphNeuralNetworks
```

# GNNGraph

Documentation page for the graph type `GNNGraph` provided GraphNeuralNetworks.jl and its related methods. 

Besides the methods documented here, one can rely on the large set of functionalities
given by [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl)
since `GNNGraph` inherits from `Graphs.AbstractGraph`.

## Index 

```@index
Order = [:type, :function]
Pages   = ["gnngraph.md"]
```

## Docs

```@autodocs
Modules = [GraphNeuralNetworks]
Pages   = ["gnngraph.jl"]
Private = false
```

```@docs
Flux.batch
SparseArrays.blockdiag
Graphs.adjacency_matrix
```
