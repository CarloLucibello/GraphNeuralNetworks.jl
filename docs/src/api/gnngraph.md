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

### GNNGraph

```@autodocs
Modules = [GraphNeuralNetworks]
Pages   = ["GNNGraphs/gnngraph.jl"]
Private = false
```

### Query

```@autodocs
Modules = [GraphNeuralNetworks]
Pages   = ["GNNGraphs/query.jl"]
Private = false
```

```@docs
Graphs.adjacency_matrix
Graphs.degree
Graphs.outneighbors
Graphs.inneighbors
```

### Transform

```@autodocs
Modules = [GraphNeuralNetworks]
Pages   = ["GNNGraphs/transform.jl"]
Private = false
```

```@docs
Flux.batch
SparseArrays.blockdiag
```

### Generate

```@autodocs
Modules = [GraphNeuralNetworks]
Pages   = ["GNNGraphs/generate.jl"]
Private = false
```

### Related methods

```@docs
SparseArrays.sparse
```
