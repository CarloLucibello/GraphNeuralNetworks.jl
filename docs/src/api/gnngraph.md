```@meta
CurrentModule = GraphNeuralNetworks
```

# GNNGraph

Documentation page for the graph type `GNNGraph` provided by GraphNeuralNetworks.jl and related methods. 

Besides the methods documented here, one can rely on the large set of functionalities
given by [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) thanks to the fact
that `GNNGraph` inherits from `Graphs.AbstractGraph`.

## Index 

```@index
Order = [:type, :function]
Pages   = ["gnngraph.md"]
```

## GNNGraph type

```@docs
GNNGraph
```

## DataStore

```@autodocs
Modules = [GraphNeuralNetworks.GNNGraphs]
Pages   = ["datastore.jl"]
Private = false
```

## Query

```@autodocs
Modules = [GraphNeuralNetworks.GNNGraphs]
Pages   = ["query.jl"]
Private = false
```

```@docs
Graphs.outneighbors
Graphs.inneighbors
```

## Transform

```@autodocs
Modules = [GraphNeuralNetworks.GNNGraphs]
Pages   = ["transform.jl"]
Private = false
```

## Generate

```@autodocs
Modules = [GraphNeuralNetworks.GNNGraphs]
Pages   = ["generate.jl"]
Private = false
Filter = t -> typeof(t) <: Function && t!=rand_temporal_radius_graph && t!=rand_temporal_hyperbolic_graph

```

## Operators

```@autodocs
Modules = [GraphNeuralNetworks.GNNGraphs]
Pages   = ["operators.jl"]
Private = false
```

```@docs
Graphs.intersect
```

## Sampling 

```@autodocs
Modules = [GraphNeuralNetworks.GNNGraphs]
Pages   = ["sampling.jl"]
Private = false
```
