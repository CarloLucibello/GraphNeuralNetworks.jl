```@meta
CurrentModule = GNNGraphs
```

# GNNGraph

Documentation page for the graph type `GNNGraph` provided by GNNGraphs.jl and related methods. 

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
Base.copy
```

## DataStore

```@autodocs
Modules = [GNNGraphs]
Pages   = ["datastore.jl"]
Private = false
```

## Query

```@autodocs
Modules = [GNNGraphs]
Pages   = ["query.jl"]
Private = false
```

```@docs
Graphs.neighbors(::GNNGraph, ::Integer)
```

## Transform

```@autodocs
Modules = [GNNGraphs]
Pages   = ["transform.jl"]
Private = false
```

## Utils

```@docs
GNNGraphs.sort_edge_index
GNNGraphs.color_refinement
``` 

## Generate

```@autodocs
Modules = [GNNGraphs]
Pages   = ["generate.jl"]
Private = false
Filter = t -> typeof(t) <: Function && t!=rand_temporal_radius_graph && t!=rand_temporal_hyperbolic_graph

```

## Operators

```@autodocs
Modules = [GNNGraphs]
Pages   = ["operators.jl"]
Private = false
```

```@docs
Base.intersect
```

## Sampling 

```@autodocs
Modules = [GNNGraphs]
Pages   = ["sampling.jl"]
Private = false
```

```@docs
Graphs.induced_subgraph(::GNNGraph, ::Vector{Int})
```