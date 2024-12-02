```@meta
CurrentModule = GNNGraphs
CollapsedDocStrings = true
```

# Heterogeneous Graphs


## GNNHeteroGraph
Documentation page for the type `GNNHeteroGraph` representing heterogeneous graphs, where  nodes and edges can have different types.


```@autodocs
Modules = [GNNGraphs]
Pages   = ["gnnheterograph.jl"]
Private = false
```

```@docs
Graphs.has_edge(::GNNHeteroGraph, ::Tuple{Symbol, Symbol, Symbol}, ::Integer, ::Integer)
```

## Query

```@autodocs
Modules = [GNNGraphs]
Pages   = ["gnnheterograph/query.jl"]
Private = false
```

## Transform

```@autodocs
Modules = [GNNGraphs]
Pages   = ["gnnheterograph/transform.jl"]
Private = false
```

## Generate

```@autodocs
Modules = [GNNGraphs]
Pages   = ["gnnheterograph/generate.jl"]
Private = false
```
