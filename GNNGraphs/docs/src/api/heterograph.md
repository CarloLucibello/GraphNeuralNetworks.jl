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

## Query

```@autodocs
Modules = [GNNGraphs]
Pages   = ["gnnheterograph/query.jl"]
Private = false
```

```@docs
Graphs.has_edge(g::GNNHeteroGraph, edge_t::EType, i::Integer, j::Integer)
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
