# Hetereogeneous Graphs


## GNNHeteroGraph
Documentation page for the type `GNNHeteroGraph` representing heterogeneous graphs, where  nodes and edges can have different types.


```@autodocs
Modules = [GNNGraphs]
Pages   = ["gnnheterograph.jl"]
Private = false
```

## Heterogeneous Graph Convolutions

Heterogeneous graph convolutions are implemented in the type [`HeteroGraphConv`](@ref).
`HeteroGraphConv` relies on standard graph convolutional layers to perform message passing on the different relations. See the table at [this page](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/api/conv/) for the supported layers.

```@docs
HeteroGraphConv
```