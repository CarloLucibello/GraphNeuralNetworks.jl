```@meta
CurrentModule = GNNLux
```

# Convolutional Layers

Many different types of graphs convolutional layers have been proposed in the literature. Choosing the right layer for your application could involve a lot of exploration. 
Multiple graph convolutional layers are typically stacked together to create a graph neural network model (see [`GNNChain`](@ref)).

The table below lists all graph convolutional layers implemented in the *GNNLux.jl*. It also highlights the presence of some additional capabilities with respect to basic message passing:
- *Sparse Ops*: implements message passing as multiplication by sparse adjacency matrix instead of the gather/scatter mechanism. This can lead to better CPU performances but it is not supported on GPU yet. 
- *Edge Weight*: supports scalar weights (or equivalently scalar features) on edges. 
- *Edge Features*: supports feature vectors on edges.
- *Heterograph*: supports heterogeneous graphs (see [`GNNHeteroGraph`](@ref)).
- *TemporalSnapshotsGNNGraphs*: supports temporal graphs (see [`TemporalSnapshotsGNNGraph`](@ref)) by applying the convolution layers to each snapshot independently.

| Layer                       |Sparse Ops|Edge Weight|Edge Features| Heterograph  | TemporalSnapshotsGNNGraphs |
| :--------                   |  :---:   |:---:      |:---:        |  :---:       | :---:                      |         ✓               |
| [`GCNConv`](@ref)           |     ✓    |     ✓     |             |       ✓      |                            |

## Docs

```@autodocs
Modules = [GraphNeuralNetworks]
Pages   = ["layers/conv.jl"]
Private = false
```