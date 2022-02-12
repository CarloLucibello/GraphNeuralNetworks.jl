```@meta
CurrentModule = GraphNeuralNetworks
```

# Convolutional Layers

There exist a rich zoology of convolutional layers. Which one to choose heavily depends on your application. Some of the most commonly used layers are the [`GCNConv`](@ref) and the [`GATv2Conv`](@ref) layers.

The table below lists all convolutional layers implemented in the library and highlights
the possible presence of some features in addition to the basic propagation of node features:
- *Sparse Ops*: implements message passing as multiplication by sparse adjacency matrix instead of the gather/scatter mechanism. This can lead to better cpu performances but it is not supported on gpu yet. 
- *Edge Weights*: supports scalar weights on edges. 
- *Edge Features*: supports feature vectors on edges.


| Layer                       |Sparse Ops|Edge Weight|Edge Features| 
| :--------                   |  :---:   |:---:      |:---:        |               
| [`AGNNConv`](@ref)          |          |           |    ✓        |
| [`CGConv`](@ref)            |          |           |             |
| [`ChebConv`](@ref)          |          |           |             |
| [`EdgeConv`](@ref)          |          |           |             |
| [`GATConv`](@ref)           |          |           |             |
| [`GATv2Conv`](@ref)         |          |           |             |
| [`GatedGraphConv`](@ref)    |          |           |             |
| [`GCNConv`](@ref)           |     ✓    |     ✓     |             | 
| [`GINConv`](@ref)           |          |           |             |
| [`GraphConv`](@ref)         |          |           |             |
| [`MEGNetConv`](@ref)        |          |           |             |
| [`NNConv`](@ref)            |          |           |     ✓       |
| [`ResGatedGraphConv`](@ref) |          |           |             |
| [`SAGEConv`](@ref)          |          |           |             |


## Docs

```@autodocs
Modules = [GraphNeuralNetworks]
Pages   = ["layers/conv.jl"]
Private = false
```
