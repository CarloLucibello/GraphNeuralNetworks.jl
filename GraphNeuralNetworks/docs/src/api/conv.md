```@meta
CurrentModule = GraphNeuralNetworks
CollapsedDocStrings = true
```

# Convolutional Layers

Many different types of graphs convolutional layers have been proposed in the literature. Choosing the right layer for your application could involve a lot of exploration. 
Some of the most commonly used layers are the [`GCNConv`](@ref) and the [`GATv2Conv`](@ref). Multiple graph convolutional layers are typically stacked together to create a graph neural network model
(see [`GNNChain`](@ref)).

The table below lists all graph convolutional layers implemented in the *GraphNeuralNetworks.jl*. It also highlights the presence of some additional capabilities with respect to basic message passing:
- *Sparse Ops*: implements message passing as multiplication by sparse adjacency matrix instead of the gather/scatter mechanism. This can lead to better CPU performances but it is not supported on GPU yet. 
- *Edge Weight*: supports scalar weights (or equivalently scalar features) on edges. 
- *Edge Features*: supports feature vectors on edges.
- *Heterograph*: supports heterogeneous graphs (see [`GNNHeteroGraph`](@ref)).
- *TemporalSnapshotsGNNGraphs*: supports temporal graphs (see [`TemporalSnapshotsGNNGraph`](@ref)) by applying the convolution layers to each snapshot independently.

| Layer                       |Sparse Ops|Edge Weight|Edge Features| Heterograph  | TemporalSnapshotsGNNGraphs |
| :--------                   |  :---:   |:---:      |:---:        |  :---:       | :---:                      |
| [`AGNNConv`](@ref)          |          |           |     ✓       |              |                    |                          
| [`CGConv`](@ref)            |          |           |     ✓       |       ✓      |             ✓             | 
| [`ChebConv`](@ref)          |          |           |             |              |                ✓           |
| [`EGNNConv`](@ref)          |          |           |     ✓       |              |                           |
| [`EdgeConv`](@ref)          |          |           |             |       ✓      |                            |  
| [`GATConv`](@ref)           |          |           |     ✓       |       ✓      |              ✓             |
| [`GATv2Conv`](@ref)         |          |           |     ✓       |       ✓      |             ✓              |
| [`GatedGraphConv`](@ref)    |     ✓    |           |             |              |            ✓               |
| [`GCNConv`](@ref)           |     ✓    |     ✓     |             |       ✓      |                            |
| [`GINConv`](@ref)           |     ✓    |           |             |       ✓      |               ✓           |
| [`GMMConv`](@ref)           |          |           |     ✓       |              |                            |
| [`GraphConv`](@ref)         |     ✓    |           |             |       ✓      |              ✓             |   
| [`MEGNetConv`](@ref)        |          |           |     ✓       |              |                            |              
| [`NNConv`](@ref)            |          |           |     ✓       |              |                            |
| [`ResGatedGraphConv`](@ref) |          |           |             |       ✓      |               ✓             |
| [`SAGEConv`](@ref)          |     ✓    |           |             |       ✓      |             ✓               |
| [`SGConv`](@ref)            |     ✓    |           |             |              |             ✓             |
| [`TransformerConv`](@ref)   |          |           |     ✓       |              |                           |



```@autodocs
Modules = [GraphNeuralNetworks]
Pages   = ["layers/conv.jl"]
Private = false
```
