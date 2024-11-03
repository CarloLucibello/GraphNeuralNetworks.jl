# Datasets

GraphNeuralNetworks.jl doesn't come with its own datasets, but leverages those available in the Julia (and non-Julia) ecosystem. In particular, the [examples in the GraphNeuralNetworks.jl repository](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/examples) make use of the [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl) package. There you will find common graph datasets such as Cora, PubMed, Citeseer, TUDataset and [many others](https://juliaml.github.io/MLDatasets.jl/dev/datasets/graphs/).

GraphNeuralNetworks.jl provides the [`GNNGraphs.mldataset2gnngraph`](@ref) method for interfacing with MLDatasets.jl.

```@docs
GNNGraphs.mldataset2gnngraph
```
