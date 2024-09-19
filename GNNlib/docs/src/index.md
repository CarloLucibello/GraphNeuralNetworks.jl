# GNNlib.jl

GNNlib.jl is a package that provides the implementation of the basic message passing functions and 
functional implementation of graph convolutional layers, which are used to build graph neural networks in both the Flux.jl and Lux.jl machine learning frameworks, created in the GraphNeuralNetworks.jl and GNNLux.jl packages, respectively.

This package depends on GNNGraphs.jl and NNlib.jl, and is primarily intended for developers looking to create new GNN architectures. For most users, the higher-level GraphNeuralNetworks.jl and GNNLux.jl packages are recommended.