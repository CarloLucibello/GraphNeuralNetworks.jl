# GraphNeuralNetworks Monorepo

This is the documentation page for [GraphNeuralNetworks.jl](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl), a graph neural network library written in Julia and based on the deep learning framework [Flux.jl](https://github.com/FluxML/Flux.jl).
GraphNeuralNetworks.jl is largely inspired by [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [Deep Graph Library](https://docs.dgl.ai/),
and [GeometricFlux.jl](https://fluxml.ai/GeometricFlux.jl/stable/).

- `GraphNeuralNetwork.jl`: Package that contains stateful graph convolutional layers based on the machine learning framework [Flux.jl](https://fluxml.ai/Flux.jl/stable/). This is fronted package for Flux users. It depends on GNNlib.jl, GNNGraphs.jl, and Flux.jl packages.

* Implements common graph convolutional layers.
* Supports computations on batched graphs. 
* Easy to define custom layers.
* CUDA support.
* Integration with [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl).
* [Examples](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/examples) of node, edge, and graph level machine learning tasks. 




Usage examples on real datasets can be found in the [examples](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/examples) folder. 

