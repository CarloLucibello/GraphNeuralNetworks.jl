<img align="right" width="300px" src="https://raw.githubusercontent.com/CarloLucibello/GraphNeuralNetworks.jl/master/docs/src/assets/logo.svg">


# GraphNeuralNetworks.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://CarloLucibello.github.io/GraphNeuralNetworks.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://CarloLucibello.github.io/GraphNeuralNetworks.jl/dev)
![](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/CarloLucibello/GraphNeuralNetworks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/CarloLucibello/GraphNeuralNetworks.jl)


GraphNeuralNetworks.jl is a graph neural network library written in Julia and based on the deep learning framework [Flux.jl](https://github.com/FluxML/Flux.jl).

Among its features:

* Implements common graph convolutional layers.
* Supports computations on batched graphs. 
* Easy to define custom layers.
* CUDA support.
* Integration with [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl).
* [Examples](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/examples) of node, edge, and graph level machine learning tasks. 
* Heterogeneous and temporal graphs. 

## Installation

GraphNeuralNetworks.jl is a registered Julia package. You can easily install it through the package manager:

```julia
pkg> add GraphNeuralNetworks
```

## Usage

Usage examples can be found in the [examples](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/examples) and in the [notebooks](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/notebooks) folder. Also, make sure to read the [documentation](https://CarloLucibello.github.io/GraphNeuralNetworks.jl/dev) for a comprehensive introduction to the library.


## Citing

If you use GraphNeuralNetworks.jl in a scientific publication, we would appreciate the following reference:

```
@misc{Lucibello2021GNN,
  author       = {Carlo Lucibello and other contributors},
  title        = {GraphNeuralNetworks.jl: a geometric deep learning library for the Julia programming language},
  year         = 2021,
  url          = {https://github.com/CarloLucibello/GraphNeuralNetworks.jl}
}
```

## Acknowledgments

GraphNeuralNetworks.jl is largely inspired by [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [Deep Graph Library](https://docs.dgl.ai/),
and [GeometricFlux.jl](https://fluxml.ai/GeometricFlux.jl/stable/).


