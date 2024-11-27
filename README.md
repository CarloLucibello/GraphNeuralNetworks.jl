<img align="right" width="300px" src="https://raw.githubusercontent.com/JuliaGraphs/GraphNeuralNetworks.jl/master/GraphNeuralNetworks/docs/src/assets/logo.svg">


# GraphNeuralNetworks.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliagraphs.org/GraphNeuralNetworks.jl/graphneuralnetworks/)
![](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/JuliaGraphs/GraphNeuralNetworks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGraphs/GraphNeuralNetworks.jl)

Libraries for deep learning on graphs in Julia, using either [Flux.jl](https://fluxml.ai/Flux.jl/stable/) or [Lux.jl](https://lux.csail.mit.edu/stable/) as backend framework.

This monorepo contains the following packages:

- `GraphNeuralNetworks.jl`: Graph convolutional layers based on the deep learning framework [Flux.jl](https://fluxml.ai/Flux.jl/stable/). This is the fronted package for Flux users. 

- `GNNLux.jl`: Graph convolutional layers based on the deep learning framework [Lux.jl](https://lux.csail.mit.edu/stable/). This is the fronted package for Lux users. This package is still under development and it is not yet registered.

- `GNNlib.jl`: Contains the message passing framework based on the gather/scatter mechanism or on
  sparse matrix multiplication. It also contained the shared implementation for the layers of the two fronted packages. This package is not meant to be used directly by the user, but its functionalities
  are used and re-exported by the fronted packages.

- `GNNGraphs.jl`: Package that contains the graph data structures and helper functions for working with graph data. It depends on Graphs.jl package.


Both `GraphNeuralNetworks.jl` and `GNNLux.jl` enjoy several features:

* Implement common graph convolutional layers.
* Support computations on batched graphs. 
* Easy to define custom layers.
* CUDA and AMDGPU support.
* Integration with [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl).
* [Examples](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/GraphNeuralNetworks/examples) of node, edge, and graph level machine learning tasks. 
* Heterogeneous and temporal graphs support. 

## Installation

GraphNeuralNetworks.jl, GNNlib.jl and GNNGraphs.jl are a registered Julia packages. You can easily install a package, for example GraphNeuralNetworks.jl, through the package manager :

```julia
pkg> add GraphNeuralNetworks
```



## Usage

Usage examples can be found in the [examples](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/GraphNeuralNetworks/examples) and in the [notebooks](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/GraphNeuralNetworks/notebooks) folder. Also, make sure to read the [documentation](https://juliagraphs.org/GraphNeuralNetworks.jl/graphneuralnetworks/) for a comprehensive introduction to the library and the [tutorials](https://juliagraphs.org/GraphNeuralNetworks.jl/tutorials/).


## Citing

If you use GraphNeuralNetworks.jl in a scientific publication, we would appreciate the following reference:

```
@misc{Lucibello2021GNN,
  author       = {Carlo Lucibello and other contributors},
  title        = {GraphNeuralNetworks.jl: a geometric deep learning library for the Julia programming language},
  year         = 2021,
  url          = {https://github.com/JuliaGraphs/GraphNeuralNetworks.jl}
}
```

## Acknowledgments

GraphNeuralNetworks.jl is largely inspired by [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [Deep Graph Library](https://docs.dgl.ai/),
and [GeometricFlux.jl](https://fluxml.ai/GeometricFlux.jl/stable/).


