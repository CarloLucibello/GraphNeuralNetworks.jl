<img align="right" width="300px" src="https://raw.githubusercontent.com/JuliaGraphs/GraphNeuralNetworks.jl/master/GraphNeuralNetworks/docs/src/assets/logo.svg">


# GraphNeuralNetworks.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliagraphs.org/GraphNeuralNetworks.jl/graphneuralnetworks/)
![](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/JuliaGraphs/GraphNeuralNetworks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGraphs/GraphNeuralNetworks.jl)


**Libraries for deep learning on graphs in Julia**, using either [Flux.jl](https://fluxml.ai/Flux.jl/stable/) or [Lux.jl](https://lux.csail.mit.edu/stable/) as backend frameworks.

This repository contains the following packages:

- **GraphNeuralNetworks.jl**: Provides graph convolutional layers based on the deep learning framework [Flux.jl](https://fluxml.ai/Flux.jl/stable/). This is the frontend package for Flux users.

- **GNNLux.jl**: Offers graph convolutional layers based on the deep learning framework [Lux.jl](https://lux.csail.mit.edu/stable/). This is the frontend package for Lux users. Note: This package is still under development and not yet registered.

- **GNNGraphs.jl**: Provides graph data structures and helper functions for working with graph data. This package is re-exported by the frontend packages.

- **GNNlib.jl**: Implements the message-passing framework based on the gather/scatter mechanism or sparse matrix multiplication. It also includes shared implementations for the layers used by the two frontend packages. This package is not intended for direct use by end-users but is re-exported by the frontend packages.

### Features

Both **GraphNeuralNetworks.jl** and **GNNLux.jl** support the following features:

- Implementation of common graph convolutional layers.
- Computation on batched graphs.
- Custom layer definitions.
- Support for CUDA and AMDGPU.
- Integration with [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl).
- [Examples](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/GraphNeuralNetworks/examples) of node, edge, and graph-level machine learning tasks.
- Support for heterogeneous and temporal graphs.

## Installation

**GraphNeuralNetworks.jl**, **GNNlib.jl**, and **GNNGraphs.jl** are registered Julia packages. You can install them easily through the package manager:

```julia
pkg> add GraphNeuralNetworks
```

## Usage

Usage examples can be found in the [examples folder](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/GraphNeuralNetworks/examples) and the [notebooks folder](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/GraphNeuralNetworks/notebooks). 

For a comprehensive introduction to the library, refer to the [Documentation](https://juliagraphs.org/GraphNeuralNetworks.jl/graphneuralnetworks/) and the [Tutorials](https://juliagraphs.org/GraphNeuralNetworks.jl/tutorials/)

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


