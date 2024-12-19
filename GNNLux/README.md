<img align="right" width="300px" src="https://raw.githubusercontent.com/JuliaGraphs/GraphNeuralNetworks.jl/master/docs/logo.svg">

# GNNLux.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/)

Graph convolutional layers based on the deep learning framework [Lux.jl](https://lux.csail.mit.edu/stable/). 
This is the frontend package for Lux users of the [GraphNeuralNetworks.jl ecosystem](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl).


### Features

**GNNLux.jl** supports the following features:

- Implementation of common graph convolutional layers.
- Computation on batched graphs.
- Custom layer definitions.
- Support for CUDA and AMDGPU.
- Integration with [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl).
- [Examples](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/GraphNeuralNetworks/examples) of node, edge, and graph-level machine learning tasks.
- Heterogeneous and dynamical graphs and convolutions.

## Installation  

Install through the Julia package manager.

```julia
pkg> add GNNLux
```

## Usage

For a comprehensive introduction to the library, refer to the [Documentation](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/).

## Citing

If you use GraphNeuralNetworks.jl in a scientific publication, we would appreciate a reference
to [our paper](https://arxiv.org/abs/2412.06354):

```
@article{lucibello2024graphneuralnetworks,
  title={GraphNeuralNetworks.jl: Deep Learning on Graphs with Julia},
  author={Lucibello, Carlo and Rossi, Aurora},
  journal={arXiv preprint arXiv:2412.06354},
  url={https://arxiv.org/abs/2412.06354},
  year={2024}
}
```