# GraphNeuralNetworks.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://CarloLucibello.github.io/GraphNeuralNetworks.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://CarloLucibello.github.io/GraphNeuralNetworks.jl/dev)
![](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/CarloLucibello/GraphNeuralNetworks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/CarloLucibello/GraphNeuralNetworks.jl)

A graph neural network library for Julia based on the deep learning framework [Flux.jl](https://github.com/FluxML/Flux.jl).
Its most relevant features are:
* Provides CUDA support.
* It's integrated with the JuliaGraphs ecosystem.
* Implements many common graph convolutional layers.
* Performs fast operations on batched graphs. 
* Makes it easy to define custom graph convolutional layers.

## Installation

```julia
]add GraphNeuralNetworks
```

## Usage

Usage examples can be found in the [examples](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/examples) folder. 
