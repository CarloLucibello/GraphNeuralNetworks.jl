# GraphNeuralNetworks.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://CarloLucibello.github.io/GraphNeuralNetworks.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://CarloLucibello.github.io/GraphNeuralNetworks.jl/dev)
![](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/CarloLucibello/GraphNeuralNetworks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/CarloLucibello/GraphNeuralNetworks.jl)

A graph neural network library for Julia based on the deep learning framework [Flux.jl](https://github.com/FluxML/Flux.jl). Its features include:

* Integration with [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl).
* Implementation of common graph convolutional layers.
* Fast operations on batched graphs. 
* Easy to define custom layers.
* CUDA support.

## Installation

GraphNeuralNetworks.jl is a registered julia package. 
You can easily install it through the package manager:

```julia
pkg> add GraphNeuralNetworks
```

## Usage

Usage examples can be found in the [examples](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/examples) folder. Also, make sure to read the [documentation](https://CarloLucibello.github.io/GraphNeuralNetworks.jl/dev) for a comprehensive introduction to the library.

## Acknowledgements

A big thanks goes to @yuehhua for creating [GeometricFlux.jl](https://github.com/FluxML/GeometricFlux.jl) of which GraphNeuralNetworks.jl is a radical redesign. 
