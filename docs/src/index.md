# GraphNeuralNetworks

This is the documentation page for the [GraphNeuralNetworks.jl](https://github.com/CarloLucibello/GraphNeuralNetworks.jl) library.

A graph neural network library for Julia based on the deep learning framework [Flux.jl](https://github.com/FluxML/Flux.jl).
Its most relevant features are:
* Provides CUDA support.
* It's integrated with the JuliaGraphs ecosystem.
* Implements many common graph convolutional layers.
* Performs fast operations on batched graphs. 
* Makes it easy to define custom graph convolutional layers.




## Package overview

### Data preparation


```
using LightGraphs

lg = LightGraphs.Graph(5) # create a light's graph graph
add_edge!(g, 1, 2)
add_edge!(g, 1, 3)
add_edge!(g, 2, 4)
add_edge!(g, 2, 5)
add_edge!(g, 3, 4)

g = GNNGraph(g)
```
### Model building 

### Training 

