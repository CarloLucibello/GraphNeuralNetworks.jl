# GraphNeuralNetworks.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://fluxml.ai/GraphNeuralNetworks.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.ai/GraphNeuralNetworks.jl/dev)
![](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/FluxML/GraphNeuralNetworks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/CarloLucibello/GraphNeuralNetworks.jl)

GraphNeuralNetworks (GNN) is a graph neural network library for Julia based on the [Flux.jl](https://github.com/FluxML/Flux.jl) deep learning framework.


## Installation

```julia
]add GraphNeuralNetworks
```

## Features

* Based on the Flux deep learning framework.
* CUDA support.
* Integrated with the JuliaGraphs ecosystem.
* Supports generic graph neural network architectures.
* Easy to define custom graph convolutional layers.

## Featured Graphs

GraphNeuralNetworks handles graph data (the graph topology + node/edge/global features)
thanks to the type `FeaturedGraph`.

A `FeaturedGraph` can be constructed out of 
adjacency matrices, adjacency lists, LightGraphs' types...

```julia
fg = FeaturedGraph(adj_list)   
```

## Graph convolutional layers

Construct a GCN layer:

```julia
GCNConv(input_dim => output_dim, relu)
```

## Usage Example

```julia
struct GNN
    conv1
    conv2 
    dense

    function GNN()
        new(GCNConv(1024=>512, relu),
            GCNConv(512=>128, relu), 
            Dense(128, 10))
    end
end

@functor GNN

function (net::GNN)(g, x)
    x = net.conv1(g, x)
    x = dropout(x, 0.5)
    x = net.conv2(g, x)
    x = net.dense(x)
    return x
end

model = GNN()

loss(x, y) = logitcrossentropy(model(fg, x), y)
accuracy(x, y) = mean(onecold(model(fg, x)) .== onecold(y))

ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM(0.01)
evalcb() = @show(accuracy(train_X, train_y))

Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
```
