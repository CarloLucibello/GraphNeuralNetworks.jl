# GraphNeuralNetworks

This is the documentation page for [GraphNeuralNetworks.jl](https://github.com/CarloLucibello/GraphNeuralNetworks.jl), a graph neural network library written in Julia and based on the deep learning framework [Flux.jl](https://github.com/FluxML/Flux.jl).
GraphNeuralNetworks.jl is largely inspired by [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [Deep Graph Library](https://docs.dgl.ai/),
and [GeometricFlux.jl](https://fluxml.ai/GeometricFlux.jl/stable/).

Among its features:

* Implements common graph convolutional layers.
* Supports computations on batched graphs. 
* Easy to define custom layers.
* CUDA support.
* Integration with [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl).
* [Examples](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/examples) of node, edge, and graph level machine learning tasks. 


## Package overview

Let's give a brief overview of the package by solving a  
graph regression problem with synthetic data. 

Usage examples on real datasets can be found in the [examples](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/examples) folder. 

### Data preparation

We create a dataset consisting in multiple random graphs and associated data features. 

```julia
using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics, MLUtils
using Flux: DataLoader

all_graphs = GNNGraph[]

for _ in 1:1000
    g = rand_graph(10, 40,  
            ndata=(; x = randn(Float32, 16,10)),  # input node features
            gdata=(; y = randn(Float32)))         # regression target   
    push!(all_graphs, g)
end
```

### Model building 

We concisely define our model as a [`GraphNeuralNetworks.GNNChain`](@ref) containing two graph convolutional layers. If CUDA is available, our model will live on the gpu.

```julia
device = CUDA.functional() ? Flux.gpu : Flux.cpu;

model = GNNChain(GCNConv(16 => 64),
                BatchNorm(64),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                x -> relu.(x),     
                GCNConv(64 => 64, relu),
                GlobalPool(mean),  # aggregate node-wise features into graph-wise features
                Dense(64, 1)) |> device

opt = Flux.setup(Adam(1f-4), model)
```

### Training 

Finally, we use a standard Flux training pipeline to fit our dataset.
We use Flux's `DataLoader` to iterate over mini-batches of graphs 
that are glued together into a single `GNNGraph` using the `MLUtils.batch` method. This is what happens under the hood when creating a `DataLoader` with the
`collate=true` option. 

```julia
train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.8)

train_loader = DataLoader(train_graphs, 
                batchsize=32, shuffle=true, collate=true)
test_loader = DataLoader(test_graphs, 
                batchsize=32, shuffle=false, collate=true)

loss(model, g::GNNGraph) = mean((vec(model(g, g.x)) - g.y).^2)

loss(model, loader) = mean(loss(model, g |> device) for g in loader)

for epoch in 1:100
    for g in train_loader
        g = g |> device
        grad = gradient(model -> loss(model, g), model)
        Flux.update!(opt, model, grad[1])
    end

    @info (; epoch, train_loss=loss(model, train_loader), test_loss=loss(model, test_loader))
end
```
