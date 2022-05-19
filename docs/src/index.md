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
using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics
using Flux.Data: DataLoader

all_graphs = GNNGraph[]

for _ in 1:1000
    g = GNNGraph(random_regular_graph(10, 4),  
            ndata=(; x = randn(Float32, 16,10)),  # input node features
            gdata=(; y = randn(Float32)))         # regression target   
    push!(all_graphs, g)
end
```

### Model building 

We concisely define our model as a [`GNNChain`](@ref) containing two graph convolutional layers. If CUDA is available, our model will live on the gpu.

```julia
device = CUDA.functional() ? Flux.gpu : Flux.cpu;

model = GNNChain(GCNConv(16 => 64),
                BatchNorm(64),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                x -> relu.(x),     
                GCNConv(64 => 64, relu),
                GlobalPool(mean),  # aggregate node-wise features into graph-wise features
                Dense(64, 1)) |> device

ps = Flux.params(model)
opt = ADAM(1f-4)
```

### Training 

Finally, we use a standard Flux training pipeline to fit our dataset.
Flux's `DataLoader` iterates over mini-batches of graphs 
(batched together into a `GNNGraph` object). 

```julia
train_size = round(Int, 0.8 * length(all_graphs))
train_loader = DataLoader(all_graphs[1:train_size], batchsize=32, shuffle=true)
test_loader = DataLoader(all_graphs[train_size+1:end], batchsize=32, shuffle=false)

loss(g::GNNGraph) = mean((vec(model(g, g.ndata.x)) - g.gdata.y).^2)

loss(loader) = mean(loss(g |> device) for g in loader)

for epoch in 1:100
    for g in train_loader
        g = g |> device
        grad = gradient(() -> loss(g), ps)
        Flux.Optimise.update!(opt, ps, grad)
    end

    @info (; epoch, train_loss=loss(train_loader), test_loss=loss(test_loader))
end
```
