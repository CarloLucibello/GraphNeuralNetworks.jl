# GraphNeuralNetworks

This is the documentation page for the [GraphNeuralNetworks.jl](https://github.com/CarloLucibello/GraphNeuralNetworks.jl) library.

A graph neural network library for Julia based on the deep learning framework [Flux.jl](https://github.com/FluxML/Flux.jl). GNN.jl is largely inspired by python's libraries [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and [Deep Graph Library](https://docs.dgl.ai/),
and by julia's [GeometricFlux](https://fluxml.ai/GeometricFlux.jl/stable/).

Among its features:

* Integratation with the JuliaGraphs ecosystem.
* Implementation of common graph convolutional layers.
* Fast operations on batched graphs. 
* Easy to define custom layers.
* CUDA support.


## Package overview

Let's give a brief overview of the package by solving a  
graph regression problem with synthetic data. 

Usage examples on real datasets can be found in the [examples](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/examples) folder. 

### Data preparation

First, we create our dataset consisting in multiple random graphs and associated data features. 
Then we batch the graphs together into a unique graph.

```julia
julia> using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics

julia> all_graphs = GNNGraph[];

julia> for _ in 1:1000
           g = GNNGraph(random_regular_graph(10, 4),  
                       ndata=(; x = randn(Float32, 16,10)),  # input node features
                       gdata=(; y = randn(Float32)))         # regression target   
           push!(all_graphs, g)
       end

julia> gbatch = Flux.batch(all_graphs)
GNNGraph:
    num_nodes = 10000
    num_edges = 40000
    num_graphs = 1000
    ndata:
        x => (16, 10000)
    edata:
    gdata:
        y => (1000,)
```


### Model building 

We concisely define our model as a [`GNNChain`](@ref) containing 2 graph convolutaional 
layers. If CUDA is available, our model will live on the gpu.

```julia
julia> device = CUDA.functional() ? Flux.gpu : Flux.cpu;

julia> model = GNNChain(GCNConv(16 => 64),
                        BatchNorm(64),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                        x -> relu.(x),     
                        GCNConv(64 => 64, relu),
                        GlobalPool(mean),  # aggregate node-wise features into graph-wise features
                        Dense(64, 1)) |> device;

julia> ps = Flux.params(model);

julia> opt = ADAM(1f-4);
```

### Training 

Finally, we use a standard Flux training pipeling to fit our dataset.
Flux's DataLoader iterates over mini-batches of graphs 
(batched together into a `GNNGraph` object). 

```julia
gtrain = getgraph(gbatch, 1:800)
gtest = getgraph(gbatch, 801:gbatch.num_graphs)
train_loader = Flux.Data.DataLoader(gtrain, batchsize=32, shuffle=true)
test_loader = Flux.Data.DataLoader(gtest, batchsize=32, shuffle=false)

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
